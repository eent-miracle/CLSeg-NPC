import argparse
import json
import os
from typing import Tuple, Dict, Any, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from medsam import ImageEncoderViT


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to fine-tuned classifier (.bin).")
    parser.add_argument("--dataset_list", required=True, type=str, help="Txt file with image paths (one per line).")
    parser.add_argument("--output_dir", required=True, type=str, help="Where to save prompts and CAMs.")
    parser.add_argument("--img_size", default=224, type=int, help="Input resolution used for the classifier.")
    parser.add_argument("--num_classes", default=2, type=int, help="Number of classes in the classifier.")
    parser.add_argument("--target_class", default=1, type=int, help="Index of the lesion class.")
    parser.add_argument("--normal_class", default=0, type=int, help="Index of the normal/background class.")
    parser.add_argument("--head_hidden_dim", default=0, type=int, help="Hidden dim for MLP head (match training).")
    parser.add_argument("--head_dropout", default=0.0, type=float, help="Dropout for MLP head (match training).")
    parser.add_argument("--lambda_r", default=0.3, type=float, help="Suppression weight for normal CAM.")
    parser.add_argument("--delta", default=0.7, type=float, help="Threshold (fraction of max) for mask.")
    parser.add_argument("--nms_radius_ratio", default=0.04, type=float, help="Min separation ratio for NMS peaks.")
    parser.add_argument("--num_points", default=4, type=int, help="Number of foreground points to output.")
    parser.add_argument("--rollout_layers", default=1, type=int, help="Use last N blocks for Grad-CAM (default last block only).")
    parser.add_argument("--draw_gt_mask", action="store_true", help="Draw GT/pseudo mask contour if mask path is provided.")
    parser.add_argument("--mask_contour_color", default="0,0,255", type=str,
                        help="Contour color in B,G,R (e.g., 0,255,0).")
    parser.add_argument("--use_fov_mask", action="store_true", help="Mask CAM to endoscopic field-of-view.")
    parser.add_argument("--fov_thresh", default=13, type=int, help="Threshold for FOV mask in grayscale (0-255).")
    parser.add_argument("--only_npc", action="store_true", help="Only process samples with NPC label in the list.")
    return parser.parse_args()


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.CenterCrop((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4978], std=[0.2449]),
    ])


def estimate_fov_mask(image_bgr: np.ndarray, thresh: int = 13) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.ones_like(gray, dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
    return mask


def load_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageEncoderViT(
        global_pool=True,
        num_classes=args.num_classes,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()
    return model, device


def resolve_input_path(path: str) -> str:
    if os.path.exists(path):
        return path
    legacy_pairs = [
        ("/eent/add-test/images/", "/eent/images-addtest/"),
        ("/eent/add-test/annotations/", "/eent/annotations-addtest/"),
    ]
    for old, new in legacy_pairs:
        if old in path:
            mapped = path.replace(old, new)
            if os.path.exists(mapped):
                return mapped
    return path


def read_image(path: str, transform):
    img = Image.open(path).convert("RGB")
    tensor = transform(img)
    return tensor.unsqueeze(0), np.array(img)


def compute_cam_pair(model, x, target_idx: int, normal_idx: int, blocks: List[int]):
    """
    Compute Grad-CAM for target and normal classes using autograd.grad.
    This matches training-time CAM extraction and avoids backward-hook ordering issues.
    """
    features = []

    def fwd_hook(_, __, output):
        features.append(output)

    handles = []
    for idx in blocks:
        layer = model.blocks[idx]
        handles.append(layer.register_forward_hook(fwd_hook))

    logits = model(x)
    target_logit = logits[:, target_idx].sum()
    normal_logit = logits[:, normal_idx].sum()
    grads_t = torch.autograd.grad(target_logit, features, retain_graph=True, create_graph=False)
    grads_n = torch.autograd.grad(normal_logit, features, retain_graph=True, create_graph=False)

    cams_t = []
    cams_n = []
    for feat, grad_t, grad_n in zip(features, grads_t, grads_n):
        # feat/grad: [B, H, W, C]
        weights_t = grad_t.mean(dim=(1, 2), keepdim=True).detach()
        weights_n = grad_n.mean(dim=(1, 2), keepdim=True).detach()
        cam_t = F.relu((feat * weights_t).sum(-1))
        cam_n = F.relu((feat * weights_n).sum(-1))
        cams_t.append(cam_t)
        cams_n.append(cam_n)

    for h in handles:
        h.remove()

    cam_t = sum(cams_t) / len(cams_t)
    cam_n = sum(cams_n) / len(cams_n)
    cam_t = cam_t - cam_t.amin(dim=(1, 2), keepdim=True)
    cam_t = cam_t / (cam_t.amax(dim=(1, 2), keepdim=True) + 1e-8)
    cam_n = cam_n - cam_n.amin(dim=(1, 2), keepdim=True)
    cam_n = cam_n / (cam_n.amax(dim=(1, 2), keepdim=True) + 1e-8)
    return logits.detach(), cam_t.detach(), cam_n.detach()


def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)


def gaussian_refine(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask
    x_c, y_c = xs.mean(), ys.mean()
    sigma_x = max(xs.std(), 1.0)
    sigma_y = max(ys.std(), 1.0)
    yy, xx = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing="ij")
    gauss = np.exp(-((xx - x_c) ** 2) / (2 * sigma_x ** 2) - ((yy - y_c) ** 2) / (2 * sigma_y ** 2))
    gauss = gauss / (gauss.max() + 1e-8)
    refined = (mask.astype(float) * gauss)
    refined = refined / (refined.max() + 1e-8)
    return refined


def extract_prompts(cam: np.ndarray, delta: float, nms_radius_ratio: float, num_points: int) -> Dict[str, Any]:
    h, w = cam.shape
    mask = (cam > cam.max() * delta).astype(np.uint8)
    mask = largest_component(mask)
    radius = max(1, int(nms_radius_ratio * min(h, w)))
    if mask.max() == 0:
        # Empty CAM: warn upstream but still return center points and full-image bbox
        cx, cy = w // 2, h // 2
        points = [(cx, cy) for _ in range(num_points)]
        bbox = [0, 0, w - 1, h - 1]
        return {"points": points, "bbox": bbox, "mask": mask, "refined": cam, "empty_cam": True}

    refined = gaussian_refine(mask)
    # select up to 4 peaks with simple NMS
    flat_idx = np.argsort(refined.ravel())[::-1]
    points = []
    for idx in flat_idx:
        if len(points) >= num_points:
            break
        y, x = np.unravel_index(idx, refined.shape)
        if refined[y, x] <= 0:
            continue
        if mask[y, x] == 0:
            continue
        too_close = False
        for px, py in points:
            if (px - x) ** 2 + (py - y) ** 2 < radius ** 2:
                too_close = True
                break
        if not too_close:
            points.append((int(x), int(y)))  # (x, y)

    if len(points) < num_points:
        # Fill with highest remaining points (ignore NMS distance)
        for idx in flat_idx:
            if len(points) >= num_points:
                break
            y, x = np.unravel_index(idx, refined.shape)
            if refined[y, x] <= 0:
                break
            if mask[y, x] == 0:
                continue
            if (int(x), int(y)) in points:
                continue
            points.append((int(x), int(y)))
        # If still not enough (very small mask), fallback to cam peaks
        if len(points) < num_points:
            cam_flat = np.argsort(cam.ravel())[::-1]
            for idx in cam_flat:
                if len(points) >= num_points:
                    break
                y, x = np.unravel_index(idx, cam.shape)
                if cam[y, x] <= 0:
                    break
                if (int(x), int(y)) in points:
                    continue
                points.append((int(x), int(y)))
        if len(points) < num_points:
            points.extend([points[0]] * (num_points - len(points)))

    ys, xs = np.nonzero(mask)
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    dilate_x = int(0.03 * w)
    dilate_y = int(0.03 * h)
    bbox = [max(0, xmin - dilate_x), max(0, ymin - dilate_y),
            min(w - 1, xmax + dilate_x), min(h - 1, ymax + dilate_y)]

    return {"points": points, "bbox": bbox, "mask": mask, "refined": refined}


def save_cam(cam: np.ndarray, save_path: str):
    cam_norm = cam - cam.min()
    cam_norm = cam_norm / (cam_norm.max() + 1e-8)
    cam_uint8 = np.uint8(255 * cam_norm)
    cv2.imwrite(save_path, cam_uint8)


def save_overlay(cam: np.ndarray, image_bgr: np.ndarray, save_path: str):
    cam_norm = cam - cam.min()
    cam_norm = cam_norm / (cam_norm.max() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.5, heatmap, 0.5, 0)
    cv2.imwrite(save_path, overlay)


def save_overlay_with_prompts(cam: np.ndarray, image_bgr: np.ndarray, bbox, points, save_path: str,
                              mask: np.ndarray = None, contour_color=(0, 255, 0)):
    cam_norm = cam - cam.min()
    cam_norm = cam_norm / (cam_norm.max() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_norm), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_bgr, 0.5, heatmap, 0.5, 0)
    # scale drawing thickness proportionally to image size (reference: xy ~479px short edge)
    h, w = image_bgr.shape[:2]
    scale = min(h, w) / 479.0
    contour_thick = max(1, round(6 * scale))
    box_thick = max(1, round(2 * scale))
    pt_radius = max(1, round(4 * scale))
    if mask is not None:
        mask_bin = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cv2.drawContours(overlay, contours, -1, contour_color, contour_thick)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), box_thick)
    for (px, py) in points:
        cv2.circle(overlay, (px, py), pt_radius, (0, 255, 0), -1)
    cv2.imwrite(save_path, overlay)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    transform = build_transform(args.img_size)
    model, device = load_model(args)
    with open(args.dataset_list, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    start_block = max(0, len(model.blocks) - args.rollout_layers)
    blocks = list(range(start_block, len(model.blocks)))
    if len(blocks) == 0:
        raise ValueError("rollout_layers must be >= 1 for CAM extraction.")
    contour_color = tuple(int(v) for v in args.mask_contour_color.split(","))
    for idx, img_path in enumerate(lines):
        # support optional label and optional mask path after image path
        line_items = img_path.split()
        path_only = line_items[0]
        mask_path = None
        label_items = []
        if len(line_items) >= 2:
            last_tok = line_items[-1]
            if ("/" in last_tok) or ("\\" in last_tok) or ("." in last_tok):
                label_items = line_items[1:-1]
                mask_path = last_tok
            else:
                label_items = line_items[1:]
        if args.only_npc and len(label_items) > 0:
            try:
                labels = [int(v) for v in label_items]
                if args.target_class >= len(labels) or labels[args.target_class] != 1:
                    continue
            except ValueError:
                pass
        image_path = resolve_input_path(path_only)
        x, img_np = read_image(image_path, transform)
        x = x.to(device)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        logits, cam_npc, cam_norm = compute_cam_pair(model, x, args.target_class, args.normal_class, blocks)

        cam_npc = cam_npc[0].cpu().numpy()
        cam_norm = cam_norm[0].cpu().numpy()
        cam = np.clip(cam_npc - args.lambda_r * cam_norm, 0, None)
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC)
        cam_for_prompt = cam_resized
        if args.use_fov_mask:
            fov_mask = estimate_fov_mask(img_bgr, thresh=args.fov_thresh)
            cam_for_prompt = cam_resized * (fov_mask.astype(np.float32) / 255.0)
        prompts = extract_prompts(cam_for_prompt, args.delta, args.nms_radius_ratio, args.num_points)
        if prompts.get("empty_cam", False):
            print(f"[WARN] Empty CAM after thresholding; using center points and full-image bbox: {path_only}")

        # build name as parent__child__filename (two-level directories + filename)
        filename_no_ext = os.path.splitext(os.path.basename(path_only))[0]
        parent = os.path.basename(os.path.dirname(path_only))
        grandparent = os.path.basename(os.path.dirname(os.path.dirname(path_only)))
        parts = [p for p in [grandparent, parent, filename_no_ext] if p]
        base = "__".join(parts) if parts else filename_no_ext
        # cam_path = os.path.join(args.output_dir, f"{base}_cam.png")
        # cam_overlay_path = os.path.join(args.output_dir, f"{base}_overlay.png")
        cam_overlay_prompt_path = os.path.join(args.output_dir, f"{filename_no_ext}_overlay_prompt.png")
        # mask_path = os.path.join(args.output_dir, f"{base}_mask.png")
        # save_cam(cam_resized, cam_path)
        # save_overlay(cam_resized, img_bgr, cam_overlay_path)
        bbox_int = None
        if prompts["bbox"] is not None:
            bbox_int = [int(b) for b in prompts["bbox"]]
        points_int = [(int(px), int(py)) for px, py in prompts.get("points", [])]
        mask_np = None
        resolved_mask_path = resolve_input_path(mask_path) if mask_path is not None else None
        if args.draw_gt_mask and resolved_mask_path is not None and os.path.exists(resolved_mask_path):
            try:
                mask_img = Image.open(resolved_mask_path).convert("L")
                if mask_img.size != (img_np.shape[1], img_np.shape[0]):
                    mask_img = mask_img.resize((img_np.shape[1], img_np.shape[0]), resample=Image.NEAREST)
                mask_np = np.array(mask_img)
            except Exception:
                mask_np = cv2.imread(resolved_mask_path, cv2.IMREAD_GRAYSCALE)
                if mask_np is not None and (mask_np.shape[1] != img_np.shape[1] or mask_np.shape[0] != img_np.shape[0]):
                    mask_np = cv2.resize(mask_np, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
        save_overlay_with_prompts(cam_resized, img_bgr, bbox_int, points_int, cam_overlay_prompt_path,
                                  mask=mask_np, contour_color=contour_color)
        # if prompts["mask"] is not None:
        #     cv2.imwrite(mask_path, (prompts["mask"] * 255).astype(np.uint8))

        # ensure all values are plain Python types for JSON serialization
        logits_list = [float(v) for v in logits.flatten().cpu().tolist()]
        bbox = prompts.get("bbox")
        if bbox is not None:
            bbox = [int(b) for b in bbox]
        points = [(int(px), int(py)) for px, py in prompts.get("points", [])]

        out_json = {
            "image": path_only,
            "points": points,
            "bbox": bbox,
            # "cam_path": cam_path,
            # "overlay_path": cam_overlay_prompt_path,
            # "mask_path": mask_path if prompts["mask"] is not None else None,
            "logits": logits_list,
        }
        with open(os.path.join(args.output_dir, f"{filename_no_ext}_prompt.json"), "w") as jf:
            json.dump(out_json, jf, indent=2)

        if idx % 10 == 0:
            print(f"[{idx}/{len(lines)}] processed {path_only}")


if __name__ == "__main__":
    main()
