import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from tqdm import tqdm
import json
import torch.nn.functional as F 

from models.sam import sam_model_registry
from models.sam_LoRa import LoRA_Sam
import cfg 
from torch.utils.data import DataLoader
from utils.dataset import Public_dataset
import random
import cv2

def largest_cc_ratio(mask_np: np.ndarray) -> float:
    from skimage.measure import label
    labeled = label(mask_np > 0, connectivity=2)
    if labeled.max() == 0:
        return 0.0
    areas = [(labeled == i).sum() for i in range(1, labeled.max() + 1)]
    largest = max(areas)
    return float(largest) / float((mask_np > 0).sum())

def circularity(mask_np: np.ndarray) -> float:
    import cv2
    from skimage.measure import label
    labeled = label(mask_np > 0, connectivity=2)
    if labeled.max() == 0:
        return 1e6
    areas = [(labeled == i).sum() for i in range(1, labeled.max() + 1)]
    largest_idx = int(np.argmax(areas)) + 1
    lcc_mask = (labeled == largest_idx).astype(np.uint8)
    contours, _ = cv2.findContours(lcc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1e6
    area = float(cv2.contourArea(contours[0]))
    peri = float(cv2.arcLength(contours[0], True))
    if area == 0:
        return 1e6
    return (peri ** 2) / (4.0 * np.pi * area)

def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return 0.0 if union == 0 else float(inter) / float(union)

def jitter_points(points: torch.Tensor, max_offset: float = 5.0) -> torch.Tensor:
    noise = (torch.rand_like(points) - 0.5) * 2.0 * max_offset
    return torch.clamp(points + noise, min=0.0, max=1023.0)


def build_prompt_circle(points: np.ndarray, star_radius: float, padding: float = 2.0):
    if len(points) == 0:
        return None, None
    if len(points) == 1:
        return points[0], star_radius + padding

    max_dist = -1.0
    farthest_pair = (points[0], points[1])
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            if dist > max_dist:
                max_dist = dist
                farthest_pair = (points[i], points[j])

    center = (farthest_pair[0] + farthest_pair[1]) / 2.0
    max_center_dist = max(np.linalg.norm(p - center) for p in points)
    radius = max_center_dist + star_radius + padding
    return center, radius


def keep_largest_cc(mask_np: np.ndarray, min_area_ratio: float = 0.005,
                    close_r: int = 5, open_r: int = 3, hole_area: int = 2048) -> np.ndarray:
    from skimage.measure import label
    from skimage.morphology import binary_closing, binary_opening, remove_small_holes, disk

    labeled = label(mask_np > 0, connectivity=2)
    if labeled.max() == 0:
        return mask_np
    areas = [(labeled == i).sum() for i in range(1, labeled.max() + 1)]
    largest_idx = int(np.argmax(areas)) + 1
    H, W = mask_np.shape
    if areas[largest_idx - 1] < H * W * min_area_ratio:
        return np.zeros_like(mask_np, dtype=np.uint8)

    m = (labeled == largest_idx)
    m = remove_small_holes(m, area_threshold=hole_area)
    m = binary_closing(m, disk(close_r))
    m = binary_opening(m, disk(open_r))
    return m.astype(np.uint8)


def keep_largest_cc_v0(mask_np: np.ndarray, min_area_ratio: float = 0.005) -> np.ndarray:
    from skimage.measure import label
    labeled = label(mask_np > 0, connectivity=2)
    if labeled.max() == 0:
        return mask_np
    areas = [(labeled == i).sum() for i in range(1, labeled.max() + 1)]
    largest_idx = int(np.argmax(areas)) + 1
    H, W = mask_np.shape
    if areas[largest_idx - 1] < H * W * min_area_ratio:
        return np.zeros_like(mask_np, dtype=np.uint8)
    return (labeled == largest_idx).astype(np.uint8)

def scope_mask_from_image(img_np: np.ndarray) -> np.ndarray:
    gray = img_np.mean(axis=2)
    scope = (gray > 13).astype(np.uint8)  # non-black region
    scope = keep_largest_cc(scope, min_area_ratio=0.01)
    return scope

def infer_once(sam_fine_tune, imgs, points, point_labels, boxes, with_points=True, with_boxes=True, image_size=1024):
    with torch.no_grad():
        img_emb = sam_fine_tune.image_encoder(imgs)
        sparse_emb, dense_emb = sam_fine_tune.prompt_encoder(
            points=(points, point_labels) if with_points else None,
            boxes=boxes if with_boxes else None,
            masks=None,
        )
        logits, iou_predictions = sam_fine_tune.mask_decoder(
            image_embeddings=img_emb,
            image_pe=sam_fine_tune.prompt_encoder.get_dense_pe(), 
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb, 
            multimask_output=True,
        )
    best_idx = torch.argmax(iou_predictions[0]).item()
    logits_best = logits[:, best_idx:best_idx+1]  # [B,1,H,W]
    logits_full_res = F.interpolate(
        logits_best,
        size=(image_size, image_size),
        mode='bilinear',
        align_corners=False
    )
    if logits_full_res.shape[1] == 1:
        pred_mask = (torch.sigmoid(logits_full_res) > 0.5).long().cpu().squeeze(0).squeeze(0).numpy().astype(np.uint8)
    else:
        pred_mask = logits_full_res.argmax(dim=1).cpu().squeeze(0).numpy().astype(np.uint8)
    best_score = torch.sigmoid(iou_predictions[0, best_idx]).item()
    return pred_mask, best_score

def main(args, test_csv_path):
    args.with_boxes = getattr(args, "with_boxes", True)
    args.with_points = getattr(args, "with_points", True)
    save_folder = os.path.join('test_results', 'demo', args.dir_checkpoint)
    mask_save_folder = os.path.join(save_folder, "pseudo_masks")
    visualization_save_folder = os.path.join(save_folder, "visualization")
    Path(mask_save_folder).mkdir(parents=True, exist_ok=True)
    Path(visualization_save_folder).mkdir(parents=True, exist_ok=True)
    
    test_dataset = Public_dataset(
        args,
        args.img_folder,
        args.mask_folder,
        args.point_folder,
        test_csv_path,
        phase='val',
        targets=[args.targets],
        normalize_type='sam',
        if_prompt=True,
        prompt_type='point',
        region_type=args.region_type,
        delete_empty_masks=False,
    )
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    if args.finetune_type == 'adapter' or args.finetune_type == 'vanilla':
        sam_fine_tune = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.dir_checkpoint,'checkpoint_best.pth'),num_classes=args.num_cls)
    elif args.finetune_type == 'lora':
        sam = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.sam_ckpt),num_classes=args.num_cls)
        sam_fine_tune = LoRA_Sam(args,sam,r=4).to('cuda').sam
        sam_fine_tune.load_state_dict(torch.load(args.dir_checkpoint + '/checkpoint_best.pth'), strict = False)
    elif args.finetune_type == 'full':
        sam_fine_tune = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.sam_ckpt),num_classes=args.num_cls)
        sam_fine_tune.load_state_dict(torch.load(args.dir_checkpoint + '/checkpoint_best.pth'), strict = True)

    sam_fine_tune = sam_fine_tune.to('cuda').eval()

    CONFIDENCE_THRESHOLD = getattr(args, "eta", 0.6)
    ALPHA = getattr(args, "alpha", 0.6)  # LCC dominance
    BETA = getattr(args, "beta", 4.0)    # shape regularity
    THRESH_SCORE = getattr(args, "threshscore", 0.4)    # filter prediction score
    WITH_FILTER = getattr(args, "disable_filter", False)
    print(f"--- eta={CONFIDENCE_THRESHOLD}, alpha={ALPHA}, beta={BETA}, threshscore={THRESH_SCORE}, disable_filter={WITH_FILTER} ---")
    with_points = getattr(args, "with_points", True)
    with_boxes = getattr(args, "with_boxes", True)
    print("with_points: {}, with_boxes: {}".format(with_points, with_boxes))

    for data in tqdm(testloader):
        imgs = data['image'].cuda()
        img_names = data['img_name']
        base_name = os.path.basename(img_names[0])
        file_name_no_ext = os.path.splitext(base_name)[0]

        orig_img_path = os.path.join(args.img_folder, img_names[0].strip())
        original_pil_img = Image.open(orig_img_path).convert('RGB')
        img_np = np.array(original_pil_img)

        points = data['point_coords'].cuda()
        point_labels = data['point_labels'].cuda()
        boxes = data['boxes'].cuda()

        mask_a, score_a = infer_once(
            sam_fine_tune, imgs, points, point_labels, boxes,
            with_points=with_points, with_boxes=with_boxes, image_size=args.image_size
        )
        points_jitter = jitter_points(points.clone(), max_offset=8.0)
        mask_b, score_b = infer_once(
            sam_fine_tune, imgs, points_jitter, point_labels, boxes,
            with_points=with_points, with_boxes=with_boxes, image_size=args.image_size
        )

        scope_mask = scope_mask_from_image(img_np)
        scope_mask_resized = np.array(
            Image.fromarray(scope_mask * 255).resize((args.image_size, args.image_size), resample=Image.NEAREST)
        ) // 255
        mask_a = keep_largest_cc(mask_a) * scope_mask_resized
        mask_b = keep_largest_cc(mask_b) * scope_mask_resized

        iou_ab = iou_binary(mask_a, mask_b)
        lcc_ratio = largest_cc_ratio(mask_a)
        circ = circularity(mask_a)
        u_i = iou_ab * (1.0 if lcc_ratio > ALPHA else 0.0) * (1.0 if circ < BETA else 0.0)
        best_score = max(score_a, score_b)
        keep_mask = WITH_FILTER or (u_i > CONFIDENCE_THRESHOLD and best_score > THRESH_SCORE)
        if not keep_mask:
            print("lcc_ratio(>{}): {}, circ(<{}): {}, iou_ab/u_i({}): {}/{}, best_score({}): {}".format(ALPHA, lcc_ratio, BETA, circ, CONFIDENCE_THRESHOLD, iou_ab, u_i, THRESH_SCORE, best_score))
            continue

        
        pred_mask_np_0_255 = mask_a * 255
        pil_mask = Image.fromarray(pred_mask_np_0_255, 'L').resize(original_pil_img.size, resample=Image.NEAREST)

        mask_save_path = os.path.join(mask_save_folder, f"{file_name_no_ext}_pred_mask.png")
        meta_save_path = os.path.join(mask_save_folder, f"{file_name_no_ext}_meta.json")
        pil_mask.save(mask_save_path)
        meta = {
            "score_a": score_a,
            "score_b": score_b,
            "iou_ab": iou_ab,
            "lcc_ratio": lcc_ratio,
            "circularity": circ,
            "u_i": u_i,
            "eta": CONFIDENCE_THRESHOLD,
            "alpha": ALPHA,
            "beta": BETA,
            "keep": keep_mask
        }
        with open(meta_save_path, "w") as f:
            json.dump(meta, f, indent=2)

        
        pil_mask_vis = pil_mask.resize(original_pil_img.size, resample=Image.NEAREST)
        pred_mask_overlay_np = np.array(pil_mask_vis)
        masked_pred_overlay = np.ma.masked_where(pred_mask_overlay_np == 0, pred_mask_overlay_np)

        pos_points = data['point_coords'][0].numpy()
        pos_labels = data['point_labels'][0].numpy()
        scale_x = original_pil_img.size[0] / args.image_size
        scale_y = original_pil_img.size[1] / args.image_size
        pos_points_arr = pos_points[pos_labels == 1] * np.array([scale_x, scale_y])
        neg_points_arr = pos_points[pos_labels == 0] * np.array([scale_x, scale_y])

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 10))

        # Left panel: original image with prompt points
        ax_left.imshow(img_np)
        ax_left.set_title("Original Image + Prompts", fontsize=14)
        if len(pos_points_arr) > 0:
            ax_left.scatter(
                pos_points_arr[:, 0], pos_points_arr[:, 1],
                color='#00E400', marker='*', s=300,
                edgecolor='#007A00', linewidth=1.5, zorder=10,
            )
        if len(neg_points_arr) > 0:
            ax_left.scatter(
                neg_points_arr[:, 0], neg_points_arr[:, 1],
                color='#FF0000', marker='X', s=250,
                edgecolor='white', linewidth=2, zorder=10,
            )
        ax_left.axis('off')

        # Right panel: prediction overlay with GT contour and prompt
        ax_right.imshow(img_np)
        ax_right.set_title("Prediction Overlay", fontsize=14)
        ax_right.imshow(masked_pred_overlay, cmap='spring', alpha=0.6, vmin=0, vmax=255)
        if 'mask' in data:
            gt_mask_np = data['mask'][0].squeeze().numpy().astype(np.uint8)
            if gt_mask_np.sum() > 0:
                gt_mask_vis = Image.fromarray(gt_mask_np).resize(original_pil_img.size, resample=Image.NEAREST)
                gt_mask_vis_np = np.array(gt_mask_vis).astype(np.uint8)
                contours_gt, _ = cv2.findContours(gt_mask_vis_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours_gt:
                    cnt = cnt.squeeze()
                    if len(cnt.shape) == 2 and cnt.shape[0] > 1:
                        ax_right.plot(cnt[:,0], cnt[:,1], color='red', linewidth=6, alpha=0.9, linestyle='-', label='GT Boundary')

        if len(pos_points_arr) > 0:
            prompt_star_radius = 8.0
            prompt_star_color = '#00E400'
            prompt_circle_edge_color = '#007A00'
            prompt_circle_fill_color = '#00E400'

            circle_center, circle_radius = build_prompt_circle(
                pos_points_arr, star_radius=prompt_star_radius, padding=10.0
            )
            if circle_center is not None:
                prompt_circle = Circle(
                    circle_center, radius=circle_radius,
                    facecolor=prompt_circle_fill_color,
                    edgecolor=prompt_circle_edge_color,
                    linewidth=2.4, alpha=0.25, zorder=8,
                )
                ax_right.add_patch(prompt_circle)

            ax_right.scatter(
                pos_points_arr[:, 0], pos_points_arr[:, 1],
                color=prompt_star_color, marker='*', s=300,
                edgecolor=prompt_circle_edge_color, linewidth=1.5,
                zorder=10,
            )

        if len(neg_points_arr) > 0:
            ax_right.scatter(
                neg_points_arr[:, 0], neg_points_arr[:, 1],
                color='#FF0000',
                marker='X',
                s=250,
                edgecolor='white',
                linewidth=2,
                zorder=10,
                label='Negative Prompt'
            )

        ax_right.axis('off')

        plt.tight_layout()
        
        vis_save_path = os.path.join(visualization_save_folder, f"{file_name_no_ext}_visualization.png")
        plt.savefig(vis_save_path, bbox_inches='tight', pad_inches=0.1, dpi=150) 
        plt.close(fig)



if __name__ == "__main__":
    args = cfg.parse_args()
    test_csv_list_path = args.val_img_list 

    main(args, test_csv_list_path)
