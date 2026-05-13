# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import random
import numpy as np
from datetime import timedelta
import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from apex import amp
from apex.parallel import DistributedDataParallel as DDP

from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader, get_loader_r1
from utils.dist_util import get_world_size
import collections

import torch.optim as optim
import models_vit
from medsam import ImageEncoderViT
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CLASS_NAMES = ['Non-Npc', 'Npc']

def _load_test_list_entries(args):
    list_path = args.test_list
    if not os.path.isabs(list_path):
        list_path = os.path.join(args.dataset_path, list_path)
    entries = []
    try:
        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                fname = parts[0]
                labels = parts[1:]
                entries.append((fname, labels))
    except FileNotFoundError:
        logger.warning("Test list not found: %s", list_path)
    return entries

def _write_test_results(args, entries, pred_indices):
    if not entries:
        logger.warning("No test list entries, skip writing result.txt")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    out_name = "result-r2.txt" if args.use_r_loader else "result-r1.txt"
    out_path = os.path.join(args.output_dir, out_name)
    pred_indices = np.array(pred_indices).reshape(-1).astype(int)
    n = min(len(entries), len(pred_indices))
    if len(entries) != len(pred_indices):
        logger.warning("Entry count (%d) != pred count (%d); writing first %d lines.",
                       len(entries), len(pred_indices), n)
    with open(out_path, "w") as f:
        for i in range(n):
            fname, labels = entries[i]
            if len(labels) < args.num_classes:
                labels = labels + ["0"] * (args.num_classes - len(labels))
            labels = labels[:args.num_classes]
            pred_onehot = ["1" if j == pred_indices[i] else "0" for j in range(args.num_classes)]
            f.write(" ".join([fname] + labels + pred_onehot) + "\n")
    logger.info("Saved test results to %s", out_path)
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, roc_auc_score, classification_report
from sklearn.metrics import auc as auc_test
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, Isomap, TSNE, LocallyLinearEmbedding
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy import interp
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.utils import resample

def ci95(accuracy_scores):
    # bootstrap 95% confidence interval
    alpha = 0.95
    p_lower = ((1.0 - alpha) / 2.0) * 100
    p_upper = (alpha + ((1.0 - alpha) / 2.0)) * 100
    lower = np.percentile(accuracy_scores, p_lower)
    upper = np.percentile(accuracy_scores, p_upper)
    # logger.info(f"95% CI: ({lower}, {upper})")
    return lower, upper

def roc_auc_cal(y_test, y_score, n_classes):
    # compute per-class ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    roc_auc_value = 0
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc_test(fpr[i], tpr[i])
        roc_auc_value += roc_auc[i]
    roc_auc_value /= n_classes 
    return roc_auc_value

def data_analysis(all_target, all_output, all_output_softmax):
    # y_test = all_target
    # y_test_binarize = label_binarize(y_test, classes=[0, 1, 2])
    y_test_binarize = all_target
    y_test = torch.topk(torch.tensor(y_test_binarize), 1).indices.numpy().squeeze()
    y_pred = all_output
    # logger.info('y_test:{}, y_pred:{}, y_pred_softmax: {}'.format(y_test, y_pred, all_output_softmax))
    ## Evaluate the model
    confusion_mat = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    # Sensitivity (TPR) and Specificity (TNR) for binary classification
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0
    f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_cal(y_test_binarize, all_output_softmax, all_output_softmax.shape[1])
    # roc_auc = roc_auc_score(y_test, all_output_softmax, multi_class='ovr')
    # roc_auc = roc_auc_score(y_test, all_output_softmax, multi_class='ovo')
    # confidence intervals
    n_iterations = 1000  # number of bootstrap resamples
    n_size = len(y_test)
    accuracy_scores = []
    sensitivity_scores = []
    specificity_scores = []
    f1_scores = []
    rocauc_scores = []
    for i in range(n_iterations):
        # resample with replacement
        y_true_resample, y_pred_resample, y_true_binarize_resample, all_output_softmax_resample = resample(y_test, y_pred, y_test_binarize, all_output_softmax, n_samples=n_size)
        # compute metrics on resampled data
        accuracy_one = accuracy_score(y_true_resample, y_pred_resample)
        accuracy_scores.append(accuracy_one)
        cm_one = confusion_matrix(y_true_resample, y_pred_resample, labels=[0, 1])
        if cm_one.shape == (2, 2):
            tn1, fp1, fn1, tp1 = cm_one.ravel()
            sensitivity_one = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0.0
            specificity_one = tn1 / (tn1 + fp1) if (tn1 + fp1) > 0 else 0.0
        else:
            sensitivity_one = 0.0
            specificity_one = 0.0
        sensitivity_scores.append(sensitivity_one)
        specificity_scores.append(specificity_one)
        f1_one = f1_score(y_true_resample, y_pred_resample, average='macro')
        f1_scores.append(f1_one)
        roc_auc_one = roc_auc_cal(y_true_binarize_resample, all_output_softmax_resample, all_output_softmax_resample.shape[1])
        rocauc_scores.append(roc_auc_one)
   
    accurace_ci95_min, accurace_ci95_max = ci95(accuracy_scores)
    sensitivity_ci95_min, sensitivity_ci95_max = ci95(sensitivity_scores)
    specificity_ci95_min, specificity_ci95_max = ci95(specificity_scores)
    f1_ci95_min, f1_ci95_max = ci95(f1_scores)
    roc_auc_ci95_min, roc_auc_ci95_max = ci95(rocauc_scores)
    logger.info('accuracy: {}({}-{}), sensitivity: {}({}-{}), specificity: {}({}-{}), f1: {}({}-{}), roc_auc: {}({}-{})'.format(
        accuracy, accurace_ci95_min, accurace_ci95_max,
        sensitivity, sensitivity_ci95_min, sensitivity_ci95_max,
        specificity, specificity_ci95_min, specificity_ci95_max,
        f1, f1_ci95_min, f1_ci95_max,
        roc_auc, roc_auc_ci95_min, roc_auc_ci95_max))
    
    ## figure: confusion matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(confusion_mat, annot=True, fmt='.0f', linewidths=.5, square = True, cmap = 'Blues')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix', size = 15)
    plt.savefig('./Confusion-Matrix.jpg')

    ## figure: Compute the ROC curve and AUC score
    # Ensure label matrix matches score columns (skip double-binarization)
    y_score = np.array(all_output_softmax)
    if y_test_binarize.ndim == 1 or y_test_binarize.shape[-1] == 1:
        y_test = label_binarize(y_test, classes=range(y_score.shape[1]))
    else:
        y_test = np.array(y_test_binarize)
        if y_test.shape[1] != y_score.shape[1]:
            raise ValueError(f"Label columns ({y_test.shape[1]}) must match score columns ({y_score.shape[1]})")
    n_classes = y_score.shape[1]
    # compute per-class ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc_test(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc_test(fpr["micro"], tpr["micro"])
    
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc_test(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('./Receiver-Operating-Characteristic.jpg')

def auc(pred_property_array, one_hot_labels, num_classes):
    AUROCs = []
    # pdb.set_trace()
    for i in range(num_classes):
        AUROCs.append(roc_auc_score(one_hot_labels[:, i], pred_property_array[:, i]))
    # print(AUROCs)
    return AUROCs

def simple_accuracy(preds, labels):
    # print(preds)
    # print(labels)
    return ((preds == labels) * 1).mean()

def classification_report(preds, labels):
    return classification_report(labels,preds)

def save_model_auc(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "{}_bestauc_checkpoint_{}.bin".format(args.name, args.data_volume))
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def save_model_loss(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_bestloss_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def resize_pos_embed(pos_embed_ckpt, target_shape):
    # ckpt: [1, Hc, Wc, C], target: [1, Ht, Wt, C]
    pos = pos_embed_ckpt.permute(0, 3, 1, 2)
    pos = F.interpolate(pos, size=(target_shape[1], target_shape[2]), mode="bicubic", align_corners=False)
    return pos.permute(0, 2, 3, 1)

def resize_rel_pos(rel_pos_ckpt, target_len):
    # ckpt: [L, C], target_len: 2*size-1
    rel = rel_pos_ckpt.transpose(0, 1).unsqueeze(0)  # [1, C, L]
    rel = F.interpolate(rel, size=target_len, mode="linear", align_corners=False)
    return rel.squeeze(0).transpose(0, 1)

def setup(args):
    # Prepare model
    num_classes = args.num_classes
    if args.model_type == "ViT-B_16":
        model = ImageEncoderViT(
            pretrained=args.pretrained_path,
            num_classes=num_classes,
            global_pool=True,
            head_hidden_dim=args.head_hidden_dim,
            head_dropout=args.head_dropout,
        )
        if args.stage=='train':
            checkpoint = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
            checkpoint_model = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            model_dict = model.state_dict()
            # load pre-trained model
            print('load pre-trained model from: {}'.format(args.pretrained_path))
            state_dict_old = checkpoint_model
            state_dict_new = collections.OrderedDict()
            for key_init, v in state_dict_old.items():
                if 'image_encoder.' in key_init:
                    k = key_init[14:]
                else:
                    k = key_init
                if 'mask_encoder.' in key_init or 'neck.' in key_init or 'mask_decoder.' in key_init or 'prompt_encoder.' in key_init:
                    continue
                if k not in model_dict:
                    print('ignore key {}'.format(k))
                    continue
                if v.shape != model_dict[k].shape:
                    if k == "pos_embed" and v.dim() == 4:
                        v = F.interpolate(v.permute(0,3,1,2), size=(model_dict[k].shape[1], model_dict[k].shape[2]), mode="bicubic", align_corners=False).permute(0,2,3,1)
                    elif "rel_pos" in k and v.dim() == 2:
                        v = resize_rel_pos(v, model_dict[k].shape[0])
                if v.shape == model_dict[k].shape:
                    state_dict_new[k] = v
                else:
                    print('key {} with diff shape: {}, {}'.format(k, v.shape, model_dict[k].shape))
            msg = model.load_state_dict(state_dict_new, strict=False)
            print(msg)
            # manually initialize fc layer
            model.init_head_weights()
        else:
            pretrained_weights = torch.load(args.pretrained_path, map_location=torch.device('cpu'))
            model.load_state_dict(pretrained_weights,  strict=True)

    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def get_data_loader(args):
    return get_loader_r1(args) if args.use_r_loader else get_loader(args)

def unpack_batch(batch):
    if len(batch) == 2:
        x, y = batch
        return x, y, None
    if len(batch) == 3:
        x, y, mask = batch
        return x, y, mask
    raise ValueError("Unexpected batch size: {}".format(len(batch)))

def valid(args, model, writer, test_loader, global_step):
# def valid(args, model, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    all_property = []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    # loss_fct = torch.nn.CrossEntropyLoss()
    loss_fct = torch.nn.BCEWithLogitsLoss(reduction="mean")
    
    for step, batch in enumerate(epoch_iterator):
        # if step > 10:  # debug code 
        #     break
        batch = tuple(t.to(args.device) for t in batch)
        x, y, _ = unpack_batch(batch)
        with torch.no_grad():
            logits = model(x)
            eval_loss = loss_fct(logits, y.float())
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1) # argmax over class dimension
            # preds = (logits.sigmoid() > 0.5) * 1

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            all_property.append(logits.sigmoid().detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            all_property[0] = np.append(
                all_property[0], logits.sigmoid().detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label, all_property = all_preds[0], all_label[0], all_property[0]
    if global_step > 550000:
        data_analysis(np.array(all_label).squeeze(), np.array(all_preds).squeeze(), np.array(all_property).squeeze())

    # accuracy = simple_accuracy(all_preds, all_label)
    all_label_indices = torch.topk(torch.tensor(np.array(all_label).squeeze()), 1).indices.numpy().squeeze()
    accuracy = accuracy_score(all_label_indices, np.array(all_preds).squeeze())

    aurocs = auc(all_property, all_label, args.num_classes)
    auroc_avg = np.array(aurocs).mean()

    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %.4f" % eval_losses.avg)
    logger.info("Valid accuracy: %.4f" % accuracy)
    logger.info("Valid Auc: %.4f" % auroc_avg)

    # writer.add_scalar("valid/loss", scalar_value=eval_losses.avg, global_step=global_step)
    # writer.add_scalar("valid/accuracy", scalar_value=accuracy, global_step=global_step)
    # writer.add_scalar("valid/auc", scalar_value=auroc_avg, global_step=global_step)
    return auroc_avg, eval_losses.avg

def test(args, model, test_loader):
    # Test!
    eval_losses = AverageMeter()
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    all_property = []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    # loss_fct = torch.nn.CrossEntropyLoss()
    loss_fct = torch.nn.BCEWithLogitsLoss(reduction="mean")

    for step, batch in enumerate(epoch_iterator):
        # if step > 10:  # debug code 
        #     break
        batch = tuple(t.to(args.device) for t in batch)
        x, y, _ = unpack_batch(batch)
        with torch.no_grad():
            # if "net" in args.model_type:
            #     logits = model(x)#[0]
            # else :
            #     logits = model(x)[0]
            logits = model(x)
            eval_loss = loss_fct(logits, y.float())
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1) # argmax over class dimension
            # preds = (logits.sigmoid() > 0.5) * 1

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            all_property.append(logits.sigmoid().detach().cpu().numpy())

        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            all_property[0] = np.append(
                all_property[0], logits.sigmoid().detach().cpu().numpy(), axis=0
            )

        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label_binarize, all_property = all_preds[0], all_label[0], all_property[0]
    # data_analysis(np.array(all_label).squeeze(), np.array(all_preds).squeeze(), np.array(all_property).squeeze())
    # np.save('all_preds.npy', all_preds)
    # np.save('all_label.npy', all_label)
    # np.save('all_property.npy', all_property)
    # all_preds = np.load('all_preds.npy')
    # all_label = np.load('all_label.npy')
    # all_property = np.load('all_property.npy')
    # data_analysis(np.array(all_label).squeeze(), np.array(all_preds).squeeze(), np.array(all_property).squeeze())

    data_analysis(np.array(all_label_binarize).squeeze(), np.array(all_preds).squeeze(), np.array(all_property).squeeze())
    # write result.txt in output_dir
    entries = _load_test_list_entries(args)
    _write_test_results(args, entries, all_preds)

def compute_cam_pair(args, model, x, target_class, normal_class, blocks):
    features = []

    def fwd_hook(_, __, output):
        features.append(output)

    handles = []
    for idx in blocks:
        layer = model.blocks[idx]
        handles.append(layer.register_forward_hook(fwd_hook))

    logits = model(x)
    target_logit = logits[:, target_class].sum()
    normal_logit = logits[:, normal_class].sum()
    grads_t = torch.autograd.grad(target_logit, features, retain_graph=True, create_graph=False)
    grads_n = torch.autograd.grad(normal_logit, features, retain_graph=True, create_graph=False)

    cams_t = []
    cams_n = []
    for feat, grad_t, grad_n in zip(features, grads_t, grads_n):
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
    return logits, cam_t, cam_n


def cam_mask_losses(args, cam, mask):
    mask_resized = F.interpolate(mask, size=cam.shape[-2:], mode="bilinear", align_corners=False)
    mask_resized = mask_resized.clamp(0, 1)
    if mask_resized.dim() == 4 and mask_resized.size(1) == 1:
        mask_resized = mask_resized[:, 0]

    cam_flat = cam.view(cam.size(0), -1)
    mask_flat = mask_resized.view(mask_resized.size(0), -1)
    cam_norm = cam_flat.norm(p=2, dim=1)
    mask_norm = mask_flat.norm(p=2, dim=1)
    valid = mask_norm > args.cam_mask_eps

    cos = (cam_flat * mask_flat).sum(dim=1) / (cam_norm * mask_norm + 1e-8)
    loss_cam = torch.where(valid, 1.0 - cos, torch.zeros_like(cos))

    fg_mean = (cam * mask_resized).sum(dim=(1, 2)) / (mask_resized.sum(dim=(1, 2)) + 1e-8)
    bg_mean = (cam * (1.0 - mask_resized)).sum(dim=(1, 2)) / ((1.0 - mask_resized).sum(dim=(1, 2)) + 1e-8)
    loss_fgbg = F.relu(bg_mean - fg_mean + args.fgbg_margin)
    loss_fgbg = torch.where(valid, loss_fgbg, torch.zeros_like(loss_fgbg))

    return loss_cam, loss_fgbg, valid


def build_fov_mask_from_input(args, x):
    mean = torch.tensor([0.4978], device=x.device).view(1, 1, 1, 1)
    std = torch.tensor([0.2449], device=x.device).view(1, 1, 1, 1)
    img = x[:, :1] * std + mean
    mask = (img > args.fov_thresh).float()
    if args.fov_smooth > 1:
        k = args.fov_smooth
        pad = k // 2
        mask = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    return mask

def train(args, model):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "logs"))  #  tensorboard Supporting documents, in logs/name/

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_data_loader(args)
    
    # Prepare optimizer and scheduler
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizers = amp.initialize(
            models=model,
            optimizers=[optimizer],
            opt_level=args.fp16_opt_level
        )
        optimizer = optimizers[0]
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_cam = AverageMeter()
    losses_fgbg = AverageMeter()
    losses_cls_npc = AverageMeter()
    losses_cls_npc_valid = AverageMeter()
    losses_cam_npc = AverageMeter()
    losses_fgbg_npc = AverageMeter()
    global_step, best_auc, best_loss = 0, 0, float("inf")
    last_auc = 10
    down = 0
    loss_fct = torch.nn.BCEWithLogitsLoss(reduction="none" if args.use_r_loader else "mean")
    if args.use_r_loader:
        if not hasattr(model, "blocks"):
            raise ValueError("R1 training requires a model with transformer blocks for CAM extraction.")
        start_block = max(0, len(model.blocks) - args.rollout_layers)
        blocks = list(range(start_block, len(model.blocks)))
        if not blocks:
            raise ValueError("rollout_layers must be >= 1 for CAM extraction.")

    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y, mask = unpack_batch(batch)
            if args.use_r_loader:
                logits, cam_t, cam_n = compute_cam_pair(args, model, x, args.target_class, args.normal_class, blocks)
                cam = F.relu(cam_t - args.lambda_r * cam_n)
                if args.use_fov_mask:
                    fov_mask = build_fov_mask_from_input(args, x)
                    fov_mask = F.interpolate(fov_mask, size=cam.shape[-2:], mode="bilinear", align_corners=False)
                    cam = cam * fov_mask.squeeze(1)
                cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
                loss_raw = loss_fct(logits.view(-1, args.num_classes), y.float())
                if loss_raw.dim() == 2:
                    loss_cls_per = loss_raw.mean(dim=1)
                elif loss_raw.dim() == 1:
                    loss_cls_per = loss_raw
                else:
                    loss_cls_per = loss_raw.view(-1)
                if y.dim() == 1:
                    is_npc = (y > 0.5).float()
                else:
                    is_npc = (y[:, args.target_class] > 0.5).float()
                weights = 1.0 + (args.npc_loss_weight - 1.0) * is_npc
                loss_cls = (loss_cls_per * weights).mean()
                loss_cam_vec, loss_fgbg_vec, valid_cam = cam_mask_losses(args, cam, mask)
                loss_cam = (loss_cam_vec * weights).mean()
                loss_fgbg = (loss_fgbg_vec * weights).mean()
                loss = loss_cls + args.lambda_cam * loss_cam + args.lambda_fg * loss_fgbg
            else:
                logits = model(x)
                loss = loss_fct(logits.view(-1, args.num_classes), y.float())
                loss_cls = loss
                loss_cam = None
                loss_fgbg = None
                loss_cls_per = None
                is_npc = None
                valid_cam = None
                loss_cam_vec = None
                loss_fgbg_vec = None
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.use_r_loader:
                    losses_cls.update(loss_cls.item())
                    losses_cam.update(loss_cam.item())
                    losses_fgbg.update(loss_fgbg.item())

                    npc_mask = is_npc > 0.5
                    npc_count = int(npc_mask.sum().item())
                    if npc_count > 0:
                        losses_cls_npc.update(loss_cls_per[npc_mask].mean().item(), n=npc_count)
                    valid_npc = npc_mask & valid_cam
                    valid_npc_count = int(valid_npc.sum().item())
                    if valid_npc_count > 0:
                        losses_cls_npc_valid.update(loss_cls_per[valid_npc].mean().item(), n=valid_npc_count)
                        losses_cam_npc.update(loss_cam_vec[valid_npc].mean().item(), n=valid_npc_count)
                        losses_fgbg_npc.update(loss_fgbg_vec[valid_npc].mean().item(), n=valid_npc_count)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    if args.use_r_loader:
                        writer.add_scalar("train/loss_cls", scalar_value=loss_cls.item(), global_step=global_step)
                        writer.add_scalar("train/loss_cam", scalar_value=loss_cam.item(), global_step=global_step)
                        writer.add_scalar("train/loss_fgbg", scalar_value=loss_fgbg.item(), global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                    if args.use_r_loader:
                        writer.add_scalar("train/loss_cls_avg", scalar_value=losses_cls.avg, global_step=global_step)
                        writer.add_scalar("train/loss_cam_avg", scalar_value=losses_cam.avg, global_step=global_step)
                        writer.add_scalar("train/loss_fgbg_avg", scalar_value=losses_fgbg.avg, global_step=global_step)
                        cam_denom = losses_cam.avg + 1e-8
                        fg_denom = losses_fgbg.avg + 1e-8
                        lambda_cam_eq = losses_cls.avg / cam_denom
                        lambda_fg_eq = losses_cls.avg / fg_denom
                        writer.add_scalar("train/lambda_cam_eq", scalar_value=lambda_cam_eq, global_step=global_step)
                        writer.add_scalar("train/lambda_fg_eq", scalar_value=lambda_fg_eq, global_step=global_step)
                        writer.add_scalar("train/lambda_cam_0p2", scalar_value=0.2 * lambda_cam_eq, global_step=global_step)
                        writer.add_scalar("train/lambda_fg_0p05", scalar_value=0.05 * lambda_fg_eq, global_step=global_step)
                        writer.add_scalar("train/loss_cls_npc_avg", scalar_value=losses_cls_npc.avg, global_step=global_step)
                        writer.add_scalar("train/loss_cls_npc_valid_avg", scalar_value=losses_cls_npc_valid.avg, global_step=global_step)
                        writer.add_scalar("train/loss_cam_npc_avg", scalar_value=losses_cam_npc.avg, global_step=global_step)
                        writer.add_scalar("train/loss_fgbg_npc_avg", scalar_value=losses_fgbg_npc.avg, global_step=global_step)
                        if losses_cam_npc.count > 0 and losses_cls_npc_valid.count > 0:
                            cam_denom_npc = losses_cam_npc.avg + 1e-8
                            fg_denom_npc = losses_fgbg_npc.avg + 1e-8
                            lambda_cam_eq_npc = losses_cls_npc_valid.avg / cam_denom_npc
                            lambda_fg_eq_npc = losses_cls_npc_valid.avg / fg_denom_npc
                            writer.add_scalar("train/lambda_cam_eq_npc", scalar_value=lambda_cam_eq_npc, global_step=global_step)
                            writer.add_scalar("train/lambda_fg_eq_npc", scalar_value=lambda_fg_eq_npc, global_step=global_step)
                            writer.add_scalar("train/lambda_cam_npc_0p2", scalar_value=0.2 * lambda_cam_eq_npc, global_step=global_step)
                            writer.add_scalar("train/lambda_fg_npc_0p05", scalar_value=0.05 * lambda_fg_eq_npc, global_step=global_step)
                
                len_train = len(train_loader)
                if global_step % len_train == 0 and args.local_rank in [-1, 0]:
                    
                    auroc_avg, val_loss = valid(args, model, writer, test_loader, global_step)
                    # writer.add_scalar("auroc", scalar_value=auroc_avg, global_step=global_step)
                    if auroc_avg >= best_auc:
                        best_auc = auroc_avg
                        save_model_auc(args, model)
                        down = 0
                    else:
                        down = down + 1
                    print(down)
                    # track best (lowest) validation loss optionally
                    if val_loss < best_loss:
                        best_loss = val_loss

                if global_step % t_total == 0:
                    break
        losses.reset()
        losses_cls.reset()
        losses_cam.reset()
        losses_fgbg.reset()
        losses_cls_npc.reset()
        losses_cls_npc_valid.reset()
        losses_cam_npc.reset()
        losses_fgbg_npc.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    
    logger.info("End Training!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # Required parameters
    parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")

    parser.add_argument("--stage", type=str, default="train", help="train or test?")
    
    parser.add_argument("--model_type", choices=["ViT-B_16", "Resnet50", "Resnet18","Resnet101","Densenet121"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--num_classes",default = 3,type=int,help="the number of class")                    
    parser.add_argument("--head_hidden_dim", default=0, type=int, help="hidden size for optional MLP head (0 keeps single linear)")
    parser.add_argument("--head_dropout", default=0.0, type=float, help="dropout applied inside optional MLP head")
    parser.add_argument("--pretrained_path", type=str, default="checkpoint/ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument("--test_list", default="test.txt", type=str,
                        help="Filename (relative to dataset_path) for test split list.")
    parser.add_argument("--train_list", default="train-v8.txt", type=str,
                        help="Filename (relative to dataset_path) for train split list.")
    parser.add_argument("--val_list", default="val-v5.txt", type=str,
                        help="Filename (relative to dataset_path) for val split list.")

    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=512, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps."
                             "Will always run one evaluation at the end of training.")

    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="adamw",
                        help="Optimizer to use for fine-tuning.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="Base learning rate for the optimizer.")
    parser.add_argument("--weight_decay", default=5e-2, type=float,
                        help="Weight decay for regularization.")
    parser.add_argument("--num_steps", default=10000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--data_volume", type=str)
    parser.add_argument("--gpu", type=str)

    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--use_r_loader", action="store_true",
                        help="Use the R1 dataset loader and CAM-guided training losses.")
    parser.add_argument("--lambda_cam", default=0.5, type=float,
                        help="Weight for CAM-mask cosine loss.")
    parser.add_argument("--lambda_fg", default=0.2, type=float,
                        help="Weight for foreground-background margin loss.")
    parser.add_argument("--fgbg_margin", default=0.1, type=float,
                        help="Margin for foreground-background separation loss.")
    parser.add_argument("--target_class", default=1, type=int,
                        help="Index of lesion/NPC class for CAM extraction.")
    parser.add_argument("--normal_class", default=0, type=int,
                        help="Index of normal/background class for CAM suppression.")
    parser.add_argument("--lambda_r", default=0.3, type=float,
                        help="Suppression weight for normal-class CAM.")
    parser.add_argument("--rollout_layers", default=1, type=int,
                        help="Use last N blocks for CAM extraction.")
    parser.add_argument("--cam_mask_eps", default=1e-6, type=float,
                        help="Epsilon for treating empty masks.")
    parser.add_argument("--use_fov_mask", action="store_true",
                        help="Mask CAM to endoscopic field-of-view to suppress black borders.")
    parser.add_argument("--fov_thresh", default=0.05, type=float,
                        help="Intensity threshold for FOV mask.")
    parser.add_argument("--fov_smooth", default=9, type=int,
                        help="MaxPool kernel for smoothing FOV mask.")
    parser.add_argument("--npc_loss_weight", default=1.0, type=float,
                        help="Extra loss weight for NPC samples.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--dataset_path", type=str)

    args = parser.parse_args()
    if args.local_rank == -1:
        env_local_rank = os.environ.get("LOCAL_RANK")
        if env_local_rank is not None:
            args.local_rank = int(env_local_rank)

    os.environ["OMP_NUM_THREADS"] = "1"

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        print('##############################')   
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                            timeout=timedelta(minutes=60)
                                            )
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model = setup(args)

    if args.stage == "train":
        # Training
        train(args, model)
    else :
        test_loader = get_data_loader(args)
        test(args, model, test_loader)
        # valid(args, model, test_loader, 60000)


if __name__ == "__main__":
    main()
