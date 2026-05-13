# Overview

A semi-supervised and unified framework for nasopharyngeal carcinoma diagnosis, segmentation, and biopsy site localization.

**Training pipeline:**
```
[Round 1] Classification → Prompts Generation → Segmentation → [Round 2] Classification → Prompts Generation → Segmentation
```

**Application pipeline:**
```
[Round 2] Classification (Diagnosis) → Prompts Generation (& Biopsy Site Recommendation) → Segmentation (NPC Boundary)
```

## Checkpoints

We provide the final Round classification and segmentation checkpoints for running the Application pipeline. Readers can use the demo images in `datasets/demo/images` to reproduce and view the corresponding demo results.

Checkpoint download link:

https://drive.google.com/drive/folders/1jnXf8JbdRa4wF3wMRyQS9BXCdZLjU9qY?usp=drive_link

## Data Formats

**train (Round 1) and test list format**:
```
image-path  0 1 
```

**Round 2 train list format**:
```
image-path  0 1  pseudo-mask-path
```

**Segmentation list format**:
```
image-path,annotation-path,prompt-path
```

**Prompt format**:
```json
{ "image": "image-path", "points": [[x1,y1], [x2,y2], ...], "bbox": [x0,y0,x1,y1], "logits": [z0, z1] }
```

`point_labels` is optional. The generated prompt files from `generate_prompts.py` do not include it, and the segmentation loader will treat all provided points as foreground prompts when the field is absent.

---

## Training pipeline

### [Round 1] Classification Training

```bash
TRAIN_LIST=""        # path to Round 1 train list
VAL_LIST=""          # path to val list
TEST_LIST=""         # path to test list
sam_ckpt=""          # Init MedSAM ViT-B checkpoint
GPU_ID=0

CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
    --name mrm_r1 --stage train \
    --model vit_base_patch16 --model_type ViT-B_16 --num_classes 2 \
    --pretrained_path "${sam_ckpt}" \
    --dataset_path './datasets/' \
    --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" --test_list "${TEST_LIST}" \
    --output_dir "outputs/" --data_volume '100' \
    --num_steps 30000 --img_size 224 \
    --train_batch_size 256 --gradient_accumulation_steps 2 --eval_batch_size 256 \
    --optimizer adamw --learning_rate 1e-4 --weight_decay 0.05 --warmup_steps 2000 \
    --head_hidden_dim 512 --head_dropout 0.2
```
---

### [Round 2] Classification Training

```bash
TRAIN_LIST=""        # path to Round 2 train list
VAL_LIST=""          # path to val list
TEST_LIST=""         # path to test list
sam_ckpt=""          # Init MedSAM ViT-B checkpoint 
GPU_ID=0

CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
    --use_r_loader \
    --name mrm_r2 --stage train \
    --model vit_base_patch16 --model_type ViT-B_16 --num_classes 2 \
    --pretrained_path "${sam_ckpt}" \
    --dataset_path './datasets/' \
    --train_list "${TRAIN_LIST}" --val_list "${VAL_LIST}" --test_list "${TEST_LIST}" \
    --output_dir "outputs/" --data_volume '100' \
    --num_steps 30000 --img_size 224 \
    --train_batch_size 256 --gradient_accumulation_steps 2 --eval_batch_size 256 \
    --optimizer adamw --learning_rate 1e-4 --weight_decay 0.05 --warmup_steps 2000 \
    --head_hidden_dim 512 --head_dropout 0.2 \
    --lambda_cam 0.2 --lambda_fg 0.05 --fgbg_margin 0.1 \
    --target_class 1 --normal_class 0 \
    --lambda_r 0.3 --rollout_layers 4 \
    --use_fov_mask --fov_thresh 0.05 --fov_smooth 9
```

---

### [Round 1&2] Prompts Generation

```bash
PRETRAINED_PATH=""    # corresponding Classification checkpoint
GPU_ID=0

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 generate_prompts.py \
    --checkpoint "${PRETRAINED_PATH}" \
    --dataset_list "" # path to the prompts generation file \
    --output_dir ""   # output directory \
    --img_size 224 --num_classes 2 \
    --target_class 1 --normal_class 0 \
    --lambda_r 0.3 --delta 0.7 \
    --nms_radius_ratio 0.04 \
    --rollout_layers 4 \
    --use_fov_mask --fov_thresh 13 \
    --head_hidden_dim 512 --head_dropout 0.2 \
    --draw_gt_mask --mask_contour_color 0,0,255
```

---

### [Round 1&2] Segmentation Training

```bash
# Paths
data_root=""             # data root directory
img_folder="${data_root}"
mask_folder="${data_root}"
point_folder="${data_root}"
train_img_list=""        # path to train list
val_img_list=""          # path to val list
sam_ckpt=""              # Init MedSAM ViT-B checkpoint
dir_checkpoint=""        # output directory
GPU_ID=0

CUDA_VISIBLE_DEVICES="${GPU_ID}" python lora-SAM/train_finetune_point_box.py \
    -finetune_type lora -arch vit_b \
    -if_warmup True \
    -if_update_encoder True \
    -if_encoder_lora_layer True \
    -if_decoder_lora_layer True \
    -dataset_name naso-seg -targets combine_all -region_type fixed_pair \
    -img_folder "$img_folder" -mask_folder "$mask_folder" -point_folder "$point_folder" \
    -sam_ckpt "$sam_ckpt" \
    -dir_checkpoint "$dir_checkpoint" \
    -train_img_list "$train_img_list" -val_img_list "$val_img_list" \
    -with_boxes True -with_points True \
    -keep_one_prompt True
```

---

## Application pipeline

### Classification Test (Diagnosis)
```bash
TEST_LIST=""         
PRETRAINED_PATH=""   # Round 2 Classification checkpoint
GPU_ID=0

CUDA_VISIBLE_DEVICES="${GPU_ID}" python train.py \
    --use_r_loader \
    --name mrm_r2 --stage test \
    --model vit_base_patch16 --model_type ViT-B_16 --num_classes 2 \
    --pretrained_path "${PRETRAINED_PATH}" \
    --dataset_path './datasets/' --data_volume '100' --img_size 224 \
    --output_dir "outputs/" --eval_batch_size 256 \
    --test_list "${TEST_LIST}" \
    --head_hidden_dim 512 --head_dropout 0.2
```

---

### Final Prompts Generation & Biopsy Site Recommendation (Demo)

```bash
PRETRAINED_PATH=""   # Round 2 Classification checkpoint
GPU_ID=0

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 generate_prompts.py \
    --checkpoint "${PRETRAINED_PATH}" \
    --dataset_list datasets/test_cam_demo.txt \
    --output_dir datasets/demo/prompts-r2 \
    --img_size 224 --num_classes 2 \
    --target_class 1 --normal_class 0 \
    --lambda_r 0.3 --delta 0.7 \
    --nms_radius_ratio 0.04 \
    --rollout_layers 4 \
    --use_fov_mask --fov_thresh 13 \
    --head_hidden_dim 512 --head_dropout 0.2 \
    --draw_gt_mask --mask_contour_color 0,0,255
```

---

### Final NPC Segmentation (Demo)

```bash
sam_ckpt=""          # Init MedSAM ViT-B checkpoint
dir_checkpoint=""    # Round 2 segmentation checkpoint
data_root="../datasets/demo"
val_img_list="lora-SAM/datasets/naso-seg/seg_test.csv"
GPU_ID=0

CUDA_VISIBLE_DEVICES="${GPU_ID}" python lora-SAM/test_finetune_point_box.py \
    -finetune_type lora -arch vit_b \
    -if_encoder_lora_layer True -if_decoder_lora_layer True \
    -dataset_name naso-seg -targets combine_all -region_type fixed_pair \
    -img_folder "${data_root}" -mask_folder "${data_root}" -point_folder "${data_root}" \
    -sam_ckpt "$sam_ckpt" -dir_checkpoint "$dir_checkpoint" \
    -num_cls 2 -image_size 1024 \
    -with_boxes True -with_points True \
    -val_img_list "$val_img_list" \
    -disable_filter False \
    -eta 0.6 -alpha 0.6 -beta 4.0 -threshscore 0.4
```

---

## References

```bibtex
@article{ma2024segment,
  title={Segment anything in medical images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={654},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
