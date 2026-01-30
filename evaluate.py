import torch
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# 引入 DETR 模块
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
from datasets.coco_eval import CocoEvaluator


model_path = r"/PE-OWOD/frozen/weights_final_VOS.pth"

coco_path = r"/PE-OWOD/train"

KNOWN_IDS = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44
}

ENERGY_THRESHOLD = 20.0
KNOWN_CONF_THRESHOLD = 0.1



def get_args_parser():
    parser = argparse.ArgumentParser('Full Eval', add_help=False)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--coco_path', type=str, default=coco_path)
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--device', default='cuda')


    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--aux_loss', action='store_true')

    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--distributed', default=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)  # 补丁
    return parser


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou


@torch.no_grad()
def main(args):
    device = torch.device(args.device)

    print("loading model...")
    model, criterion, postprocessors = build_model(args)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    print("loading dataset...")
    dataset_val = build_dataset(image_set='val', args=args)
    data_loader = DataLoader(dataset_val, args.batch_size, shuffle=False,
                             collate_fn=utils.collate_fn, num_workers=args.num_workers)

    base_ds = get_coco_api_from_dataset(dataset_val)
    iou_types = tuple(k for k in ('bbox', 'segm') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    total_unknown = 0
    detected_unknown = 0
    a_ose_errors = 0

    print("Start Evaluation...")

    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        coco_evaluator.update(res)

        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        energy = -torch.logsumexp(pred_logits, dim=-1)
        probas = pred_logits.softmax(-1)[:, :, :-1]
        max_scores, max_labels = probas.max(-1)

        for b in range(len(targets)):
            img_h, img_w = targets[b]['size']
            tgt_labels = targets[b]['labels']
            tgt_boxes = targets[b]['boxes']

            t_boxes_abs = box_cxcywh_to_xyxy(tgt_boxes) * torch.tensor([img_w, img_h, img_w, img_h], device=device)
            p_boxes_abs = box_cxcywh_to_xyxy(pred_boxes[b]) * torch.tensor([img_w, img_h, img_w, img_h], device=device)

            is_unknown_gt = torch.tensor([l.item() not in KNOWN_IDS for l in tgt_labels], device=device)
            if is_unknown_gt.sum() == 0:
                continue

            unknown_gt_boxes = t_boxes_abs[is_unknown_gt]
            total_unknown += len(unknown_gt_boxes)

            pred_unknown_mask = energy[b] < ENERGY_THRESHOLD
            pred_unknown_boxes = p_boxes_abs[pred_unknown_mask]

            if len(pred_unknown_boxes) > 0:
                iou_mat = box_iou(unknown_gt_boxes, pred_unknown_boxes)

                detected = (iou_mat.max(dim=1)[0] > 0.5).sum().item()
                detected_unknown += detected

            pred_known_mask = max_scores[b] > KNOWN_CONF_THRESHOLD
            pred_known_boxes = p_boxes_abs[pred_known_mask]

            if len(pred_known_boxes) > 0:

                iou_mat_err = box_iou(unknown_gt_boxes, pred_known_boxes)

                errors = (iou_mat_err.max(dim=1)[0] > 0.5).sum().item()
                a_ose_errors += errors

    print("\n" + "=" * 50)
    print("Final Results Summary")
    print("=" * 50)

    # 1. mAP
    print("[1] Known Class Performance (mAP):")
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # 2. U-Recall
    u_recall = (detected_unknown / total_unknown * 100) if total_unknown > 0 else 0
    print(f"\n[2] Unknown Discovery (U-Recall):")
    print(f"Total Unknown GT: {total_unknown}")
    print(f"Detected: {detected_unknown}")
    print(f"U-Recall: {u_recall:.2f}%")

    # 3. A-OSE
    print(f"\n[3] Absolute Open-Set Error (A-OSE):")
    print(f"A-OSE: {a_ose_errors} (Lower is better)")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)