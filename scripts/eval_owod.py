import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

import util.misc as utils
from util.box_ops import box_cxcywh_to_xyxy, box_iou
from datasets import build_dataset
from models import build_model


model_path = r"/PE-OWOD/frozen/weights_final_VOS.pth"

coco_path = r"/PE-OWOD/train"

BATCH_SIZE = 4
NUM_WORKERS = 2
ENERGY_THRESHOLD = 20.0
IOU_THRESHOLD = 0.5

KNOWN_IDS = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44
}


def get_args_parser():
    parser = argparse.ArgumentParser('Recall Eval', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
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
    parser.add_argument('--masks', action='store_true')
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
    parser.add_argument('--coco_path', type=str, default=coco_path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--distributed', default=False)
    return parser


def main(args):
    device = torch.device(args.device)

    print(f"Loading model from {model_path} ...")
    model, _, _ = build_model(args)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    print("Loading Validation Set (val2017)...")
    dataset_val = build_dataset(image_set='val', args=args)
    data_loader = DataLoader(dataset_val, BATCH_SIZE, shuffle=False,
                             collate_fn=utils.collate_fn, num_workers=NUM_WORKERS)

    total_unknown_gt_count = 0
    detected_unknown_count = 0

    print(f"Start the assessment U-Recall (Energy Threshold={ENERGY_THRESHOLD}, IoU={IOU_THRESHOLD})...")

    for samples, targets in tqdm(data_loader):
        samples = samples.to(device)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(samples)

        pred_logits = outputs['pred_logits']  # [B, 100, 41]
        pred_boxes = outputs['pred_boxes']  # [B, 100, 4] (cxcywh, 0-1 relative)

        for b in range(len(targets)):
            target = targets[b]
            img_h, img_w = target['size']

            tgt_labels = target['labels']

            is_unknown = torch.tensor([l.item() not in KNOWN_IDS for l in tgt_labels], device=device)

            if is_unknown.sum() == 0:
                continue

            tgt_unknown_boxes = target['boxes'][is_unknown]  # [N_unk, 4]

            tgt_unknown_boxes = box_cxcywh_to_xyxy(tgt_unknown_boxes)
            tgt_unknown_boxes = tgt_unknown_boxes * torch.tensor([img_w, img_h, img_w, img_h], device=device)

            total_unknown_gt_count += len(tgt_unknown_boxes)


            energy = -torch.logsumexp(pred_logits[b], dim=-1)  # [100]


            pred_mask = energy < ENERGY_THRESHOLD

            if pred_mask.sum() == 0:
                continue

            pred_unknown_boxes = pred_boxes[b][pred_mask]  # [M, 4]

            pred_unknown_boxes = box_cxcywh_to_xyxy(pred_unknown_boxes)
            pred_unknown_boxes = pred_unknown_boxes * torch.tensor([img_w, img_h, img_w, img_h], device=device)


            iou_matrix = box_iou(tgt_unknown_boxes, pred_unknown_boxes)[0]


            max_ious, _ = iou_matrix.max(dim=1)


            detected_count = (max_ious > IOU_THRESHOLD).sum().item()
            detected_unknown_count += detected_count


    print("\n" + "=" * 40)
    print(f"Evaluation result (Weights: {os.path.basename(model_path)})")
    print("=" * 40)
    print(f"Total Unknown Objects (GT): {total_unknown_gt_count}")
    print(f"Detected Unknown Objects  : {detected_unknown_count}")

    if total_unknown_gt_count > 0:
        recall = (detected_unknown_count / total_unknown_gt_count) * 100
        print(f"U-Recall: {recall:.2f}%")
    else:
        print("No unknown class objects were found in the verification set")
    print("=" * 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)