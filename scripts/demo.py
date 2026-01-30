import argparse
import time
import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.ops import nms

from models import build_model

model_path = r"/PE-OWOD/frozen/weights_final_VOS.pth"

image_path = r"/PE-OWOD/pkq2.png"

ENERGY_THRESHOLD = 12.0
KNOWN_ENERGY_TH = 16.0
KNOWN_SCORE_TH = 0.2
NMS_TH = 0.3

COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

COLOR_KNOWN = (0, 255, 0)
COLOR_UNKNOWN = (0, 0, 255)

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_args_parser():
    parser = argparse.ArgumentParser('Demo', add_help=False)
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
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--distributed', default=False)
    return parser


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def main(args):
    device = torch.device(args.device)

    print(f"Loading model from {model_path} ...")
    model, _, _ = build_model(args)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    if not os.path.exists(image_path):
        print(f"Error: The image file cannot be found-> {image_path}")
        return

    print(f"Reading image: {image_path}")
    im_pil = Image.open(image_path).convert('RGB')
    img = transform(im_pil).unsqueeze(0).to(device)

    print("Inference...")
    start = time.time()
    with torch.no_grad():
        outputs = model(img)
    print(f"Done in {time.time() - start:.3f}s")

    logits = outputs['pred_logits'][0]
    bboxes = outputs['pred_boxes'][0]
    probas = logits.softmax(-1)[:, :-1]
    scores, labels = probas.max(-1)
    bboxes_scaled = rescale_bboxes(bboxes.cpu(), im_pil.size)
    im_cv = cv2.cvtColor(np.array(im_pil), cv2.COLOR_RGB2BGR)

    known_boxes, known_scores, known_lbls = [], [], []
    unknown_boxes, unknown_scores, unknown_energies = [], [], []

    img_h, img_w = im_pil.size[1], im_pil.size[0]
    img_area = img_h * img_w

    for i in range(100):

        logit = logits[i]
        energy = -torch.logsumexp(logit, dim=-1).item()
        score = scores[i].item()
        cl = labels[i].item()
        box = bboxes_scaled[i]

        x0, y0, x1, y1 = box.numpy()
        box_area = (x1 - x0) * (y1 - y0)

        if box_area > 0.6 * img_area: continue
        if box_area < 0.005 * img_area: continue


        if energy < KNOWN_ENERGY_TH and score > KNOWN_SCORE_TH:
            if cl < 40:
                known_boxes.append(box)
                known_scores.append(score)
                known_lbls.append(cl)

        elif energy < ENERGY_THRESHOLD and score < 0.5:
            unknown_boxes.append(box)
            unknown_scores.append(-energy)
            unknown_energies.append(energy)

    print(f"find {len(known_boxes)} known objects，{len(unknown_boxes)} Potentially unknown objects。")

    if len(known_boxes) > 0:
        known_boxes_t = torch.stack(known_boxes)
        known_scores_t = torch.tensor(known_scores)
        keep = nms(known_boxes_t, known_scores_t, NMS_TH)

        for idx in keep:
            box = known_boxes[idx].numpy().astype(int)
            sc = known_scores[idx]
            cl = known_lbls[idx]

            label_text = f"{COCO_CLASSES[cl]} {sc:.2f}"
            print(f"Known: {label_text}")
            cv2.rectangle(im_cv, (box[0], box[1]), (box[2], box[3]), COLOR_KNOWN, 4)
            cv2.putText(im_cv, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, COLOR_KNOWN, 8)

    if len(unknown_boxes) > 0:
        unknown_boxes_t = torch.stack(unknown_boxes)
        unknown_scores_t = torch.tensor(unknown_scores)
        keep = nms(unknown_boxes_t, unknown_scores_t, NMS_TH)

        for idx in keep:
            box = unknown_boxes[idx].numpy().astype(int)
            en = unknown_energies[idx]
            label_text = f"Unknown E:{en:.1f}"
            print(f"Unknown: {label_text}")
            cv2.rectangle(im_cv, (box[0], box[1]), (box[2], box[3]), COLOR_UNKNOWN, 4)
            cv2.putText(im_cv, label_text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, COLOR_UNKNOWN, 8)

    cv2.imwrite("demo_result_final.jpg", im_cv)
    print("The picture has been saved to demo_result_final.jpg")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)