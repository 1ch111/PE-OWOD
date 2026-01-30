import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
import argparse
from pathlib import Path
from tqdm import tqdm

model_path = r"/PE-OWOD/frozen/weights_final_VOS.pth"

coco_path = r"/PE-OWOD/train"

KNOWN_CLASSES = set(range(1, 41))


def get_args_parser():
    parser = argparse.ArgumentParser('Analysis', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float, help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default=coco_path)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def compute_energy(logits):
    return -torch.logsumexp(logits, dim=-1)


def main(args):
    device = torch.device(args.device)

    print("Loading the model...")
    model, _, _ = build_model(args)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.eval()

    print("The validation set is being loaded...")
    dataset_val = build_dataset(image_set='val', args=args)
    data_loader_val = DataLoader(dataset_val, args.batch_size, shuffle=False,
                                 collate_fn=utils.collate_fn, num_workers=args.num_workers)

    known_energy = []
    unknown_energy = []
    background_energy = []

    print("Start reasoning and analysis (only run the first 200 batches)...")
    for i, (samples, targets) in enumerate(tqdm(data_loader_val)):
        if i > 200: break

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(samples)

        logits = outputs['pred_logits']
        energy = compute_energy(logits)

        prob = logits.softmax(-1)
        scores, labels = prob[..., :-1].max(-1)

        for b in range(len(targets)):
            tgt_labels = targets[b]['labels']
            pred_scores = scores[b]
            pred_energy = energy[b]

            foreground_mask = pred_scores > 0.3
            if foreground_mask.sum() > 0:
                e_vals = pred_energy[foreground_mask].cpu().numpy()

                known_energy.extend(e_vals)

            background_mask = pred_scores < 0.1
            if background_mask.sum() > 0:
                bg_vals = pred_energy[background_mask].cpu().numpy()
                np.random.shuffle(bg_vals)
                background_energy.extend(bg_vals[:10])

    plt.figure(figsize=(10, 6))
    plt.hist(background_energy, bins=50, alpha=0.5, label='Background (Low Conf)', color='gray', density=True)
    plt.hist(known_energy, bins=50, alpha=0.5, label='Foreground (High Conf)', color='blue', density=True)

    plt.title("Energy Distribution (Preliminary)")
    plt.xlabel("Energy Score (Lower is more confident)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("fig_energy_dist_final.png")
    print("Completed，The picture has been saved as fig_energy_dist_final.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)