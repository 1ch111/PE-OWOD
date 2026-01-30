# PE-OWOD: Parameter-Efficient Open-World Object Detection

This is the official implementation of the paper:
**"PE-OWOD: Parameter-Efficient Open-World Object Detection via Semantic-Adapter and Virtual Outlier Synthesis"**.

## Introduction

PE-OWOD is a lightweight framework designed for Open-World Object Detection (OWOD) on resource-constrained devices. 
*   **Parameter-Efficient:** We freeze the backbone and Transformer encoder, updating only **<2%** of parameters (Residual Adapters).
*   **Virtual Outlier Synthesis (VOS):** We introduce a novel mechanism to synthesize unknown objects in the feature space, significantly improving Unknown Recall.
*   **Performance:** Achieves **64.7% U-Recall** on MS-COCO with only **2.2GB GPU memory** usage.

![Architecture]

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/PE-OWOD.git
   cd PE-OWOD

1.Install dependencies:
pip install -r requirements.txt
Note: We recommend using PyTorch >= 1.10 with CUDA support.

Data Preparation
1.Download MS-COCO 2017 dataset (train2017 and val2017) from the https://www.google.com/url?sa=E&q=https%3A%2F%2Fcocodataset.org%2F.
2.Organize the data as follows:
PE-OWOD/
├── train/            # Contains train2017 images
│   └── annotations/  # Place instances_train2017.json here
├── val/              # Contains val2017 images
└── ...
3.Generate the open-world task split:
python scripts/split_coco.py

Model Zoo
Model	                 Backbone	mAP(Known)  U-Recall	       Download
PE-OWOD (Ver 4.0)	ResNet-50 (Frozen)	21.4	64.7%	[Google Drive] / [Baidu Netdisk]
(Pre-extracted CLIP embeddings are included in clip_embeddings_40.pth)

Training
To train the model on a single GPU (e.g., RTX 4060):

python main.py \
  --coco_path train \
  --output_dir outputs \
  --batch_size 4 \
  --epochs 15 \
  --use_clip_init \
  --resume /path/to/your/pretrained_weights.pth (Optional

Evaluation
To evaluate the model and calculate U-Recall:

python scripts/eval_owod.py \
  --coco_path train \
  --resume frozen/weights_final_VOS.pth

Visualization
To visualize detection results (including Unknown objects):

python scripts/demo.py \
  --resume weights_final_VOS.pth \
  --image_path test_image.jpg

Citation
If you find this work useful, please consider citing:

@article{pe_owod2026,
  title={PE-OWOD: Parameter-Efficient Open-World Detection with Semantic Priors and Virtual Outlier Synthesis},
  author={Jiaming Gu},
  journal={arXiv preprint},
  year={2026}
}