# PE-OWOD: Parameter-Efficient Open-World Detection with Semantic Priors and Virtual Outlier Synthesis

This is the official implementation of the paper:
"PE-OWOD: Parameter-Efficient Open-World Detection with Semantic Priors and Virtual Outlier Synthesis".

## Abstract

Open-world object detection demands a balance between knowing categories and discovering new objects never seen in training. In practice, full model fine tuning often fails in such a dynamic environment. We find fully tuned models overfit known classes and lose sensitivity to unknown objects, with computationally large computational costs. To address this, we propose PE-OWOD, a light-weight approach that avoids heavy retraining. Instead of updating the entire network, we lock backbone and encoder to maintain stable visual priors, injecting compact Residual Adapters only to decoder for task adaptation. We also introduce VOS to define explicit decision boundary for open space with optional semantic initialization. The MS-COCO benchmarks show a striking efficiency advantage. Update less than 2% of model parameters, PE-OWOD achieves 64.7% Unknown Recall (significantly outperform fully tuned baselines), and reduce peak GPU memory usage by 86%.These results suggest that parameter efficient adaptation is not merely a constraint but a reliable strategy for robust open-world detection.

Install dependencies:
pip install -r requirements.txt
Note: We recommend using PyTorch >= 1.10 with CUDA support.

Data Preparation
1.Download MS-COCO 2017 dataset (train2017 and val2017) from the https://www.google.com/url?sa=E&q=https%3A%2F%2Fcocodataset.org%2F.
2.Organize the data as follows:
PE-OWOD/
├── train/            
│   └── annotations/  # Place instances_train2017.json here
│   └── train2017/    # Contains train2017 images
│   └── val2017/ # Contains val2017 images
└── ...
3.Generate the open-world task split:
python scripts/split_coco.py

Model Zoo
Model	                 Backbone	     mAP(Known)     U-Recall	       
PE-OWOD (Ver 4.0)	 ResNet-50 (Frozen)	 21.4        	64.7%	
(Pre-extracted CLIP embeddings are included in clip_embeddings_40.pth)

Training
To train the model on a single GPU:

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
  journal={preprint},
  year={2026}
}
