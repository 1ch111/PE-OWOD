# PE-OWOD: Parameter-Efficient Open-World Detection with Semantic Priors and Virtual Outlier Synthesis

This is the official implementation of the paper:
"PE-OWOD: Parameter-Efficient Open-World Detection with Semantic Priors and Virtual Outlier Synthesis".

## Abstract

Open-world object detection requires knowledge of categories and discovery of new objects never seen in training. Full model fine tuning fails often in such dynamic environment. Fully tuned models overfit known classes and lose their sensitivity to unknown objects at high computational cost. To solve this problem, we propose PE-OWOD, a simple and light approach to retrain. Instead of updating the whole network, we lock backbone and encoder to maintain stable visual priors and inject compact Residual adapters only into decoder to adapt tasks. We also introduce VOS, which defines explicit decision boundary for open space with optional semantic initialization. MS-COCO benchmarks show remarkable efficiency advantages: Update less than 27% of models, PE-OWOD achieves 64.7% Unknown Recall (significantly outperform fully tuned baselines), and GPU memory usage is reduced by 86%.These results indicate that effective parameter adaptation is not a constraint; rather, it is a reliable and robust open-world detection strategy.

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
