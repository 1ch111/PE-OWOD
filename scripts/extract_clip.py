#2
import torch
import os
from transformers import CLIPProcessor, CLIPModel


save_path = r"/PE-OWOD/clip_embeddings_40.pth"

COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

SEEN_CLASSES = COCO_CLASSES[:40]
print(f"Prepare to extract the text features of {len(SEEN_CLASSES)} known classes...")
print(f"Category example: {SEEN_CLASSES[:5]} ... {SEEN_CLASSES[-5:]}")

model_name = "openai/clip-vit-base-patch32"
print(f"Loading CLIP model: {model_name} ...")
try:
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
except Exception as e:
    print("Failed to download the model")
    raise e

text_inputs = [f"a photo of a {c}" for c in SEEN_CLASSES]

print("start...")
inputs = processor(text=text_inputs, return_tensors="pt", padding=True)

with torch.no_grad():
    text_features = model.get_text_features(**inputs)

text_features = text_features / text_features.norm(dim=-1, keepdim=True)

bg_embedding = torch.zeros((1, text_features.shape[1]))
final_embedding = torch.cat([text_features, bg_embedding], dim=0)

torch.save(final_embedding, save_path)
print(f"Success，The features have been saved to: {save_path}")
print(f"Feature dimension： {final_embedding.shape} (should [41, 512])")