import torch

src_path = r"/detr-r50-e632da11.pth"

dst_path = r"/detr-r50-finetune.pth"

print(f"the weight: {src_path}")
try:
    checkpoint = torch.load(src_path, map_location='cpu')
except FileNotFoundError:
    print(f"error，The source file cannot be found {src_path}")
    exit()

model_state_dict = checkpoint['model']

keys_to_remove = [
    "class_embed.weight", "class_embed.bias",
    "bbox_embed.layers.0.weight", "bbox_embed.layers.0.bias",
    "bbox_embed.layers.1.weight", "bbox_embed.layers.1.bias",
    "bbox_embed.layers.2.weight", "bbox_embed.layers.2.bias",
]

for key in keys_to_remove:
    if key in model_state_dict:
        del model_state_dict[key]

checkpoint['model'] = model_state_dict

for key in ['optimizer', 'lr_scheduler', 'epoch']:
    if key in checkpoint:
        del checkpoint[key]

torch.save(checkpoint, dst_path)
print(f"Successfully generated， File location {dst_path}")