import json
import os
from pathlib import Path


data_root = r"/PE-OWOD/train"

ann_file = os.path.join(data_root, "annotations", "instances_train2017.json")

save_file = os.path.join(data_root, "annotations", "instances_train2017_seen_40.json")



def filter_coco(ann_path, save_path, max_category_index=40):
    print(f"Loading {ann_path} ...")
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)

    categories = sorted(coco_data['categories'], key=lambda x: x['id'])

    seen_categories = categories[:max_category_index]
    seen_ids = set([cat['id'] for cat in seen_categories])

    print(f"Total number of categories: {len(categories)}")
    print(f"Retain the known number of categories: {len(seen_categories)}")
    print(f"An example of a known category ID: {list(seen_ids)[:5]}...")

    new_annotations = []
    for ann in coco_data['annotations']:
        if ann['category_id'] in seen_ids:
            new_annotations.append(ann)

    valid_image_ids = set([ann['image_id'] for ann in new_annotations])
    new_images = [img for img in coco_data['images'] if img['id'] in valid_image_ids]

    new_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'images': new_images,
        'annotations': new_annotations,
        'categories': seen_categories
    }

    print(f"The number of images before filtering: {len(coco_data['images'])}, Number of annotations: {len(coco_data['annotations'])}")
    print(f"The number of images after filtering: {len(new_images)}, Number of annotations: {len(new_annotations)}")

    with open(save_path, 'w') as f:
        json.dump(new_data, f)
    print(f"The newly annotated file has been saved to: {save_path}")


if __name__ == '__main__':
    filter_coco(ann_file, save_file)