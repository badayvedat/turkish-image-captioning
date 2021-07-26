import json
import scipy.io
import numpy as np
import os
from tqdm import tqdm

coco = 'data/coco/'
coco_train_caps = 'data/coco/captions/train/coco_train_captions_tr.json'
coco_val1_caps = 'data/coco/captions/val/val_baseline_1.json'
coco_val2_caps = 'data/coco/captions/val/val_baseline_2.json'
coco_features = 'data/coco/feats/vgg_feats.mat'
coco_data = 'data/coco/feats/dataset.json'
features_path = 'data/features'

def parse_id(raw_string):
    return raw_string.split('_')[-1].split('.')[0]

with open(coco_data) as f:
    data = json.load(f)['images']

features = scipy.io.loadmat(coco_features)['feats']

if not os.path.exists(features_path):
    os.mkdir(features_path)

# Saving features as npy
for i in tqdm(range(len(data))):
    _id = parse_id(data[i]['filename'])
    np.save(os.path.join(features_path, 'coco_' + _id), features[:, i])

# Training data
with open(coco_train_caps) as f:
    data = json.load(f)

train_files = []
train_caps = []
for i in range(len(data)):
    out_path = os.path.join(features_path, 'coco_' + parse_id(data[i]['file_path']) + '.npy')
    train_files.extend([out_path] * len(data[i]['captions']))
    for j in range(len(data[i]['captions'])):
        train_caps.append(f"<start> {data[i]['captions'][j]} <end>")

# Validation data
with open(coco_val2_caps) as f:
    data = json.load(f)

id_to_image = {}
for x in data['images']:
    id_to_image[x['id']] = parse_id(x['file_name'])

val_files = []
val_caps = []
for x in data['annotations']:
    out_path = os.path.join(features_path, 'coco_' + id_to_image[x['image_id']] + '.npy')
    val_files.append(out_path)
    val_caps.append(f"<start> {x['caption']} <end>")

with open(os.path.join(coco, 'coco.json'), 'w') as f:
    json.dump({
        'train_files': train_files,
        'train_captions': train_caps,
        'val_files': val_files,
        'val_captions': val_caps
    }, f)
