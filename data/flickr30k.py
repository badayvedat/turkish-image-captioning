import json
import scipy.io
import numpy as np
import os
from tqdm import tqdm

flickr30k = 'data/flickr30k'
flickr30k_captions = 'data/flickr30k/captions/flickr30k_captions_tr.txt'
flickr30k_filenames = 'data/flickr30k/captions/flicker30k_file_names.txt'
flickr30k_features = 'data/flickr30k/feats/vgg_feats.mat'
flickr30k_data = 'data/flickr30k/feats/dataset.json'
features_path = 'data/features'

def parse_id(raw_string):
    return raw_string.split('_')[-1].split('.')[0]

with open(flickr30k_data) as f:
    data = json.load(f)['images']

features = scipy.io.loadmat(flickr30k_features)['feats']

if not os.path.exists(features_path):
    os.mkdir(features_path)

# Saving features as npy
for i in tqdm(range(len(data))):
    _id = parse_id(data[i]['filename'])
    np.save(os.path.join(features_path, 'flickr30k_' + _id), features[:, i])

with open(flickr30k_captions) as f:
    captions = f.read().splitlines()
    captions = [f'<start> {cap} <end>' for cap in captions]

with open(flickr30k_filenames) as f:
    filenames = f.read().splitlines()
    filenames = [f"{os.path.join(features_path, 'flickr30k_' + filename[:-4] + '.npy')}" for filename in filenames]

filenames_exist = []
captions_exist = []
for filename, caption in zip(filenames, captions):
    if os.path.exists(filename):
        filenames_exist.append(filename)
        captions_exist.append(caption)


# All captions will be used in training
with open(os.path.join(flickr30k, 'flickr30k.json'), 'w') as f:
    json.dump({
        'train_files': filenames_exist,
        'train_captions': captions_exist,
        'val_files': [],
        'val_captions': []
    }, f)
