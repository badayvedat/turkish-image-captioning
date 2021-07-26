import json
import scipy.io
import numpy as np
import os
from tqdm.notebook import tqdm

flickr8k = 'data/flickr8k'
flickr8k_features = 'data/flickr8k/feats/vgg_feats.mat'
flickr8k_data = 'data/flickr8k/feats/dataset.json'
flickr8k_captions = 'data/flickr8k/captions/tasviret8k_captions.json'
features_path = 'data/features'

def parse_id(raw_string):
    return raw_string.split('_')[-1].split('.')[0]

def parse_filename(raw_string):
    return raw_string.split('_')[0]

with open(flickr8k_data) as f:
    data = json.load(f)['images']

features = scipy.io.loadmat(flickr8k_features)['feats']

if not os.path.exists(features_path):
    os.mkdir(features_path)

# Saving features as npy
for i in tqdm(range(len(data))):
    _id = parse_filename(data[i]['filename'])
    np.save(os.path.join(features_path, 'flickr8k_' + _id), features[:, i])

with open(flickr8k_captions) as f:
    data = json.load(f)['images']

filenames = []
captions = []
for cap_data in data:
    filename = parse_filename(cap_data['filename'])
    out_path = os.path.join(features_path, 'flickr8k_' + filename + '.npy')
    for i in range(len(cap_data['sentences'])):
        captions.append(f"<start> {cap_data['sentences'][i]['raw']} <end>")
        filenames.append(out_path)

with open(os.path.join(flickr8k, 'flickr8k.json'), 'w') as f:
    json.dump({
        'train_files': [],
        'train_captions': [],
        'val_files': filenames,
        'val_captions': captions
    }, f)
