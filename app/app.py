import numpy as np
import os
import yaml
import pickle

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from preprocess_utils import load_image
from train_utils import build_caption_model

CONFIG_PATH = '../configs/16heads20kvoc.yaml'
VOCAB_PATH = '../20kVocab.pkl'
WEIGHTS_PATH = '../checkpoints/16head20k9epoch'

with open(CONFIG_PATH) as f:
    config_file = yaml.safe_load(f)

EMBED_DIM = config_file['EMBED_DIM']
FF_DIM = config_file['FF_DIM']
NUM_HEADS = config_file['NUM_HEADS']
SEQ_LENGTH = config_file['SEQ_LENGTH']
VOCAB_SIZE = config_file['VOCAB_SIZE']
USE_FEATURES = config_file['USE_FEATURES']

def generate_caption(img_path, caption_model, vectorization, max_len):
    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))

    img = load_image(img_path)

    # Pass the image to the CNN
    img = tf.expand_dims(img, 0)

    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_len):
        tokenized_caption = vectorization(np.array([decoded_caption]))[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
    
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == " <end>" or sampled_token == "":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "").replace(" <end>", "").strip()
    return decoded_caption

with open(VOCAB_PATH, 'rb') as f:
    vec_pkl = pickle.load(f)

vectorization = TextVectorization.from_config(vec_pkl['config'])
vectorization.set_weights(vec_pkl['weights'])

caption_model = build_caption_model(
    EMBED_DIM, FF_DIM, NUM_HEADS, SEQ_LENGTH, VOCAB_SIZE, from_features=False
)

caption_model.load_weights(WEIGHTS_PATH)


UPLOAD_FOLDER = 'static/images'

app = Flask(__name__, instance_relative_config=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_extension(filename):
    return os.path.splitext(filename)[1] == ".jpg"


@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_extension(f.filename):
            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path)
            caption = generate_caption(file_path, caption_model, vectorization, SEQ_LENGTH - 1)
            return render_template('index.html', filename=file_path, caption=caption)
        else:
            filename = 'cat.jpg'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            caption = generate_caption(file_path, caption_model, vectorization, SEQ_LENGTH - 1)
            return render_template('index.html', filename=file_path, caption=caption)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
