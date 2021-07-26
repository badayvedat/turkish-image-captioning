import numpy as np
import pickle
import yaml
import argparse

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from preprocess_utils import load_image
from train_utils import build_caption_model

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="config file path")
parser.add_argument("-v", "--vocab", help="vocabulary file path")
parser.add_argument("-w", "--weights", help="weights file path")
parser.add_argument("file", help="image file to generate caption")
args = parser.parse_args()

CONFIG_PATH = args.config
VOCAB_PATH = args.vocab
WEIGHTS_PATH = args.weights
FILE_PATH = args.file

if CONFIG_PATH is None:
    print(f"Config file set as 16heads20kvoc.yaml")
    CONFIG_PATH = "./configs/16heads20kvoc.yaml"

if VOCAB_PATH is None:
    print("Vocabulary set as 20kVocab.pkl")
    VOCAB_PATH = "./20kVocab.pkl"

if WEIGHTS_PATH is None:
    print("Weights set as 16head20k9epoch from checkpoints")
    WEIGHTS_PATH = "./checkpoints/16head20k9epoch"


with open(CONFIG_PATH) as f:
    config_file = yaml.load(f)

EMBED_DIM = config_file["EMBED_DIM"]
FF_DIM = config_file["FF_DIM"]
NUM_HEADS = config_file["NUM_HEADS"]
SEQ_LENGTH = config_file["SEQ_LENGTH"]
VOCAB_SIZE = config_file["VOCAB_SIZE"]
USE_FEATURES = config_file["USE_FEATURES"]


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

    decoded_caption = (
        decoded_caption.replace("<start> ", "").replace(" <end>", "").strip()
    )
    return decoded_caption


with open(VOCAB_PATH, "rb") as f:
    vec_pkl = pickle.load(f)

vectorization = TextVectorization.from_config(vec_pkl["config"])
vectorization.set_weights(vec_pkl["weights"])

caption_model = build_caption_model(
    EMBED_DIM, FF_DIM, NUM_HEADS, SEQ_LENGTH, VOCAB_SIZE, from_features=False
)

caption_model.load_weights(WEIGHTS_PATH)

if __name__ == "__main__":
    caption = generate_caption(FILE_PATH, caption_model, vectorization, SEQ_LENGTH)
    print("Predicted Caption:", caption)
