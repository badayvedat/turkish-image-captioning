import json
import tensorflow as tf
import re


def parse_id(raw_string):
    return raw_string.split("_")[-1].split(".")[0]


def load_data(coco=True, flickr30k=True, flickr8k=True):
    train_files = []
    train_captions = []
    val_files = []
    val_captions = []

    if coco:
        with open("data/coco/coco.json") as f:
            coco_data = json.load(f)

        train_files.extend(coco_data["train_files"])
        train_captions.extend(coco_data["train_captions"])
        val_files.extend(coco_data["val_files"])
        val_captions.extend(coco_data["val_captions"])

    if flickr30k:
        with open("data/flickr30k/flickr30k.json") as f:
            flickr30k_data = json.load(f)

        train_files.extend(flickr30k_data["train_files"])
        train_captions.extend(flickr30k_data["train_captions"])
        val_files.extend(flickr30k_data["val_files"])
        val_captions.extend(flickr30k_data["val_captions"])

    if flickr8k:
        with open("data/flickr8k/flickr8k.json") as f:
            flickr8k_data = json.load(f)

        train_files.extend(flickr8k_data["train_files"])
        train_captions.extend(flickr8k_data["train_captions"])
        val_files.extend(flickr8k_data["val_files"])
        val_captions.extend(flickr8k_data["val_captions"])

    return train_files, train_captions, val_files, val_captions


# Preprocess image for VGG16.
# By default input shape of VGG16 is (224, 224, 3) where 3 denotes the number of channels
# Change to another model if you are using different model for feature extraction
# For more information check:
# https://keras.io/api/applications/vgg/
def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return img


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    strip_chars = "!\"#$%&'()*+,-./:;=?@[\]^_`{|}~"
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")
