import tensorflow as tf
from tensorflow import keras

import numpy as np

from model import TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptionModel


# CNN model to be used as feature extractor
def get_cnn_model():
    ## VGG takes the shape (224, 224, 3) as input shape
    base_model = keras.applications.VGG16(
        input_shape=(224, 224, 3), weights="imagenet", include_top=True
    )

    # Later can be set to True if fine-tuning is needed
    base_model.trainable = False

    # Create new keras model with all layers same as VGG16
    # except of last layer, because it does classification
    # and we are not interested in classification
    base_model_out = base_model.layers[-2].output
    # Output shape of VGG16 is (4096,)
    base_model_out = keras.layers.Reshape((-1, 4096))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model


def get_feature_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(4096,)),
            tf.keras.layers.Reshape((-1, 4096)),
        ]
    )


# Given captions and images, using tf dataset API make datasets
def make_dataset(images, captions, image_fn=None, caption_fn=None, batch_size=1):
    img_dataset = tf.data.Dataset.from_tensor_slices(images)
    if image_fn:
        img_dataset = img_dataset.map(image_fn, num_parallel_calls=tf.data.AUTOTUNE)

    cap_dataset = tf.data.Dataset.from_tensor_slices(captions)
    if caption_fn:
        cap_dataset = cap_dataset.map(caption_fn, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))
    dataset = (
        dataset.shuffle(256, seed=42, reshuffle_each_iteration=False)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def build_caption_model(
    embed_dim, ff_dim, num_heads, seq_length, vocab_size, from_features=True
):
    if from_features:
        cnn_model = get_feature_model()
    else:
        cnn_model = get_cnn_model()

    cnn_model.trainable = False
    encoder = TransformerEncoderBlock(
        embed_dim=embed_dim, dense_dim=ff_dim, num_heads=num_heads
    )

    decoder = TransformerDecoderBlock(
        embed_dim=embed_dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        seq_length=seq_length,
        vocab_size=vocab_size,
    )

    caption_model = ImageCaptionModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder,
    )

    return caption_model


def read_feature(path):
    return np.load(path.decode("utf-8"))


def load_feature(path):
    return tf.numpy_function(read_feature, [path], tf.float32)
