import yaml
import argparse
import pickle

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from train_utils import *
from preprocess_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("config", help="Config file path")
parser.add_argument(
    "-wo",
    "--weights_output",
    help="Output path to save weights. \
                                                If no path is given, weights will be saved in checkpoints folder as: {NUM_HEADS}heads{VOCAB_SIZE}_{EPOCHS}epoch",
)
parser.add_argument(
    "-vo",
    "--vocab_output",
    help="Output path to save vocabulary. \
                                              If no path is given, vocabulary will be saved in current directory as: {vocab_size}vocab.pkl",
)

args = parser.parse_args()

CONFIG_PATH = args.config
VOCAB_OUT = args.vocab_output
WEIGHTS_OUT = args.weights_output


with open(CONFIG_PATH) as f:
    config_file = yaml.load(f)

EMBED_DIM = config_file["EMBED_DIM"]
FF_DIM = config_file["FF_DIM"]
NUM_HEADS = config_file["NUM_HEADS"]
SEQ_LENGTH = config_file["SEQ_LENGTH"]
VOCAB_SIZE = config_file["VOCAB_SIZE"]
BATCH_SIZE = config_file["BATCH_SIZE"]
EPOCHS = config_file["EPOCHS"]
LEARNING_RATE = config_file["LEARNING_RATE"]
USE_FEATURES = config_file["USE_FEATURES"]
COCO = config_file["COCO"]
FLICKR30K = config_file["FLICKR30K"]
FLICKR8K = config_file["FLICKR8K"]

if VOCAB_OUT is None:
    VOCAB_OUT = f"{VOCAB_SIZE}.pkl"

if WEIGHTS_OUT is None:
    WEIGHTS_OUT = f"checkpoints/{NUM_HEADS}heads{VOCAB_SIZE}_{EPOCHS}epoch"

train_files, train_captions, val_files, val_captions = load_data(
    coco=COCO, flickr30k=FLICKR30K, flickr8k=FLICKR8K
)

train_captions = [[cap] for cap in train_captions]
val_captions = [[cap] for cap in val_captions]

vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)

vectorization.adapt(train_captions)

# Pickle the vocabulary
pickle.dump(
    {"config": vectorization.get_config(), "weights": vectorization.get_weights()},
    open(VOCAB_OUT, "wb"),
)

train_dataset = make_dataset(
    train_files, train_captions, load_feature, vectorization, BATCH_SIZE
)

validation_dataset = make_dataset(
    val_files, val_captions, load_feature, vectorization, BATCH_SIZE
)

caption_model = build_caption_model(
    EMBED_DIM, FF_DIM, NUM_HEADS, SEQ_LENGTH, VOCAB_SIZE, USE_FEATURES
)

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
caption_model.compile(
    optimizer=optimizer,
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none"),
)

# Early stopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=[early_stopping],
)

caption_model.save_weights(WEIGHTS_OUT)
