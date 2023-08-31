import os
import io
import imageio
import medmnist
import ipywidgets
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers


# Setting seed for reproducibility
SEED = 42
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
keras.utils.set_random_seed(SEED)

# DATA
DATASET_PATH = "data_time_series"
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 10, IMAGE_CHANNELS)
NUM_CLASSES = 7

# OPTIMIZER
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# TRAINING
EPOCHS = 60

# TUBELET EMBEDDING
PATCH_SIZE = (8, 8, 8)
NUM_PATCHES = (INPUT_SHAPE[0] // PATCH_SIZE[0]) ** 2

# ViViT ARCHITECTURE
LAYER_NORM_EPS = 1e-6
PROJECTION_DIM = 128
NUM_HEADS = 8
NUM_LAYERS = 8

def load_track_from_class_folder(class_folder):
    tracks = os.listdir(os.path.join(DATASET_PATH, class_folder))
    tracks.sort()
    return tracks
def load_image(path):
    tf.io.decode_jpeg(tf.io.read_file(path), channels=IMAGE_CHANNELS)

def load_images_from_track(class_folder, track):
    track_image_files = sorted(
        os.listdir(
            os.path.join(
                DATASET_PATH, 
                class_folder, 
                track)))
    print (track_image_files)
    x_data = tf.data.Dataset.from_tensor_slices(track_image_files)
    x_data = x_data.map(load_image, num_parallel_calls=AUTO)
    return x_data

def prepare_data():
    # list all class folders in the dataset path
    class_folders = os.listdir(DATASET_PATH)
    class_folders.sort()
    for class_folder in class_folders:
        class_tracks = load_track_from_class_folder(class_folder)
        for track in class_tracks:
            track_images = load_images_from_track(class_folder, track)
            print(track_images)


# Get the dataset
prepared_dataset = prepare_data()
# (train_videos, train_labels) = prepared_dataset[0]
# (valid_videos, valid_labels) = prepared_dataset[1]
# (test_videos, test_labels) = prepared_dataset[2]
# Get the shape of the dataset
# print(f"Train videos shape: {train_videos.shape}")
# print(f"Train labels shape: {train_labels.shape}")
# print(f"Validation videos shape: {valid_videos.shape}")
# print(f"Validation labels shape: {valid_labels.shape}")
# print(f"Test videos shape: {test_videos.shape}")
# print(f"Test labels shape: {test_labels.shape}")
# print(f"Example: {train_videos[0].shape}")

# @tf.function
# def preprocess(frames: tf.Tensor, label: tf.Tensor):
#     """Preprocess the frames tensors and parse the labels."""
#     # Preprocess images
#     frames = tf.image.convert_image_dtype(
#         frames[
#             ..., tf.newaxis
#         ],  # The new axis is to help for further processing with Conv3D layers
#         tf.float32,
#     )
#     # Parse label
#     label = tf.cast(label, tf.float32)
#     return frames, label


# def prepare_dataloader(
#     videos: np.ndarray,
#     labels: np.ndarray,
#     loader_type: str = "train",
#     batch_size: int = BATCH_SIZE,
# ):
#     """Utility function to prepare the dataloader."""
#     dataset = tf.data.Dataset.from_tensor_slices((videos, labels))

#     if loader_type == "train":
#         dataset = dataset.shuffle(BATCH_SIZE * 2)

#     dataloader = (
#         dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
#         .batch(batch_size)
#         .prefetch(tf.data.AUTOTUNE)
#     )
#     return dataloader


# trainloader = prepare_dataloader(train_videos, train_labels, "train")
# validloader = prepare_dataloader(valid_videos, valid_labels, "valid")
# testloader = prepare_dataloader(test_videos, test_labels, "test")

# class TubeletEmbedding(layers.Layer):
#     def __init__(self, embed_dim, patch_size, **kwargs):
#         super().__init__(**kwargs)
#         self.projection = layers.Conv3D(
#             filters=embed_dim,
#             kernel_size=patch_size,
#             strides=patch_size,
#             padding="VALID",
#         )
#         self.flatten = layers.Reshape(target_shape=(-1, embed_dim))

#     def call(self, videos):
#         projected_patches = self.projection(videos)
#         flattened_patches = self.flatten(projected_patches)
#         return flattened_patches

# class PositionalEncoder(layers.Layer):
#     def __init__(self, embed_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim

#     def build(self, input_shape):
#         _, num_tokens, _ = input_shape
#         self.position_embedding = layers.Embedding(
#             input_dim=num_tokens, output_dim=self.embed_dim
#         )
#         self.positions = tf.range(start=0, limit=num_tokens, delta=1)

#     def call(self, encoded_tokens):
#         # Encode the positions and add it to the encoded tokens
#         encoded_positions = self.position_embedding(self.positions)
#         encoded_tokens = encoded_tokens + encoded_positions
#         return encoded_tokens

# def create_vivit_classifier(
#     tubelet_embedder,
#     positional_encoder,
#     input_shape=INPUT_SHAPE,
#     transformer_layers=NUM_LAYERS,
#     num_heads=NUM_HEADS,
#     embed_dim=PROJECTION_DIM,
#     layer_norm_eps=LAYER_NORM_EPS,
#     num_classes=NUM_CLASSES,
# ):
#     # Get the input layer
#     inputs = layers.Input(shape=input_shape)
#     # Create patches.
#     patches = tubelet_embedder(inputs)
#     # Encode patches.
#     encoded_patches = positional_encoder(patches)

#     # Create multiple layers of the Transformer block.
#     for _ in range(transformer_layers):
#         # Layer normalization and MHSA
#         x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#         attention_output = layers.MultiHeadAttention(
#             num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=0.1
#         )(x1, x1)

#         # Skip connection
#         x2 = layers.Add()([attention_output, encoded_patches])

#         # Layer Normalization and MLP
#         x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
#         x3 = keras.Sequential(
#             [
#                 layers.Dense(units=embed_dim * 4, activation=tf.nn.gelu),
#                 layers.Dense(units=embed_dim, activation=tf.nn.gelu),
#             ]
#         )(x3)

#         # Skip connection
#         encoded_patches = layers.Add()([x3, x2])

#     # Layer normalization and Global average pooling.
#     representation = layers.LayerNormalization(epsilon=layer_norm_eps)(encoded_patches)
#     representation = layers.GlobalAvgPool1D()(representation)

#     # Classify outputs.
#     outputs = layers.Dense(units=num_classes, activation="softmax")(representation)

#     # Create the Keras model.
#     model = keras.Model(inputs=inputs, outputs=outputs)
#     return model

# def run_experiment():
#     # Initialize model
#     model = create_vivit_classifier(
#         tubelet_embedder=TubeletEmbedding(
#             embed_dim=PROJECTION_DIM, patch_size=PATCH_SIZE
#         ),
#         positional_encoder=PositionalEncoder(embed_dim=PROJECTION_DIM),
#     )

#     # Compile the model with the optimizer, loss function
#     # and the metrics.
#     optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
#     model.compile(
#         optimizer=optimizer,
#         loss="sparse_categorical_crossentropy",
#         metrics=[
#             keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
#             keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
#         ],
#     )
#     model.summary()
#     # Train the model.
#     _ = model.fit(trainloader, epochs=EPOCHS, validation_data=validloader)

#     _, accuracy, top_5_accuracy = model.evaluate(testloader)
#     print(f"Test accuracy: {round(accuracy * 100, 2)}%")
#     print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

#     return model


# model = run_experiment()

# NUM_SAMPLES_VIZ = 25
# testsamples, labels = next(iter(testloader))
# testsamples, labels = testsamples[:NUM_SAMPLES_VIZ], labels[:NUM_SAMPLES_VIZ]

# ground_truths = []
# preds = []
# videos = []

# for i, (testsample, label) in enumerate(zip(testsamples, labels)):
#     # Generate gif
#     with io.BytesIO() as gif:
#         imageio.mimsave(gif, (testsample.numpy() * 255).astype("uint8"), "GIF", fps=5)
#         videos.append(gif.getvalue())

#     # Get model prediction
#     output = model.predict(tf.expand_dims(testsample, axis=0))[0]
#     pred = np.argmax(output, axis=0)

#     ground_truths.append(label.numpy().astype("int"))
#     preds.append(pred)


# def make_box_for_grid(image_widget, fit):
#     """Make a VBox to hold caption/image for demonstrating option_fit values.

#     Source: https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Styling.html
#     """
#     # Make the caption
#     if fit is not None:
#         fit_str = "'{}'".format(fit)
#     else:
#         fit_str = str(fit)

#     h = ipywidgets.HTML(value="" + str(fit_str) + "")

#     # Make the green box with the image widget inside it
#     boxb = ipywidgets.widgets.Box()
#     boxb.children = [image_widget]

#     # Compose into a vertical box
#     vb = ipywidgets.widgets.VBox()
#     vb.layout.align_items = "center"
#     vb.children = [h, boxb]
#     return vb


# boxes = []
# for i in range(NUM_SAMPLES_VIZ):
#     ib = ipywidgets.widgets.Image(value=videos[i], width=100, height=100)
#     true_class = info["label"][str(ground_truths[i])]
#     pred_class = info["label"][str(preds[i])]
#     caption = f"T: {true_class} | P: {pred_class}"

#     boxes.append(make_box_for_grid(ib, caption))

# ipywidgets.widgets.GridBox(
#     boxes, layout=ipywidgets.widgets.Layout(grid_template_columns="repeat(5, 200px)")
# )
