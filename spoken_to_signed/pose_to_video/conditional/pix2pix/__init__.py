import cv2
import numpy as np
import tensorflow as tf
from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from tensorflow.keras.models import load_model

from spoken_to_signed.pose_to_video.utils import set_tensorflow_memory_growth

video_model_g = load_model("./spoken_to_signed/pose_to_video/conditional/pix2pix/pix_to_pix.h5")

def translate_image(model, image):
    # Assume 'image' is a NumPy array
    pixels = image.astype(np.float32) / 255.0  # Convert to float and normalize pixel values to [0, 1]
    pixels = (pixels - 0.5) * 2  # Normalizing the images to [-1, 1]

    tensor = np.expand_dims(np.expand_dims(pixels, 0), 0)  # Add batch and time dimensions

    tensor = tf.convert_to_tensor(tensor, dtype=tf.float32)  # Convert input to tensor for model prediction

    # Note: The training=True is intentional here since you want the batch statistics,
    # while running the model on the test dataset.
    # If you use training=False, you get the accumulated statistics learned from the
    # training dataset (which you don't want).
    pred = model(tensor, training=True)

    # Convert prediction back to numpy and normalize to range [0, 1]
    pred = pred.numpy()
    pred = (pred * 0.5) + 0.5
    pred = pred * 255.0

    pred = np.squeeze(pred, 0)  # Remove time dimension
    pred = np.squeeze(pred, 0)  # Remove batch dimension

    return pred.astype(np.uint8)


def pose_to_video(pose: Pose, model_path: str) -> iter:
    set_tensorflow_memory_growth()

    model = load_model(model_path)

    # Scale pose to 256x256
    scale_w = pose.header.dimensions.width / 256
    scale_h = pose.header.dimensions.height / 256
    pose.body.data /= np.array([scale_w, scale_h, 1])
    pose.header.dimensions.width = pose.header.dimensions.height = 256

    visualizer = PoseVisualizer(pose, thickness=1)
    for pose_img_bgr in visualizer.draw():
        pose_img_rgb = cv2.cvtColor(pose_img_bgr, cv2.COLOR_BGR2RGB)
        yield translate_image(model, pose_img_rgb)

def pose_to_video_pix2pix(pose: Pose) -> iter:
    set_tensorflow_memory_growth()

    global video_model_g

    # Scale pose to 256x256
    scale_w = pose.header.dimensions.width / 256
    scale_h = pose.header.dimensions.height / 256
    pose.body.data /= np.array([scale_w, scale_h, 1])
    pose.header.dimensions.width = pose.header.dimensions.height = 256

    visualizer = PoseVisualizer(pose, thickness=1)
    for pose_img_bgr in visualizer.draw():
        pose_img_rgb = cv2.cvtColor(pose_img_bgr, cv2.COLOR_BGR2RGB)
        yield translate_image(video_model_g, pose_img_rgb)