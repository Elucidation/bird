import numpy as np
from six import BytesIO
from PIL import Image
import glob
import json
import time
prog_start_time = time.perf_counter()


with open('coco_labels.json', 'r', encoding='utf8') as f:
    label_map = json.loads(f.read())

# import os
# os.environ["TFHUB_CACHE_DIR"] = "gs://samop-tf-cache/tfhub-modules-cache"
import tensorflow_hub as hub
import tensorflow as tf


# Print Tensorflow version
print('\n------\n')
print(
    f'Program took {time.perf_counter() - prog_start_time:.2f} seconds to import TF & Hub')
print('tf', tf.__version__)
print('hub', hub.__version__)
print('cpus/gpus', tf.config.get_visible_devices())
if tf.test.gpu_device_name():
    print(f'GPU device available: {tf.test.gpu_device_name()}')


# model_display_name = 'SSD MobileNet v2 320x320'
# model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'

model_display_name = 'SSD MobileNet V2 FPNLite 320x320'
model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1'

# model_display_name = 'SSD MobileNet V2 FPNLite 640x640'
# model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1'


print('\n------\n')
print(f'Selected model: {model_display_name}')
print(f'Model Handle at TensorFlow Hub: {model_handle}')

print('loading model...')
model_start_time = time.perf_counter()
hub_model = hub.load(model_handle)
print('Model cached at :', hub.resolve(model_handle))
print(f"Model loaded, took time: {time.perf_counter()-model_start_time:2f} s")
print('\n------\n')


def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = None
    image_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(image_data))

    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)



def search_image(img_path):
    """Searches an image and returns labels and scores"""
    print(f'Loading image {img_path}')
    img_np = load_image_into_numpy_array(img_path)
    print(f'Doing inference on {img_path}')

    start_time = time.perf_counter()
    result = hub_model(img_np)
    end_time = time.perf_counter()

    print('Finished inference in ')
    print(f"Inference time:  {end_time-start_time:.2f} s")
    result = {key: value.numpy() for key, value in result.items()}
    print(
        f'Getting result took {time.perf_counter() - prog_start_time:.2f} seconds total')

    # result_json = json.dumps({key: value.tolist()
    #                           for key, value in result.items()})
    # with open('result.json', 'w') as f:
    #     print(result_json, file=f)

    labels = [label_map[str(int(cat_id))]
              for cat_id in result['detection_classes'][0]]
    scores = result['detection_scores'][0]
    return labels, scores


# img_path = 'Birbcamera_14-42-57.jpg'
camera_img_paths = glob.glob('Birb*.jpg')
print('----')
for idx, img_path in enumerate(camera_img_paths):
    print(f'{idx+1}/{len(camera_img_paths)} - Searching {img_path}')
    labels, scores = search_image(img_path)
    if 'bird' in labels:
        print(f'Found a bird in {img_path}')
    print('----')
print('Done')