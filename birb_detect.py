import tensorflow as tf
import tensorflow_hub as hub
import json
from PIL import Image
from six import BytesIO
import numpy as np
import time
prog_start_time = time.time()

# import os
# os.environ["TFHUB_CACHE_DIR"] = "gs://samop-tf-cache/tfhub-modules-cache"


# Print Tensorflow version
print('\n------\n')
print(f'Program took {time.time() - prog_start_time} seconds to init TF/Hub')
print('tf', tf.__version__)
print('hub', hub.__version__)
print('cpus/gpus', tf.config.get_visible_devices())
if tf.test.gpu_device_name():
    print(f'GPU device available: {tf.test.gpu_device_name()}')


# model_display_name = 'SSD MobileNet V2 FPNLite 640x640'
# model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1'

# model_display_name = 'SSD MobileNet v2 320x320'
# model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2'

model_display_name = 'SSD MobileNet V2 FPNLite 320x320'
model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1'


print('\n------\n')
print(f'Selected model: {model_display_name}')
print(f'Model Handle at TensorFlow Hub: {model_handle}')

print('loading model...')
hub_model = hub.load(model_handle)
print('model loaded!')
print('resolves to:', hub.resolve(model_handle))


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


img_path = 'Birbcamera_14-43-07.jpg'
print(f'Loading image {img_path}')
img_np = load_image_into_numpy_array(img_path)
print(f'Doing inference on {img_path}')

start_time = time.time()
result = hub_model(img_np)
end_time = time.time()

print('Finished inference in ')
print("Inference time: ", end_time-start_time)
result = {key: value.numpy() for key, value in result.items()}
result_json = json.dumps({key: value.tolist()
                         for key, value in result.items()})
print(f'Getting result took {time.time() - prog_start_time} seconds total')

with open('result.json', 'w') as f:
    print(result_json, file=f)

with open('coco_labels.json', 'r') as f:
    label_map = json.loads(f.read())

print([(label_map[str(int(cat_id))], f'{score:.2f}') for cat_id, score in zip(
    result['detection_classes'][0], result['detection_scores'][0])])
