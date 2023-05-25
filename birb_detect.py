import numpy as np
from six import BytesIO
from PIL import Image
import glob
import json
import time
import os
import shutil
prog_start_time = time.perf_counter()


with open('coco_labels.json', 'r', encoding='utf8') as f:
    label_map = json.loads(f.read())

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


# model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2' # slow
# model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1'  # slow
model_handle = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1'
# model_handle = 'https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1'


print('\n------\n')
print(f'Selected Model Handle at TensorFlow Hub: {model_handle}')

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
    # print(
    #     f'Getting result took {time.perf_counter() - prog_start_time:.2f} seconds total')    

    labels = np.array([label_map[str(int(cat_id))]
              for cat_id in result['detection_classes'][0]])
    result['detection_class_names'] = labels
    return result


camera_img_paths = glob.glob('birb_camera_images/*.jpg')
print('----')
for idx, img_path in enumerate(camera_img_paths):
    print(f'{idx+1}/{len(camera_img_paths)} - Searching {img_path}')
    result, labels, scores = search_image(img_path)
    # print(list(zip(labels, scores)))
    if 'bird' in result['detection_class_names'][result['detection_class_names'][0] > 0.2]:
        print(f'{idx+1}/{len(camera_img_paths)} - Found a bird in {img_path}')
        shutil.copy(img_path, f'bird_only_images/{os.path.basename(img_path)}')
        result_json = json.dumps({key: value.tolist() for key, value in result.items()},  indent=2, sort_keys=True)
        result_json_path = f'bird_only_images/result_{os.path.splitext(os.path.basename(img_path))[0]}.json'
        with open(result_json_path, 'w', encoding='utf8') as f:
            print(result_json, file=f)
    else:
        print(f'{idx+1}/{len(camera_img_paths)} - No bird in {img_path}')
    

    print('----')
print('Done')
