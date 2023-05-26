import numpy as np
from six import BytesIO
from PIL import Image, ImageOps
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


print('\n------\n')
print(
    f'Program took {time.perf_counter() - prog_start_time:.2f} seconds to import TF & Hub')
print('tf', tf.__version__)
print('hub', hub.__version__)
print('cpus/gpus', tf.config.get_visible_devices())
if tf.test.gpu_device_name():
    print(f'GPU device available: {tf.test.gpu_device_name()}')


model_handle = 'https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1'


print('\n------\n')
print(f'Selected Model Handle at TensorFlow Hub: {model_handle}')

print('loading model...')
model_start_time = time.perf_counter()
hub_model = hub.load(model_handle).signatures['default']
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

def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def open_and_resize_image(path, new_width=256, new_height=256):
    pil_image = Image.open(path)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert("RGB")
    img = tf.convert_to_tensor(pil_image_rgb)
    return img

def search_image(img_path):
    """Searches an image and returns labels and scores"""
    # print(f'Loading image {img_path}')
    # img_np = load_image_into_numpy_array(img_path)
    img = load_img(img_path)
    # img = open_and_resize_image(img_path, 640, 480)
    img_tf  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    # print(f'Doing inference on {img_path}')

    start_time = time.perf_counter()
    result = hub_model(img_tf)
    end_time = time.perf_counter()
    # print(f"Inference time:  {end_time-start_time:.2f} s")

    del result['detection_class_labels']
    del result['detection_class_names']
    # Keep dict_keys('detection_class_entities', 'detection_boxes', 'detection_scores'])
    result = {key: value.numpy() for key, value in result.items()}

    result['detection_class_entities'] = np.array([entry.decode('utf-8') for entry in result['detection_class_entities']])

    is_good_score = result['detection_scores']>0.1
    is_animal = result['detection_class_entities'] == 'Animal'
    is_bird = result['detection_class_entities'] == 'Bird'
    is_raven = result['detection_class_entities'] == 'Raven'
    
    valid = is_good_score & (is_animal | is_bird | is_raven)    
    result['valid'] = valid
    return result


# camera_img_paths = glob.glob('birb_camera_images/*.jpg')
output_dir = 'bird_only_images'
input_subdir = '2023-05-dev'
output_fullpath = os.path.join(output_dir, input_subdir)
if not os.path.exists(output_fullpath):
    os.makedirs(output_fullpath)
paths = glob.glob(f'E:/birbcam/{input_subdir}/*.jpg')
print('----')
if not os.path.exists('processed_paths.txt'):
    processed = set()
else:
    with open('processed_paths.txt', 'r', encoding='utf-8') as f:
        processed = set(f.read().splitlines())
if not os.path.exists('processed_bad_paths.txt'):
    bad_paths = set()
else:
    with open('processed_bad_paths.txt', 'r', encoding='utf-8') as f:
        bad_paths = set(f.read().splitlines())

t_start = time.perf_counter()
for idx, img_path in enumerate(paths):
    if img_path in processed or img_path in bad_paths:
        continue
    t = time.perf_counter()
    t_str = f'{t - t_start:.2f}s - {idx}/{len(paths)}'
    print(f'{t_str} - {idx+1}/{len(paths)} - Searching {img_path}')
    try:
        result = search_image(img_path)
    except tf.errors.InvalidArgumentError:
        print(f'{t_str} - Error loading image on {img_path}, skipping')
        bad_paths.add(img_path)
        with open('processed_bad_paths.txt', 'a', encoding='utf-8') as f:
            print(img_path, file=f)
        continue
    except KeyboardInterrupt:
        print(f'{t_str} - Keyboard interrupt, stopping')
        break
    
    if any(result['valid']):
        num_valid = sum(result['valid'])
        print(f'{idx+1}/{len(paths)} - Found {num_valid} creatures in {img_path}')
        output_img_path = os.path.join(output_fullpath, os.path.basename(img_path))
        shutil.copy(img_path, output_img_path)
        result_json = json.dumps({key: value.tolist() for key, value in result.items()},  indent=2, sort_keys=True)
        result_json_path = os.path.join(output_fullpath, f'result_{os.path.splitext(os.path.basename(img_path))[0]}.json')
        with open(result_json_path, 'w', encoding='utf-8') as f:
            print(result_json, file=f)
        print('----')
    
    processed.add(img_path)
    with open('processed_paths.txt', 'a', encoding='utf-8') as f:
        print(img_path, file=f)
print('Done')
