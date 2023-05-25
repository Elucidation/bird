import json
import numpy as np
with open('coco_labels.json', 'r') as f:
    label_map = json.loads(f.read())

with open('result.json', 'r') as f:
    result = json.loads(f.read())

result = {key: np.array(value) for key, value in result.items()}
#  'raw_detection_boxes'
#  'detection_multiclass_scores'
#  'detection_classes'
#  'detection_boxes'
#  'raw_detection_scores'
#  'num_detections'
#  'detection_anchor_indices'
#  'detection_scores'
labels = np.array([label_map[str(int(cat_id))]
              for cat_id in result['detection_classes'][0]])
scores = result['detection_scores'][0]
print(labels)
print(scores[labels == 'bird'])
