import json
import pickle

# Path to the captions_train2017.json file
annotations_file = 'C://Users//rajes//Downloads//archive (2)//coco2017//annotations//captions_train2017.json'

# Load the annotations
with open(annotations_file, 'r') as f:
    annotations = json.load(f)['annotations']

# Extract captions and image URLs
coco_data_list = []
for annotation in annotations:
    image_id = annotation['image_id']
    caption = annotation['caption']
    image_url = f'http://images.cocodataset.org/train2017/{str(image_id).zfill(12)}.jpg'
    coco_data_list.append({'image_url': image_url, 'caption': caption})

# Save as pickle file
with open("coco_captions.pickle", "wb") as fp:
    pickle.dump(coco_data_list, fp)
