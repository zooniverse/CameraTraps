import json
import os
import shutil
import pickle
import numpy as np


image_data = json.load(open('/home/ubuntu/cct_data/empty_images/empty_images.json','r'))
im_id_to_im = {im['id']:im for im in image_data['images']}

data = pickle.load(open('/home/ubuntu/efs/models/megadetector_predictions/empty_cct.p','rb'))

min_score = 0.9

empty_im_ids = {}
for i,im_id in enumerate(data['images']):
    if np.max(data['detection_scores'][i]) < min_score:
        empty_im_ids[im_id] = np.max(data['detection_scores'][i])
        
new_ims = [im_id_to_im[im_id] for im_id in empty_im_ids]

new_anns = []
for ann in image_data['annotations']:
    if ann['image_id'] in empty_im_ids:
        new_anns.append(ann)
        
image_data['images'] = new_ims
image_data['annotations'] = new_anns

output_folder = '/home/ubuntu/cct_data/true_empty_images/'
image_root = '/home/ubuntu/cct_data/cct_images/'

json.dump(image_data,open(output_folder+'empty_images.json','w'))

for im in new_ims:
    shutil.copyfile(image_root+im['file_name'],output_folder+'images/')ls 
    

