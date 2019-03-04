import json

import numpy as np
import os
import random
from PIL import ImageStat


class COCODataLoader():

    def __init__(self, json_file, image_file_root=None):
        self.json_file = json_file
        self.data = json.load(open(self.json_file,'r'))
        self.images = self.data['images']
        self.n_images = len(self.images)
        self.annotations = self.data['annotations']
        #deal with int/string mismatch issue
        for ann in self.annotations:
            ann['id'] = str(ann['id'])
        self.n_annotations = len(self.annotations)
        self.categories = self.data['categories']
        self.n_categories = len(self.categories)
        if 'info' in self.data:
            self.info = self.data['info']

        self.im_id_to_im = {im['id']:im for im in self.images}
        self.ann_id_to_ann = {ann['id']:ann for ann in self.annotations}
        self.cat_name_to_cat_id = {cat['name']:cat['id'] for cat in self.categories}
        self.cat_id_to_cat_name = {cat['id']:cat['name'] for cat in self.categories}
        self.im_id_to_anns = {im['id']:[] for im in self.images}
        for ann in self.annotations:
            self.im_id_to_anns[ann['image_id']].append(ann)

        if image_file_root is not None:
            self.im_id_to_path = {im['id']:image_file_root+im['file_name'] for im in self.images}

        

    def print_stats(self):
        print('Images: '+str(self.n_images))
        print('Annotations: '+str(self.n_annotations))
        print('Categories: '+str(self.n_categories))

    def update_category_list(self, category_file,cats_to_ignore):
        new_categories = json.load(open(category_file,'r'))
        valid_cats = [cat for cat in new_categories if cat['name'] not in cats_to_ignore]
        old_cat_id_to_new_cat_id = {valid_cats[idx]['id']:idx+1 for idx in range(len(valid_cats))}
        print('Mapping to new categories...')
        print(old_cat_id_to_new_cat_id)
        new_cats = []
        for cat in valid_cats:
            new_cat = {}
            new_cat['name'] = cat['name']
            new_cat['id'] = old_cat_id_to_new_cat_id[cat['id']]
            new_cats.append(new_cat)

        self.categories = new_cats
        self.n_categories = len(self.categories)
        self.cat_name_to_cat_id = {cat['name']:cat['id'] for cat in self.categories}
        self.cat_id_to_cat_name = {cat['id']:cat['name'] for cat in self.categories}

        for ann in self.annotations:
            if ann['category_id'] in old_cat_id_to_new_cat_id:
                ann['category_id'] = old_cat_id_to_new_cat_id[ann['category_id']]

        self.ann_id_to_ann = {ann['id']:ann for ann in self.annotations}
        self.im_id_to_anns = {im['id']:[] for im in self.images}
        for ann in self.annotations:
            self.im_id_to_anns[ann['image_id']].append(ann)

    def get_location_info(self):
        if 'location' in self.images[0]:
            self.locations = list(set([im['location'] for im in self.images]))
            self.im_id_to_loc = {im['id']:im['location'] for im in self.images}
            self.loc_to_im_ids = {loc:[] for loc in self.locations}
            for im in self.images:
                self.loc_to_im_ids[im['location']].append(im['id'])

            self.loc_to_ann_ids = {loc:[] for loc in self.locations}
            for ann in self.annotations:
                self.loc_to_ann_ids[self.im_id_to_loc[ann['image_id']]].append(ann['id'])



    #TODO add new file
    #def CombineFiles(self,new_file)

