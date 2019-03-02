import numpy as np
import pickle
import os
import random
from PIL import ImageStat


class ResultsLoader():

    def __init__(self, result_file, database=None):
        self.result_file = result_file
        if '.npz' in self.result_file:
            data = np.load(open(self.result_file,'r'))
        else:
            data = pickle.load(open(self.result_file,'rb'))
        ids = data['ids']
        idx = [i for i in range(len(ids)) if ids[i] is not None]
        self.preds = data['labels'][idx]
        self.ids = data['ids'][idx]
        self.logits = data['logits'][idx]
        self.probs = np.asarray([self.softmax(logit) for logit in self.logits.tolist()])
        self.labels = None
        if database is not None:
            self.get_ground_truth(database)
            self.get_correctly_classified_idx()

        if 'activations' in data:
            self.activations = data['activations']
            self.layer_names = data['layer_names']

    def softmax(self,logit):
        prob = [np.exp(i)/sum(np.exp(logit)) for i in logit]
        return prob

    def print_stats(self):
        print('There are '+str(len(self.ids))+' valid results')

    def evaluate_on_subset(self,keep_ids):
        keep_ids = set(keep_ids)
        ids = self.ids
        idx = [i for i in range(len(ids)) if ids[i] in keep_ids]
        print(len(idx))
        self.preds = self.preds[idx]
        self.ids = self.ids[idx]
        self.logits = self.logits[idx]
        self.probs = self.probs[idx]
        if self.labels is not None:
            self.labels = self.labels[idx]

    def get_ground_truth(self,database,data_type='bbox'):
        #json_data is class COCODataLoader
        #data_type is 'image' or 'bbox'
        
        assert data_type in ['image','bbox']

        labels = []
        for idx in self.ids:
            if data_type == 'image':
                labels.append(database.im_id_to_ann[idx]['category_id'])

            else:
                labels.append(database.ann_id_to_ann[idx]['category_id'])

        self.labels = np.asarray(labels)

    def get_paths_to_images(self,image_file_root,database,data_type='bbox'):
        assert data_type in ['bbox','image']

        if data_type == 'bbox':
            self.paths = np.asarray([image_file_root+database.ann_id_to_ann[idx]['image_id']+'.jpg' for idx in self.ids])
        else:
            self.paths = np.asarray([image_file_root+self.ids[idx]+'.jpg' for idx in self.ids])


    def get_bbox_list(self, database):
        self.bboxes = np.asarray([database.ann_id_to_ann[idx]['bbox'] for idx in self.ids])

    def get_correctly_classified_idx(self):
        self.correct = np.zeros(len(self.ids))
        print(list(set(self.preds)),list(set(self.labels)))
        for i in range(len(self.ids)):
            if self.labels[i] == self.preds[i]:
                self.correct[i] = 1

        print(sum(self.correct))

    def get_flattened_layer_activations(self,layer_index):
        layer = self.activations[layer_index]
        image_idxs = range(len(self.labels))
        X = []
        for im_idx in image_idxs:
            vec = layer[im_idx]
            X.append(vec.flatten())
    
        X = np.asarray(X)
        return X
