#
# make_tfrecords_with_train_test_split.py
#
# Given a .json file that contains a three-element list (train/val/test) of image IDs and a .json database that contains
# those image IDs, generates tfrecords whose filenames include "train"/"val"/"test"
# 

import json
import random
import pickle
import numpy as np
from create_tfrecords import create
from create_tfrecords_format import create_tfrecords_format
from create_classification_tfrecords_format import create_classification_tfrecords_format
import tensorflow as tf

output_dir = '/home/ubuntu/efs/eccv_cct_tfrecords/multiclass/'
database_folder = '/home/ubuntu/cct_data/eccv_18_annotation_files/'
#output_dir = '/ss_data/tfrecords'
image_file_root = '/home/ubuntu/cct_data/cct_images/'
experiment_type = 'detection'
ims_per_record = 500.0
num_threads = 5

'''
data = create_tfrecords_format(database_file, image_file_root)

print('Images: ',len(data))
print(data[0])
data_split = json.load(open(datafolder+'databases/snapshotserengeti/oneclass/SnapshotSerengeti_Seasons_1_to_4_classification_train_test_split.json'))
im_id_to_im = {im['id']:im for im in data}

train = [i for i in data_split['train_ims'] if i in im_id_to_im]
trans_val = [i for i in data_split['val_ims'] if i in im_id_to_im]
trans_test = [i for i in data_split['test_ims'] if i in im_id_to_im]

print('train: ', len(train), ', val: ', len(trans_val), ' test: ', len(trans_test) )
'''
print('Creating train tfrecords')
database_file = database_folder+'train_annotations.json'
dataset = create_tfrecords_format(database_file, image_file_root)
random.shuffle(dataset)
print(dataset[0])
num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
while num_shards % num_threads:
    num_shards += 1
print(num_shards)
failed_images = create(
  dataset=dataset,
  dataset_name="train",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=num_threads,
  store_images=True
)

print('Creating cis_val tfrecords')
#dataset = [im_id_to_im[idx] for idx in cis_val]
database_file = database_folder+'cis_val_annotations.json'
dataset = create_tfrecords_format(database_file, image_file_root)
random.shuffle(dataset)
num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
while num_shards % num_threads:
    num_shards += 1
failed_images = create(
  dataset=dataset,
  dataset_name="cis_val",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5,
  store_images=True
)


print('Creating trans_val tfrecords')
#dataset = [im_id_to_im[idx] for idx in trans_val]
database_file = database_folder+'trans_val_annotations.json'
dataset = create_tfrecords_format(database_file, image_file_root)
random.shuffle(dataset)
num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
while num_shards % num_threads:
    num_shards += 1
failed_images = create(
  dataset=dataset,
  dataset_name="trans_val",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5,
  store_images=True
)


print('Creating cis_test tfrecords')
#dataset = [im_id_to_im[idx] for idx in cis_test]
database_file = database_folder+'cis_test_annotations.json'
dataset = create_tfrecords_format(database_file, image_file_root)
random.shuffle(dataset)
num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
while num_shards % num_threads:
    num_shards += 1
failed_images = create(
  dataset=dataset,
  dataset_name="cis_test",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5,
  store_images=True
)

print('Creating trans_test tfrecords')
#dataset = [im_id_to_im[idx] for idx in trans_test]
database_file = database_folder+'trans_test_annotations.json'
dataset = create_tfrecords_format(database_file, image_file_root)
random.shuffle(dataset)
num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
while num_shards % num_threads:
    num_shards += 1
failed_images = create(
  dataset=dataset,
  dataset_name="trans_test",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5,
  store_images=True
)


