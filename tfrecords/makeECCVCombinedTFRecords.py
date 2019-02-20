import json
import pickle
import random
import sys
#sys.path.insert(0, '/home/ubuntu/efs/tfrecords/')
from create_tfrecords import create
import numpy as np

#val_set_percentage = 0.1
num_ims_per_shard = 250

vis_annotation_file = 'camera_trap_database/visipedia_ECCV_bboxes_clean_with_sequences_binary.json'


with open(vis_annotation_file,'r')  as f:
    data = json.load(f)
print('Total images: ' + str(len(data)))

train_test_split = pickle.load(open('camera_trap_database/comb_train_test_split_no_empty.p','rb'))
train = pickle.load(open('camera_trap_database/train_ids_to_keep.p','rb'))
#train_test_split = pickle.load(open('camera_trap_database/multibox_inter_train_test_split.p','rb'))

output_dir = "/home/ubuntu/efs/datasets/eccv_datasets/combined_images_binary/"

#train_dataset_ids = train_test_split[0]
train_dataset_ids = train
val_dataset_ids_inter = train_test_split[1]
test_dataset_ids_inter = train_test_split[2]
val_dataset_ids_loc = train_test_split[3]
test_dataset_ids_loc = train_test_split[4]
train_dataset = []
test_dataset = []
val_dataset = []
test_dataset_2 = []
val_dataset_2 = []

for im in data:
    if im['id'] in val_dataset_ids_inter:
	val_dataset.append(im)
    if im['id'] in val_dataset_ids_loc:
        val_dataset_2.append(im)
    elif im['id'] in train_dataset_ids:
	train_dataset.append(im)
    elif im['id'] in test_dataset_ids_inter:
        test_dataset.append(im)
    elif im['id'] in test_dataset_ids_loc:
	test_dataset_2.append(im)
     
random.shuffle(train_dataset)
random.shuffle(val_dataset)
random.shuffle(test_dataset)
random.shuffle(val_dataset_2)
random.shuffle(test_dataset_2)

print('Testing images: ' + str(len(test_dataset))) 
print('Training Images: ' + str(len(train_dataset)))
print('Validation Images: ' + str(len(val_dataset)))

print('creating inter testing dataset')
num_shards = np.round(len(test_dataset)/num_ims_per_shard)
while num_shards % 5 != 0:
	num_shards += 1
 
print(num_shards)
failed_images = create(
  dataset=test_dataset,
  dataset_name="test_inter",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5
    )
  
print('creating location testing dataset')
num_shards = np.round(len(test_dataset_2)/num_ims_per_shard)
while num_shards % 5 != 0:
        num_shards += 1

print(num_shards)
failed_images = create(
  dataset=test_dataset_2,
  dataset_name="test_loc",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5
    )

print('creating training dataset')
num_shards = np.round(len(train_dataset)/num_ims_per_shard)
while num_shards % 5 != 0:
    num_shards += 1
failed_images = create(
  dataset=train_dataset,
  dataset_name="train",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5
)
print('creating inter validation dataset')
num_shards = np.round(len(val_dataset)/num_ims_per_shard)
while num_shards % 5 != 0:
    num_shards += 1
failed_images = create(
  dataset=val_dataset,
  dataset_name="val_inter",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5
)
print('creating location validation dataset')
num_shards = np.round(len(val_dataset_2)/num_ims_per_shard)
while num_shards % 5 != 0:
    num_shards += 1
failed_images = create(
  dataset=val_dataset_2,
  dataset_name="val_loc",
  output_directory=output_dir,
  num_shards=num_shards,
  num_threads=5
)
