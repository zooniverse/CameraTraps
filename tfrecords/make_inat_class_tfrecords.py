import json
import numpy as np
import argparse
import sys
import os
import random

from create_tfrecords import create
from create_tfrecords_format import create_tfrecords_format


def make_tfrecords_for_inat_experiments(input_annotation_folder, output_base_folder, image_file_root, 
                             max_images_in_train, num_threads=5, ims_per_record=200): 
    
    # check whether the input file has already been converted to the tfrecords format,
    # if not, convert
    files = os.listdir(input_annotation_folder)
    for filename in files:
        input_json_file = input_annotation_folder+'/'+ filename        
        filename = filename.split('_')
        superclass = filename[0]
        species = filename[1]
        num_ims = filename[2]
        output_tfrecords_folder = output_base_folder+superclass+'/'+species+'/tfrecords/'
        #input_json_file = input_annotation_folder+'/'+ filename
        if 'tfrecord_format' in input_json_file:
            with open(input_json_file,'r') as f:
                dataset = json.load(f)
        else:
            dataset = create_tfrecords_format(input_json_file, image_file_root)

        train = random.sample(dataset, min(max_images_in_train, len(dataset)-10))
        val = [i for i in dataset if i not in train]
        print(len(train), len(val), len(train)+len(val),len(dataset))
        print('Creating tfrecords for train from {} images'.format(len(train)))
    
        # Calculate number of shards to get the desired number of images per record, 
        # ensure it is evenly divisible by the number of threads
        dataset = train
        num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
        while num_shards % num_threads:
            num_shards += 1
        print('Number of shards: ' + str(num_shards))

        failed_train_images = create(
          dataset=dataset,
          dataset_name='train',
          output_directory=output_tfrecords_folder,
          num_shards=num_shards,
          num_threads=num_threads,
          store_images=True
        )

        print('Creating tfrecords for val from {} images'.format(len(val)))
    
        # Calculate number of shards to get the desired number of images per record, 
        # ensure it is evenly divisible by the number of threads
        dataset = val
        num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
        while num_shards % num_threads:
            num_shards += 1
        print('Number of shards: ' + str(num_shards))

        failed_val_images = create(
          dataset=dataset,
          dataset_name='val',
          output_directory=output_tfrecords_folder,
          num_shards=num_shards,
          num_threads=num_threads,
          store_images=True
        )


    return failed_train_images,failed_val_images

def parse_args():

    parser = argparse.ArgumentParser(description = 'Make tfrecords from a CCT style json file')

    parser.add_argument('--input_annotation_folder', dest='input_annotation_folder',
                         help='Path to folder containing .jsons to create tfrecords from',
                         type=str, required=True)
    parser.add_argument('--output_base_folder', dest='output_base_folder',
                         help='Path to base folder to save tfrecords in',
                         type=str, required=True)
    parser.add_argument('--image_file_root', dest='image_file_root',
                         help='Path to the folder the raw image files are stored in',
                         type=str, required=True)
    parser.add_argument('--max_images_in_train', dest='max_images_in_train',
                         help='max training images',
                         type=int, required=True)
    parser.add_argument('--num_threads', dest='num_threads',
                         help='Number of threads to use while creating tfrecords',
                         type=int, default=5)
    parser.add_argument('--ims_per_record', dest='ims_per_record',
                         help='Number of images to store in each tfrecord file',
                         type=int, default=200)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    make_tfrecords_for_inat_experiments(args.input_annotation_folder, args.output_base_folder, 
                             args.image_file_root, args.max_images_in_train, args.num_threads,
                             args.ims_per_record)


if __name__ == '__main__':
    main()
