#
# make_tfrecords_from_json.py
#
# Given a coco-camera-traps .json file, creates tfrecords
#

import json
import pickle
import numpy as np
import argparse

# from create_tfrecords_py3 import create
from create_tfrecords import create
from create_tfrecords_format import *


def make_tfrecords_from_json(input_json_file, output_tfrecords_folder, image_file_root, dataset_name, num_threads=5,ims_per_record=200,is_one_class=False): 
    #check if the input file has already been converted to the tfrecords format, if not, convert
    if 'tfrecord_format' in input_json_file:
        with open(database_file,'r') as f:
            dataset = json.load(f)
    else:
        dataset = create_tfrecords_format(input_json_file, image_file_root,is_one_class=is_one_class)

    print('Images: ',len(dataset))

    print('Creating '+dataset_name+' tfrecords')
    
    #calculate number of shards to get the desired number of images per record, ensure it is evenly divisible
    #by the number of threads
    num_shards = int(np.ceil(float(len(dataset))/ims_per_record))
    while num_shards % num_threads:
        num_shards += 1
    print('Number of shards: ' + str(num_shards))

    failed_images = create(
      dataset=dataset,
      dataset_name=dataset_name,
      output_directory=output_tfrecords_folder,
      num_shards=num_shards,
      num_threads=num_threads,
      store_images=True
    )

def parse_args():

    parser = argparse.ArgumentParser(description = 'Make tfrecords from a CCT style json file')

    parser.add_argument('--input_json_file', dest='input_json_file',
                         help='Path to .jon to create tfrecods from',
                         type=str, required=True)
    parser.add_argument('--output_tfrecords_folder', dest='output_tfrecords_folder',
                         help='Path to folder to save tfrecords in',
                         type=str, required=True)
    parser.add_argument('--image_file_root', dest='image_file_root',
                         help='Path to the folder the raw image files are stored in',
                         type=str, required=True)
    parser.add_argument('--dataset_name', dest='dataset_name',
                         help='name for the tfrecords, ex: train',
                         type=str, required=True)
    parser.add_argument('--num_threads', dest='num_threads',
                         help='Number of threads to use while creating tfrecords',
                         type=int, default=5)
    parser.add_argument('--ims_per_record', dest='ims_per_record',
                         help='Number of images to store in each tfrecord file',
                         type=int, default=200)
    parser.add_argument('--oneclass', dest='is_one_class',
                         help='Should the tfrecords be a oneclass version?',
                         action='store_true', default=False)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    make_tfrecords_from_json(args.input_json_file, args.output_tfrecords_folder, args.image_file_root,
                             args.dataset_name, num_threads=args.num_threads,
                             ims_per_record=args.ims_per_record,is_one_class=args.is_one_class)


if __name__ == '__main__':
    main()


