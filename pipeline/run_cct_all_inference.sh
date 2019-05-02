#!/bin/bash

export PYTHONPATH=$PYTHONPATH:`pwd`/../../tfmodels/research:`pwd`/../../tfmodels/research/slim:`pwd`/../tfrecords:`pwd`/../detection_eval

EXPERIMENT="cct_iwildcam"

MODEL="/home/ubuntu/efs/models/megadetector/"

DETECTION_TFRECORD_FILE="/home/ubuntu/efs/models/megadetector_predictions/"$EXPERIMENT"_detections.tfrecord-00000-of-00001" 

DETECTION_DICT_FILE="/home/ubuntu/efs/models/megadetector_predictions/"$EXPERIMENT".p"

TF_RECORD_FILES=$(ls -1 /home/ubuntu/efs/tfrecords/cct_iwildcam/* | tr '\n' ',')

python ../../tfmodels/research/object_detection/inference/infer_detections.py --input_tfrecord_paths=$TF_RECORD_FILES --output_tfrecord_path=$DETECTION_TFRECORD_FILE --inference_graph=$MODEL/frozen_inference_graph.pb --discard_image_pixels

python ../tfrecords/read_from_tfrecords.py --input_tfrecord_file $DETECTION_TFRECORD_FILE --output_file $DETECTION_DICT_FILE 


