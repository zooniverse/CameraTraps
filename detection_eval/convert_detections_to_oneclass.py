import pickle
import argparse


def convert_detections_to_oneclass(detections):
    
    detections['detection_labels'] = [[1 for label in labels] for labels in detections['detection_labels']]

    detections['gt_labels'] = [[1 for label in labels] for labels in detections['gt_labels']]
 
    return detections

def parse_args():

    parser = argparse.ArgumentParser(description = 'Convert returned detections to oneclass for evaluation')
    
    parser.add_argument('--input_file', dest='input_file',
                         help='Path to detection pickle file returned by tfrecords/read_from_tfrecords.py',
                         type=str, required=True)
    parser.add_argument('--output_file', dest='output_file',
                         help='Path to store oneclass output dict',
                         type=str, required=True)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    print('Reading input file...')
    with open(args.input_file,'rb') as f:
        detections = pickle.load(f)

    print('Converting detections to oneclass...')
    oneclass_detections = convert_detections_to_oneclass(detections)

    print('Saving output file...')
    with open(args.output_file,'wb') as f:
        pickle.dump(oneclass_detections,f)

if __name__ == '__main__':
    main()
