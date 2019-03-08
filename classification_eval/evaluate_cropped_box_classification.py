#python evaluate_cropped_box_classification.py --database_json_file /Users/sarabeery/Documents/CameraTrapClass/Fixing_CCT_Anns/Corrected_versions/CombinedBBoxAndECCV18.json --results_file /Users/sarabeery/Documents/CameraTrapClass/sim_classification/general/train_on_cct/unity_night.p --alternate_category_file /Users/sarabeery/Documents/CameraTrapClass/code/CameraTraps/tfrecords/eccv_categories.json --save_folder /Users/sarabeery/Documents/CameraTrapClass/sim_classification/general/train_on_cct/unity_night_eval_dict.p


import json
import pickle
import numpy as np
import argparse
import sys
import os
sys.path.append('/Users/sarabeery/Documents/CameraTrapClass/code/CameraTraps/database_tools/')

from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score, accuracy_score, roc_auc_score, auc
#from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import itertools

from data_loader_coco_format import COCODataLoader
from results_loader import ResultsLoader

def load_database(data_file, image_file_root, alternate_category_file=None, cats_to_ignore=['empty']):
    data = COCODataLoader(data_file, image_file_root)
    if alternate_category_file is not None:
        data.update_category_list(alternate_category_file, cats_to_ignore=cats_to_ignore)

    return data

def load_results(result_file,image_file_root=None,database=None):
    results = ResultsLoader(result_file,database=database)
    if image_file_root is not None:
        results.get_paths_to_images(image_file_root,database=database,data_type='bbox')
    results.get_bbox_list(database)

    return results

def get_tp_fp(results,cats):
    TP = {i:0 for i in cats}
    FP = {i:0 for i in cats}
    TN = {i:0 for i in cats}
    FN = {i:0 for i in cats}
    total_ims = {i:0 for i in cats}

    for i in range(len(results.ids)):
        total_ims[results.labels[i]] += 1
        if results.labels[i] == results.preds[i]:
            TP[results.labels[i]] +=1
            for j in cats:
                if j != i:
                    TN[j] += 1
        else:
            FP[results.preds[i]] += 1
            FN[results.labels[i]] += 1

    return TP,FP,TN,FN, total_ims

def calculate_per_class_pr(results,cats):
    TP,FP,TN,FN,total_ims = get_tp_fp(results,cats)
    per_class_prec = {}
    per_class_rec = {}
    for i in cats:
        if total_ims[i] > 0:
            if TP[i] > 0:
                per_class_prec[i] = TP[i]/float(TP[i]+FP[i])
                per_class_rec[i] = TP[i]/float(TP[i]+FN[i])
            else:
                per_class_prec[i] = 0
                per_class_rec[i] = 0
        else:
            per_class_prec[i] = None
            per_class_rec[i] = None

    return per_class_prec,per_class_rec

def calculate_per_class_acc(results,cats):
    TP,FP,TN,FN,total_ims = get_tp_fp(results,cats)
    per_class_acc = {}

    for i in cats:
        if total_ims[i] > 0:
            per_class_acc[i] = TP[i]/float(total_ims[i])
        else:
            per_class_acc[i] = None

    return per_class_acc, total_ims


def per_class_eval(results, cats):

    Y_test = label_binarize(results.labels, classes=range(len(cats)))
    y_score = np.asarray(results.probs)
    n_classes = len(cats)

    precision = dict()
    recall = dict()
    average_precision = dict()
    per_class_auc = dict()
    classes = []
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
        per_class_auc[i] = auc(recall[i],precision[i])
        print(i,average_precision[i])

    per_class_acc, support = calculate_per_class_acc(results,cats)
    per_class_prec,per_class_rec = calculate_per_class_pr(results,cats)

    return {'precision':precision, 'recall':recall, 'average_precision':average_precision,'classes':classes,'per_class_acc':per_class_acc,'per_class_prec':per_class_prec,'per_class_rec':per_class_rec,'per_class_auc':per_class_auc,'per_class_support':support}

def night_day_eval(results, night_day):

    Y_test = label_binarize(results.labels, classes=range(len(cats)))
    y_score = np.asarray(results.probs)
    n_classes = len(cats)

    precision = dict()
    recall = dict()
    average_precision = dict()
    per_class_auc = dict()
    classes = []
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
        per_class_auc[i] = auc(recall[i],precision[i])

    per_class_acc = calculate_per_class_acc(results,cats)
    per_class_prec,per_class_rec = calculate_per_class_pr(results,cats)

    return {'precision':precision, 'recall':recall, 'average_precision':average_precision,'classes':classes,'per_class_acc':per_class_acc,'per_class_prec':per_class_prec,'per_class_rec':per_class_rec,'per_class_auc':per_class_auc}


def per_location_eval(data,results):
    data.get_location_info()

def accuracy(results):
    acc = accuracy_score(results.labels,results.preds)
    return acc

def balanced_accuracy(results):
    #this is average per-class accuracy, aka average per-class recall
    balanced_acc = balanced_accuracy_score(results.labels,results.preds)
    return balanced_acc

def average_precision(results, cats):

    Y_test = label_binarize(results.labels, classes=range(len(cats)))
    y_score = np.asarray(results.probs)
    n_classes = len(cats)

    precision, recall, _ = precision_recall_curve(Y_test.ravel(),y_score.ravel())

    micro_avg = average_precision_score(Y_test, y_score,average="micro")

    return  {'micro_avg':micro_avg, 'micro_prec':precision,'micro_rec':recall}


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          axisNum = 1,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        if axisNum==1:
            cm = cm.astype('float') / cm.sum(axis=axisNum)[:, np.newaxis]
        else:
            cm = cm.astype('float') / cm.sum(axis=axisNum)[np.newaxis, :]


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=7)
    plt.yticks(tick_marks, classes, fontsize=7)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_per_class_prec_rec(eval_dict):

    n_classes = len(eval_dict['classes'])
    plt.figure(figsize=(7, 8))
    lines = []
    labelNames = []
    if 'average_precision_recall' in eval_dict:
        l, = plt.plot(eval_dict['average_precision_recall']['micro_rec'], eval_dict['average_precision_recall']['micro_prec'], color='grey', lw=15, alpha=0.3)
        lines.append(l)
        labelNames.append('micro-average (area = {0:0.2f})'
                    ''.format(eval_dict['average_precision_recall']['micro_avg']))
    for i in range(n_classes):
        if not np.isnan(eval_dict['per_class_eval']['average_precision'][i]):
            l, = plt.plot(eval_dict['per_class_eval']['recall'][i], eval_dict['per_class_eval']['precision'][i], lw=2)
            lines.append(l)
            labelNames.append('{0} (area = {1:0.2f})'
                    ''.format(eval_dict['class_id_to_name'][i], eval_dict['per_class_eval']['average_precision'][i]))

    fig = plt.gcf()
    ax = plt.subplot(111)
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall for each class')
    box = ax.get_position()
    ax.legend(lines, labelNames,loc='center left', bbox_to_anchor=(1.005, 0.5),prop=dict(size=7))
    plt.tight_layout(rect=[0, 0.2, 0.75, 0.8])
    return plt


def evaluate_classification(data, results):

    eval_dict = {}

    eval_dict['classes'] = [cat['id'] for cat in data.categories]

    eval_dict['class_id_to_name'] = {i:data.cat_id_to_cat_name[i] for i in eval_dict['classes']}

    eval_dict['accuracy'] = accuracy(results)

    #eval_dict['balanced_accuracy'] = balanced_accuracy(results)

    eval_dict['per_class_eval'] = per_class_eval(results,eval_dict['classes'])

    eval_dict['average_precision_recall'] = average_precision(results,eval_dict['classes'])


    return eval_dict



def parse_args():
    parser = argparse.ArgumentParser(description = 'Make tfrecords from a CCT style json file')

    parser.add_argument('--database_json_file', dest='database_json_file',
                         help='Path to .json database for data you are evaluating',
                         type=str, required=True)
    parser.add_argument('--results_file', dest='results_file',
                         help='Path to a .npz file of results returned from Visipedia/tf_classification/classify.py',
                         type=str, required=True)
    parser.add_argument('--alternate_category_file', dest='alternate_category_file',
                         help='Path to a category file to replace the one in your database, if you want to map to different category ids',
                         type=str, required=False, default=None)
    parser.add_argument('--save_folder', dest='save_folder',
                         help='File path to save data at',
                         type=str, required=False, default=None)
    parser.add_argument('--image_file_root', dest='image_file_root',
                         help='Path to the folder the raw image files are stored in',
                         type=str, required=False, default=None)
    parser.add_argument('--data_split_to_evaluate', dest='data_split_to_evaluate',
                         help='name of the data split to evaluate, mutch match names in data_split_file',
                         type=str, required=False, default=None)
    parser.add_argument('--data_split_file', dest='data_split_file',
                         help='path to a json file containing a dict of split_name:[split_ids]',
                         type=str, required=False, default=None)    
    parser.add_argument('--plot', dest='plot',
                         help='Should we visualize and save plots',
                         action='store_true', required=False)    

    args = parser.parse_args()

    return args


def main(plot=True):
    args = parse_args()
    
    print('Loading database...')
    data = load_database(args.database_json_file, args.image_file_root, args.alternate_category_file)

    data.print_stats()

    print('Loading results...')
    results = load_results(args.results_file, database=data)


    if args.data_split_to_evaluate is not None:
        if args.data_split_file is not None:
            split = json.load(open(args.data_split_file,'r'))
            assert args.data_split_to_evaluate in split
            print('Evaluating on split: '+args.data_split_to_evaluate)
            image_ids = set(split[args.data_split_to_evaluate])
            ann_ids = []
            print(list(image_ids)[0])
            print(data.im_id_to_anns[list(image_ids)[0]])
            for idx in image_ids:
                anns = [ann['id'] for ann in data.im_id_to_anns[idx]]
                ann_ids.extend(anns)
            results.evaluate_on_subset(ann_ids)
        else:
            print('Evaluating on all valid results')
    else:
        print('Evaluating on all valid results')

    results.print_stats()

    eval_dict = evaluate_classification(data, results)

    print(classification_report(results.labels,results.preds))

    print('Accuracy: '+str(eval_dict['accuracy']))
    #print(eval_dict['per_class_eval']['per_class_acc'])
    print('Balanced Accuracy: '+str(np.mean([i for i in eval_dict['per_class_eval']['per_class_acc'].values() if i is not None])))
    #print(eval_dict['per_class_eval']['per_class_auc'])

    if args.save_folder is not None:
        print(args.save_folder)
        print(args.data_split_to_evaluate)
        print('saving file at: '+args.save_folder+args.data_split_to_evaluate+'_eval_dict.p')
        pickle.dump(eval_dict,open(args.save_folder+args.data_split_to_evaluate+'_eval_dict.p','wb'))

    if plot:
        plt = plot_per_class_prec_rec(eval_dict)
        plt.savefig(args.save_folder+'_'+args.data_split_to_evaluate+'_per_class_prec_rec.jpg')


if __name__ == '__main__':
    main()
