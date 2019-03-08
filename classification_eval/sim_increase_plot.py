import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

def get_metric_from_files(data_list,metric):
    deer_metric = []
    other_metric_common = []
    other_metric_rare = []

    common = 10

    for data in data_list:
        per_class_metric = data['per_class_eval'][metric]
        support = data['per_class_eval']['per_class_support']
        print(support)
        if 0 not in support:
            support[0] = 0
        print(per_class_metric)
        deer_metric.append(1-per_class_metric[deer_id])
        if metric == 'per_class_auc':
            other_metric_common.append(np.mean([1-per_class_metric[i] for i in per_class_metric if i != deer_id and not np.isnan(per_class_metric[i]) and support[i] >= common]))
            other_metric_rare.append(np.mean([1-per_class_metric[i] for i in per_class_metric if i != deer_id and not np.isnan(per_class_metric[i]) and support[i] < common and support[i] > 10]))
        else:
            other_metric_common.append(np.mean([1-per_class_metric[i] for i in per_class_metric if i != deer_id and per_class_metric[i] is not None and support[i] >= common]))
            other_metric_rare.append(np.mean([1-per_class_metric[i] for i in per_class_metric if i != deer_id and per_class_metric[i] is not None and support[i] < common and support[i] > 10]))

    return deer_metric, other_metric_common, other_metric_rare


deer_id = 13

metric = 'per_class_prec'

num_sim = [0,100000,300000]

split = 'trans_test_with_imerit'

data = []

file_1 = '/Users/sarabeery/Documents/CameraTrapClass/sim_classification/general/train_on_cct/results/'+split+'_eval_dict.p'
data.append(pickle.load(open(file_1,'rb')))

file_2 = '/Users/sarabeery/Documents/CameraTrapClass/sim_classification/unity/model_2_100K/results/'+split+'_eval_dict.p'
data.append(pickle.load(open(file_2,'rb')))

file_3 = '/Users/sarabeery/Documents/CameraTrapClass/sim_classification/unity/unity_300K_with_deer/results/'+split+'_eval_dict.p'
data.append(pickle.load(open(file_3,'rb')))

deer_metric, other_metric_common, other_metric_rare = get_metric_from_files(data,metric)


plt.plot(num_sim,deer_metric,'r:', label='trans deer')
plt.plot(num_sim,other_metric_common,'r--', label='trans common other classes (avg)')
#plt.plot(num_sim,other_metric_rare,'r-', label='trans rare other classes (avg)')


split = 'cis_test'

data = []

file_1 = '/Users/sarabeery/Documents/CameraTrapClass/sim_classification/general/train_on_cct/results/'+split+'_eval_dict.p'
data.append(pickle.load(open(file_1,'rb')))

file_2 = '/Users/sarabeery/Documents/CameraTrapClass/sim_classification/unity/model_2_100K/results/'+split+'_eval_dict.p'
data.append(pickle.load(open(file_2,'rb')))

file_3 = '/Users/sarabeery/Documents/CameraTrapClass/sim_classification/unity/unity_300K_with_deer/results/'+split+'_eval_dict.p'
data.append(pickle.load(open(file_3,'rb')))

deer_metric, other_metric_common, other_metric_rare = get_metric_from_files(data,metric)

plt.plot(num_sim,deer_metric,'b:', label='cis deer')
plt.plot(num_sim,other_metric_common,'b--', label='cis common other classes (avg)')
#plt.plot(num_sim,other_metric_rare,'b-', label='cis rare other classes (avg)')

plt.title('Error vs. number of simulated images')
plt.xlabel('Number of simulated images')
plt.ylabel('Error: top-1 accuracy')
plt.legend()


plt.show()

# metric = {i:[] for i in num_sim}

# for i,data in enumerate([data_1,data_2,data_3]):
#   per_class_metric = data['per_class_eval'][metric]
#   support = data['per_class_eval']['per_class_support']
#   for cat in per_class_metric:
#       per


