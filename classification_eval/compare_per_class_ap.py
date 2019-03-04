import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

split = 'imerit_deer'

file_1 = '/Users/sarabeery/Documents/CameraTrapClass/sim_classification/general/train_on_cct/_'+split+'_evaluation.p'
data_1 = pickle.load(open(file_1,'rb'))

file_2 = '/Users/sarabeery/Documents/CameraTrapClass/sim_classification/unity/unity_300K_with_deer/_'+split+'_evaluation.p'
data_2 = pickle.load(open(file_2,'rb'))

cats = data_1['classes']
ap = data_1['per_class_eval']['average_precision']
cat_id_to_cat = data_1['class_id_to_name']

cat_ids = [i for i in ap if not np.isnan(ap[i])]
print(cat_ids)

N = len(cat_ids)
ind = np.arange(N)
width = 0.3


fig = plt.figure()
ax = fig.add_subplot(111)
aps = [ap[i] for i in cat_ids]
print(aps)

print(len(ind),len(aps))
rects1 = ax.bar(ind, aps, width, color='royalblue')

ap = data_2['per_class_eval']['average_precision']
rects2 = ax.bar(ind+width, [ap[i] for i in cat_ids], width, color='seagreen')

ax.set_ylabel('mAP per class')
ax.set_title('mAP per class with and without sim data')
ax.set_xticks(ind + width/2 )
ax.set_xticklabels([cat_id_to_cat[i] for i in cat_ids])
plt.xticks(rotation=90)

ax.legend((rects1[0],rects2[0]),('CCT','CCT+sim'), loc='lower center')

plt.tight_layout()

plt.savefig('/Users/sarabeery/Documents/CameraTrapClass/sim_classification/per_class_eval_'+split+'.jpg')

#plt.show()
