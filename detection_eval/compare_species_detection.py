import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#file_1 = '/home/ubuntu/efs/airsim_experiments/train_airsim_only/exported_models/predictions/eccv_trans_test_per_cat_prec_recall_data.npz'
file_1 = '/home/ubuntu/efs/models/train_on_eccv_and_imerit_2/predictions/trans_test_per_cat_prec_recall_data.npz'
data_1 = np.load(open(file_1,'r'))

file_2 = '/home/ubuntu/efs/airsim_experiments/train_cct_and_airsim/exported_models/predictions/eccv_trans_test_per_cat_prec_recall_data.npz'
data_2 = np.load(open(file_2,'r'))

save_folder='/home/ubuntu/efs/airsim_experiments/train_cct_and_airsim/exported_models/predictions/'

ap = data_1['ap'].tolist()
cat_id_to_cat = data_1['cat_id_to_cat'].tolist()

cat_ids = [i for i in ap if not np.isnan(ap[i])]
print(cat_ids)

N = len(cat_ids)
ind = np.arange(N)
width = 0.35


fig = plt.figure()
ax = fig.add_subplot(111)
aps = [ap[i] for i in cat_ids]
print(aps)

print(len(ind),len(aps))
rects1 = ax.bar(ind, aps, width, color='royalblue')

ap = data_2['ap'].tolist()
rects2 = ax.bar(ind+width, [ap[i] for i in cat_ids], width, color='seagreen')

ax.set_ylabel('mAP per class')
ax.set_title('mAP per class training on CCT with and without airsim data')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels([cat_id_to_cat[i] for i in cat_ids])
plt.xticks(rotation=90)

ax.legend((rects1[0],rects2[0]),('cct','cct+airsim'))

plt.tight_layout()

plt.savefig(save_folder+'compare_per_seq_mAP_cct_and_cct_plus_airsim.jpg')



















