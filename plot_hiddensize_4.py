# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 15,
        }

# ACC

bar_width=0.1

x = np.arange(3)
x_label = ["16","32","64"]


# acc_nc = [90.36,90.78,90.78]
# acc_emci = [92.11,93.43,92.10]
# acc_lmci = [83.01,85.14,87.67]
#
# mean_acc_nc = [3.49,4.25,4.25]
# mean_acc_emci = [5.38,2.72,5.38]
# mean_acc_lmci = [11.56,9.89,7.01]

# auc_nc = [92.83,93.83,93.83]
# auc_emci = [96.21,96.78,96.67]
# auc_lmci = [94.78,95.66,97.64]
#
# mean_auc_nc = [0.75,2.6,2.6]
# mean_auc_emci = [1.2,1.9,2.24]
# mean_auc_lmci = [4.9,4.7,1.4]

# sen_nc = [87.12,87.29,87.29]
# sen_emci = [94.57,94.57,94.14]
# mean_sen_nc = [5.42,7.50,7.5]
# mean_sen_emci = [2.85,2.85,5.71]

spe_nc = [92.65,91.22,91.22]
spe_emci = [89.29,90.79,90.54]

mean_spe_nc = [2.7,4.87,4.872]
mean_spe_emci = [7.64,2.64,5.14]


plt.bar(x, spe_nc,width=bar_width,yerr=mean_spe_nc,capsize=4,label='NC vs. EMCI')
plt.bar(x+bar_width,spe_emci,width=bar_width,yerr=mean_spe_emci,capsize=4,label='EMCI vs. LMCI')
# plt.bar(x+2*bar_width,auc_lmci,width=bar_width,align="center",yerr=mean_auc_lmci,capsize=4,label='NC vs. EMCI vs. LMCI')



plt.grid(axis='y',ls="-",color="purple",alpha=0.7)
plt.legend(loc='upper right')  # 让图例生效
plt.ylabel('SPE (%)', fontdict=font)
plt.xlabel('Hidden size of SA_BiGRU', fontdict=font)


plt.xticks(x+0.5*bar_width,x_label,fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.ylim(70,100)

plt.savefig('images/hiddensize/size_16.png', dpi=900)
# plt.savefig('images/hiddensize/emci_lmci.png', dpi=900)
# plt.savefig('images/hiddensize/nc_emci_lmci.png', dpi=900)
plt.show()



#

