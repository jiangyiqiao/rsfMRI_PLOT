# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt

bar_width=0.1

x = np.arange(3)
x_label = ["16","32","64"]

# # NC vs. EMCI
# acc = [93.36,92.78,92.78]
# sen = [89.12,90.29,90.29]
# spe = [97.65,95.22,95.22]
# fscore = [92.97,92.50,92.50]
# auc = [98.83,97.83,97.83]
# mean_acc = [4.49,5.25,5.25]
# mean_sen = [7.42,8.50,8.5]
# mean_spe = [4.7,6.87,6.872]
# mean_fscore = [4.96,5.54,5.543]
# mean_auc = [0.75,2.6,2.6]

# # EMCI vs. LMCI
# acc = [94.11,95.43,94.10]
# sen = [98.57,98.57,97.14]
# spe = [90.29,92.79,91.54]
# fscore = [94.06,95.21,93.83]
# auc = [99.21,98.78,98.67]
# mean_acc = [5.38,2.72,5.38]
# mean_sen = [2.85,2.85,5.71]
# mean_spe = [7.64,2.64,5.14]
# mean_fscore = [5.09,2.78,5.54]
# mean_auc = [1.2,1.9,2.24]

# # NC vs. EMCI false
# acc = [90.36,90.78,90.78]
# sen = [87.12,87.29,87.29]
# spe = [92.65,91.22,91.22]
# fscore = [90.97,90.50,90.50]
# auc = [92.83,93.83,93.83]
# mean_acc = [3.49,4.25,4.25]
# mean_sen = [5.42,7.50,7.5]
# mean_spe = [2.7,4.87,4.872]
# mean_fscore = [2.96,3.54,3.54]
# mean_auc = [0.75,2.6,2.6]

# # EMCI vs. LMCI
acc = [92.11,93.43,92.10]
sen = [94.57,94.57,94.14]
spe = [89.29,90.79,90.54]
fscore = [92.06,93.21,91.83]
auc = [96.21,96.78,96.67]
mean_acc = [5.38,2.72,5.38]
mean_sen = [2.85,2.85,5.71]
mean_spe = [7.64,2.64,5.14]
mean_fscore = [5.09,2.78,5.54]
mean_auc = [1.2,1.9,2.24]

plt.bar(x-bar_width*2, acc,width=bar_width,yerr=mean_acc,capsize=4,label='ACC')
plt.bar(x-bar_width,sen,width=bar_width,yerr=mean_sen,capsize=4,label='SEN')
plt.bar(x,spe,width=bar_width,align="center",yerr=mean_spe,capsize=4,label='SPE')
plt.bar(x+bar_width,fscore,width=bar_width,yerr=mean_fscore,capsize=4,label='F1_score')
plt.bar(x+bar_width*2,auc,width=bar_width,yerr=mean_auc,capsize=4,label='AUC')

# NC vs. EMCI vs. LMCI
# acc = [83.01,85.14,87.67]
# auc = [94.78,95.66,97.64]
# mean_acc = [11.56,9.89,7.01]
# mean_auc = [4.9,4.7,1.4]
# plt.bar(x, acc,width=bar_width,yerr=mean_acc,capsize=6,label='ACC')
# plt.bar(x+bar_width,auc,width=bar_width,yerr=mean_auc,label='AUC')


plt.grid(axis='y',ls="-",color="purple",alpha=0.7)
plt.legend(loc='lower right')  # 让图例生效

plt.xlabel('Number of hidden size')  # X轴标签
plt.ylabel("ACC(%)")  # Y轴标签
plt.xticks(x+bar_width/2,x_label)
plt.ylim(50,100)
# plt.title("NC vs. EMCI") #标题
plt.title("EMCI vs. LMCI") #标题
# plt.title("NC vs. EMCI vs. LMCI") #标题

# plt.savefig('images/hiddensize/nc_emci.png', dpi=900)
plt.savefig('images/hiddensize/emci_lmci.png', dpi=900)
# plt.savefig('images/hiddensize/nc_emci_lmci.png', dpi=900)
plt.show()

