# encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt


# 该改变方式 准确率分别比较
bar_width=0.2

x = np.arange(5)
x_label = ["ACC","SEN","SPE","F1_score","AUC"]


# # NC vs. EMCI
size_16 = [90.36,87.12,92.65,90.97,92.83]
size_32 = [90.78,87.29,91.22,90.50,93.83]
size_64 = [90.78,87.29,91.22,90.50,93.83]
mean_16 = [3.49,5.42,2.7,2.96,0.75]
mean_32 = [4.25,7.50,4.87,3.54,2.6]
mean_64 = [4.25,7.5,4.872,3.54,2.6]


# # EMCI vs. LMCI
# size_16 = [92.11,94.57,89.29,92.06,96.21]
# size_32 = [93.43,94.57,90.79,93.21,96.78]
# size_64 = [92.10,94.14,90.54,91.83,96.67]
# mean_16 = [5.38,2.85,7.64,5.09,1.2]
# mean_32 = [2.72,2.85,2.64,2.78,1.9]
# mean_64 = [5.38,5.71,5.14,5.54,2.24]



plt.bar(x, size_16,width=bar_width,yerr=mean_16,capsize=4,label='hidden size 16')
plt.bar(x+bar_width,size_32,width=bar_width,yerr=mean_32,capsize=4,label='hidden size 32')
plt.bar(x+2*bar_width,size_64,width=bar_width,align="center",yerr=mean_64,capsize=4,label='hidden size 64')



# # NC vs. EMCI vs. LMCI
# size_16 = [83.01,94.78]
# size_32 = [85.14,95.66]
# size_64 = [87.67,97.64]
# mean_16 = [11.56,4.9]
# mean_32 = [9.89,4.7]
# mean_64 = [7.01,1.4]


# x = np.arange(2)
# x_label = ["ACC","AUC"]
# bar_width=0.1
# plt.bar(x, size_16,width=bar_width,yerr=mean_16,capsize=4,label='hidden size 16')
# plt.bar(x+bar_width,size_32,width=bar_width,yerr=mean_32,capsize=4,label='hidden size 32')
# plt.bar(x+bar_width*2,size_64,width=bar_width,align="center",yerr=mean_64,capsize=4,label='hidden size 64')


plt.grid(axis='y',ls="-",color="purple",alpha=0.7)
plt.legend(loc='lower right')  # 让图例生效


from matplotlib.ticker import FuncFormatter
plt.rcParams['font.family'] = ['Times New Roman']
def to_percent(temp, position):
    return '%1.0f'%(temp) + '%'
plt.gca().yaxis.set_major_formatter(FuncFormatter(to_percent))
plt.xticks(x+bar_width,x_label)
plt.ylim(70,100)

plt.title("NC vs. EMCI") #标题
# plt.title("EMCI vs. LMCI") #标题
# plt.title("NC vs. EMCI vs. LMCI") #标题

plt.savefig('images/hiddensize/nc_emci.png', dpi=900)
# plt.savefig('images/hiddensize/emci_lmci.png', dpi=900)
# plt.savefig('images/hiddensize/nc_emci_lmci.png', dpi=900)
plt.show()

