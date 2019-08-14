# encoding=utf-8
from matplotlib import pyplot
import matplotlib.pyplot as plt


x = [i/10 for i in range(9)]
x_label = [i/10 for i in range(10)]
#
# y_1 = [86.82,86.82,85.63,86.22,86.82,86.22,86.22,86.22,86.22]
# y_2 = [88.02,88.62,88.02,88.02,88.62,88.62,89.82,88.62,88.62]
# y_3 = [89.22,89.22,89.22,89.82,89.22,89.82,90.42,89.82,90.42]


# y_1 = [88.38,89.03,89.03,87.1,89.03,89.03,88.38,89.03,89.03]
# y_2 = [90.97,90.97,90.97,89.03,90.32,90.97,89.03,90.97,90.97]
# y_3 = [91.61,91.61,91.61,90.32,91.61,91.61,90.32,90.32,90.32]

y_1 = [86.13,87.3,86.13,86.13,87.3,85.29,84.45,84.45,84]
y_2 = [87.39,87.39,87.39,86.13,86.13,87.5,86.,86.13,87.39]
y_3 = [87.39,87.82,86.55,86.55,89.5,87.39,85.29,87.39,86.13]

plt.plot(x, y_1, marker='+', color='black',ms=5, mec='r', mfc='w', label='proposed SA-BiGRU')
plt.plot(x, y_2, marker='*', ms=5, label='BiLSTM')
plt.plot(x, y_3, marker='o', color='red',ms=6, label='BiGRU')
plt.legend()  # 让图例生效

plt.xlabel('Obeserved ROI noise covariance (R1)')  # X轴标签
plt.ylabel("ACC(%)")  # Y轴标签
pyplot.yticks([75, 80, 85, 90, 95])
# plt.title("NC vs. EMCI") #标题
# plt.title("EMCI vs. LMCI") #标题
plt.title("NC vs. EMCI vs. LMCI") #标题
# plt.savefig('images/zhexian_acc/nc_emci.png', dpi=900)
# plt.savefig('images/zhexian_acc/emci_lmci.png', dpi=900)
plt.savefig('images/zhexian_acc/nc_emci_lmci.png', dpi=900)
plt.show()

