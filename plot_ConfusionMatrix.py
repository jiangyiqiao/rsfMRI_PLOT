from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# Only use the labels that appear in the data
cmap=plt.cm.Blues

# NC EMCI

# labels_name = ['NC', 'EMCI']
# cm = np.asarray([[73,10],[6,78]])
# EMCI LMCI
# labels_name = ['EMCI', 'LMCI']
# cm = np.asarray([[75,9],[1,70]])
# # NC EMCI LMCI
labels_name = ['NC','EMCI', 'LMCI']
cm = np.asarray([[73,2,8],[3,79,2],[9,1,61]])
print(type(cm))
print(cm)


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    ax.figure.colorbar(im,ax=ax)
    num_local = np.array(range(len(labels_name)))

    plt.xticks(num_local, labels_name)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    # 结果显示在网格上
    fmt = '.2f'

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),size=15,ha="center", va="center",
                    color="black" if i!=j else "white")

plot_confusion_matrix(cm, labels_name, "Normalized confusion matrix")

plt.show()
plt.savefig('data/confusion_'+str(labels_name)+'.png',dpi=600)