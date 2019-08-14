import scipy.io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap():
    # 加載mat文件
    x_ticks = []
    y_ticks = []
    data = scipy.io.loadmat('data/corrMatrix_NC.mat')
    matrix = data['corrMatrix_NC']
    sns.heatmap(matrix,square=True,cmap="Reds")
    plt.xticks(rotation=90)  # 将字体进行旋转
    plt.yticks(rotation=90)
    plt.savefig('images/heatmap/NC_EMCI.png', pad_inches = 0.2, bbox_inches = 'tight')
    plt.show()


def get_matrix():
    # 讀取所有fMRI文件
    # fMRI文件路徑

    mat_filepath = 'data/NC_EMCI/kalman/kalmancorr_0.01_0.7.mat'

    # 加載mat文件
    data = scipy.io.loadmat(mat_filepath)

    # print(data['danao'])
    roi_data = data['corr']
    corr_data = np.array([np.array(roi_data[i][0], dtype=np.float32) for i in range(len(roi_data))])  # num*90*90*130

    sample_num, _, _, frame = corr_data.shape

    aa = np.zeros(8100 * sample_num).reshape(sample_num, 90, 90)
    bb = np.zeros(8100).reshape(90, 90)

    print(corr_data.shape)
    for k in range(sample_num):
        for i in range(90):
            for j in range(90):
                # print(corr_data[k][i][j].shape)
                # print(np.sum([corr_data[k][i][j]]))
                aa[k][i][j] = np.sum([corr_data[k][i][j]])
        aa[k] = aa[k] / frame * (-1)
    for k in range(sample_num):
        bb = np.sum([bb, aa[k]], axis=0)  # 0行加 1列加

    bb = bb / sample_num
    scipy.io.savemat('data/corrMatrix.mat', {'corrMatrix': bb})

    # 删除全0行和列
    delete_row = []
    delete_col = []

    sum_row = np.sum(bb, axis=1)  # 1 行
    sum_col = np.sum(bb, axis=0)  # 0 列
    for i in range(90):
        if sum_row[i] < 1:
            print('row:', i)
            delete_row.append(i)
        if sum_col[i] < 8:
            print('col:', i)
            delete_col.append(i)
    bbb = np.delete(bb, delete_row, axis=0)  # 0 行 1 列
    bbbb = np.delete(bbb, delete_col, axis=1)
    print(bbbb.shape)
    scipy.io.savemat('data/corrMatrix_NC.mat', {'corrMatrix_NC': bbbb})
    return bbbb


if __name__=='__main__':
    # matrix = get_matrix()
    plot_heatmap()
