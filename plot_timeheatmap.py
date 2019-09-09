import scipy.io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plotFeature():
    # mat_filepath = 'data/NC/kalmancorr/kalmancorr_0.01_0.6.mat'
    mat_filepath = 'data/EMCI/kalmancorr/kalmancorr_0.01_0.6.mat'
    # mat_filepath = 'data/LMCI/kalmancorr/kalmancorr_0.01_0.6.mat'

    # 加載mat文件
    data = scipy.io.loadmat(mat_filepath)

    # print(data['danao'])
    roi_data = data['corr']
    corr_data = np.array([np.array(roi_data[i][0], dtype=np.float32) for i in range(len(roi_data))])  # num*90*90*130

    sample_num, _, _, frame = corr_data.shape
    corr_data= corr_data.transpose((0,3,1,2))
    aa = np.zeros(frame * 90).reshape(frame, 90)

    print(corr_data.shape)
    for k in range(frame):
            for j in range(90):
                if k==0:
                    corr_data[2][k][60][j] = 0
                if corr_data[2][k][60][j] !=0 and k!=0:
                    corr_data[2][k][60][j] = (corr_data[2][k][60][j]+1)*10
                    print(corr_data[2][k][60][j])

    for k in range(frame):
        aa[k] = corr_data[2][k][60]

    x_ticklabel = list(range(90))
    y_ticklabel = list(range(130))
    f, ax = plt.subplots(figsize=(14,10))
    sns.heatmap(aa,cmap="Reds",square=True,mask=(aa==0),ax=ax,xticklabels=5, yticklabels=10)
    ax.set_title('Correlation changes in brain regions at different time points (EMCI)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=-45)
    ax.set_x
    # plt.savefig('images/heatmap/NC.png', pad_inches = 0.5, bbox_inches = 'tight',dpi=1200)
    # plt.savefig('images/heatmap/EMCI.png', pad_inches=0.5, bbox_inches='tight', dpi=1200)
    # plt.savefig('images/heatmap/LMCI.png', pad_inches=0.5, bbox_inches='tight', dpi=1200)
    plt.show()


if __name__=='__main__':
    plotFeature()
