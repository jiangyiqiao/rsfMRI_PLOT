import scipy.io
import numpy as np
# label_name = 'nc'
label_name = 'emci'
# label_name = 'lmci'



with open('data/brainNet/Edge_AAL90_Weighted_'+label_name+'.edge',"w") as file:
    # 加載mat文件
    data = scipy.io.loadmat('data/brainNet/EMCI/kalmancorr/kalmancorr_0.01_0.6.mat')


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
    for i in range(90):
        for j in range(90):
            #
            if bb[i][j]!=0:
                bb[i][j] = (bb[i][j]-0.992)*1000
            if bb[i][j]<0 or bb[i][j]>1:
               bb[i][j]=0
            file.write(str(bb[i][j])+'\t')
        file.write('\n')
