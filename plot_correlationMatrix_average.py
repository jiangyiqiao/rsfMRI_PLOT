import scipy.io
import numpy as np





mat_filepath = 'data/NC_EMCI_LMCI/features/kalmancorr_0.01_0.5_422.mat'

# 加載mat文件
data = scipy.io.loadmat(mat_filepath)

roi_data = data['datas']
corr_data = np.array([np.array(roi_data[i][0], dtype=np.float32) for i in range(len(roi_data))])  # num*130*features
sample_num, frame, features = corr_data.shape
aa = np.zeros(frame*features).reshape(frame,features)

print(corr_data.shape)
aa
for k in range(167,238):
    # print(corr_data[k])
    print(k)
    aa = [list(map(lambda x, y: x + y, x, y)) for (x, y) in zip(aa, corr_data[k])]
    # aa = np.sum(aa,corr_data[k].reshape(frame,features))



scipy.io.savemat('data/ave_lmci.mat', {'average': aa})

