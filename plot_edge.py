import scipy.io

# label_name = 'nc'
# label_name = 'emci'
label_name = 'lmci'

matdata = scipy.io.loadmat('data/brainNet/LMCI/groupLasso/'+label_name+'.mat')
data = matdata[label_name]

with open('data/brainNet/Edge_AAL90_Binary_'+label_name+'.edge',"w") as file:
    for i in range(90):
        for j in range(90):
            if data[i][j]<0.01:
                file.write('0\t')
            else:
                file.write('1\t')
        file.write('\n')
