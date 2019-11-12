## 保存模型

import os
import scipy.io
from sklearn import preprocessing
from keras import optimizers
from keras.layers import *
from keras.models import Sequential,Model
from keras_self_attention import SeqSelfAttention
from keras.callbacks import LearningRateScheduler
from math import pow, floor
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.compat.v1.set_random_seed(2)

import warnings
warnings.filterwarnings("ignore")

# 计算学习率
def step_decay(epoch):
    init_lrate = 10e-4
    drop = 0.95
    epochs_drop = 10
    lrate = init_lrate * pow(drop, floor(1 + epoch) / epochs_drop)
    return lrate


def one_hot(y_,n_classes):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS



if __name__ == '__main__':

    n_classes = 3
    hidden_size = 64
    droup_out = 0.2

    Featurefile = 'data/NC_EMCI_LMCI/features/' + 'kalmancorr_0.01_0.5_422.mat'
    print(os.path.join(Featurefile))

    datas = scipy.io.loadmat(Featurefile)
    corr = datas['datas']
    sample_nums = len(corr)

    X = np.array([np.array(corr[i][0], dtype=np.float32) for i in range(sample_nums)])
    _, n_step, featureNum = X.shape

    # 均值-标准差归一化具体公式是(x - mean)/std。
    # 其含义是：对每一列的数据减去这一列的均值，然后除以这一列数据的标准差。最终得到的数据都在0附近，方差为1。

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = preprocessing.scale(X[i][j])


    labels = np.array([0 if corr[i][3] == 'NC  ' else 1 if corr[i][3] == 'EMCI' else 2 for i in range(sample_nums)], dtype=np.int32)  # 三分类，0 1

    model = Sequential()
    model.add(Bidirectional(GRU(hidden_size, input_shape=(n_step, featureNum), return_sequences=True), name='bigru'))
    model.add(SeqSelfAttention(
        attention_width=15,
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation=None,
        kernel_regularizer=regularizers.l2(1e-6),
        use_attention_bias=True,
        name='self_Attention'))
    model.add(Flatten())
    model.add(Dense(hidden_size))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=10e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                  metrics=['accuracy'])

    lrate = LearningRateScheduler(step_decay)
    model.fit(X, one_hot(labels,n_classes), epochs=10, verbose=2, validation_split=0.2, shuffle=True, initial_epoch=0, callbacks=[lrate])

    # 已有的model在load权重过后
    # 取某一层的输出为输出新建为model，采用函数模型
    bigru_layer_model = Model(inputs=model.input, outputs=model.get_layer('bigru').output)
    # 以这个model的预测值作为输出
    bigru_output = bigru_layer_model.predict(X)

    attention_layer_model = Model(inputs=model.input, outputs=model.get_layer('self_Attention').output)
    attention_output = attention_layer_model.predict(X)

    print(bigru_output.shape)
    print(bigru_output[9].shape)

    import seaborn as sns
    import matplotlib.pyplot as plt

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 15,
            }

    # NC
    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(X[9], cmap="Reds", square=True)
    ax.set_title('Original Input of dFC Networks', fontdict=font)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/nc.png')

    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(bigru_output[9][:, :64], cmap="Reds", square=True)
    ax.set_title('Forward Output of BiGRU', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/nc_forwardgru.png')

    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(bigru_output[9][:, 64:128], cmap="Reds", square=True)
    ax.set_title('Backward Output of BiGRU', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/nc_backattention.png')
    # plt.show()

    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(attention_output[9][:, :64],  cmap="Reds", square=True)
    ax.set_title('Forward Output of Self_Attention', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/nc_forwardattention.png')
    # plt.show()

    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(attention_output[9][:, 64:128], cmap="Reds", square=True)
    ax.set_title('Backward Output of Self_Attention', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/nc_backwardattention.png')
    # plt.show()

    # EMCI
    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(X[100], cmap="Reds", square=True)
    ax.set_title('Original Input of dFC Networks', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/emci.png')
    # plt.show()

    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(bigru_output[100][:, :64], cmap="Reds", square=True)
    ax.set_title('Forward Output of BiGRU', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/emci_forwardgru.png')
    # plt.show()

    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(bigru_output[100][:, 64:128], cmap="Reds", square=True)
    ax.set_title('Backward Output of BiGRU', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/emci_backattention.png')
    # plt.show()

    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(attention_output[100][:, :64], cmap="Reds", square=True)
    ax.set_title('Forward Output of Self_Attention', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/emci_forwardattention.png')
    # plt.show()

    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(attention_output[100][:, 64:128],  cmap="Reds", square=True)
    ax.set_title('Backward Output of Self_Attention', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/emci_backwardattention.png')
    # plt.show()


    # LMCI
    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(X[200], cmap="Reds", square=True)
    ax.set_title('Original Input of dFC Networks', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/lmci.png')
    # plt.show()

    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(bigru_output[200][:, :64],  cmap="Reds", square=True)
    ax.set_title('Forward Output of BiGRU', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/lmci_forwardgru.png')
    # plt.show()

    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(bigru_output[200][:, 64:128], cmap="Reds", square=True)
    ax.set_title('Backward Output of BiGRU', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/lmci_backattention.png')
    # plt.show()

    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(attention_output[200][:, :64], cmap="Reds", square=True)
    ax.set_title('Forward Output of Self_Attention', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/lmci_forwardattention.png')
    # plt.show()

    fig, ax = plt.subplots(figsize=(9, 9))  # 设置画面大小
    sns.heatmap(attention_output[200][:, 64:128], cmap="Reds", square=True)
    ax.set_title('Backward Output of Self_Attention', fontsize=18)
    ax.set_ylabel('Time Points', fontsize=18)
    ax.set_xlabel('Features', fontsize=18)
    plt.savefig('images/heatmap/lmci_backwardattention.png')
    # plt.show()



