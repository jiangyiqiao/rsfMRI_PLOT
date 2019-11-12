## 保存模型

import os
import scipy.io
from sklearn import preprocessing
from keras.layers import *
from keras.models import Sequential,Model
from keras_self_attention import SeqSelfAttention

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.compat.v1.set_random_seed(2)

import warnings
warnings.filterwarnings("ignore")



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

    X_reverse = X
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j] = X[i][X.shape[1]-j-1]
    print(X_reverse.shape)


    labels = np.array([0 if corr[i][3] == 'NC  ' else 1 if corr[i][3] == 'EMCI' else 2 for i in range(sample_nums)], dtype=np.int32)  # 三分类，0 1

    model_left = Sequential()
    model_left.add(GRU(hidden_size, input_shape=(n_step, featureNum), return_sequences=True, name='left_gru',))
    model_left.add(SeqSelfAttention(
        attention_width=15,
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation=None,
        kernel_regularizer=regularizers.l2(1e-6),
        use_attention_bias=False,
        name='left_Attention'))
    print(model_left)
    # model_left.add(Flatten())



    model_right = Sequential()
    model_right.add(GRU(hidden_size, input_shape=(n_step, featureNum), return_sequences=True, name='reverse_gru'))
    model_right.add(SeqSelfAttention(
        attention_width=15,
        attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
        attention_activation=None,
        kernel_regularizer=regularizers.l2(1e-6),
        use_attention_bias=False,
        name='right_Attention'))
    print(model_right)
    # model_right.add(Flatten())

    model = Sequential()
    model.add(Concatenate([model_left,model_right.reverse()]))
    model.add(Dense(hidden_size))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit([X, X_reverse], one_hot(labels,n_classes), epochs=1000, verbose=2, validation_split=0.2, shuffle=True, batch_size=20)

    # 已有的model在load权重过后
    # 取某一层的输出为输出新建为model，采用函数模型
    left_gru_layer_model = Model(inputs=model.input, outputs=model.get_layer('left_gru').output)
    # 以这个model的预测值作为输出
    left_gru_output = left_gru_layer_model.predict([X, X_reverse])


    print(left_gru_output.shape)
    print(left_gru_output[9])

