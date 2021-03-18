import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Conv1D
from keras.layers.pooling import GlobalMaxPooling1D
import glob
import os
from random import shuffle
from nltk.tokenize import TreebankWordTokenizer
from gensim.models.keyedvectors import KeyedVectors
from nlpia.loaders import get_data
np.random.seed(1234)
#1.数据预处理
filepath = 'D:/aclImdb_v1/aclImdb/train'
def pre_process_data(filepath):
    positive_path = os .path.join(filepath,'pos')
    negative_path = os .path.join(filepath,'neg')
    pos_label = 1
    neg_label = 0
    dataset = []

    for filename in glob.glob(os.path.join(positive_path,'*.txt')):
        with open(filename,'r',encoding='UTF-8') as f:
            dataset.append((pos_label,f.read()))
    for filename in glob.glob(os.path.join(negative_path,'*.txt')):
        with open(filename,'r',encoding='UTF-8') as f:
            dataset.append((neg_label,f.read()))
    shuffle(dataset)
    return dataset

dataset = pre_process_data(filepath)

# # print(type(dataset))
print(len(dataset))
# print(dataset[0:5])
# print(dataset[1])
# [(0,xxxx),(1,xxxxxx),(0,xxxxx)]

#2.向量化 分词
# word_vectors = get_data('w2v',limit = 200000)
word_vectors = KeyedVectors.load_word2vec_format('D:/ProgramData/Anaconda3/Lib/site-packages/nlpia/bigdata/GoogleNews-vectors-negative300.bin/GoogleNews-vectors-negative300.bin',binary=True,limit=200000)
print(len(word_vectors.vocab))


def tokennize_and_vectorize(dataset):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass
        vectorized_data.append(sample_vecs)
    return vectorized_data
#生成数据对应标签
def collect_expected(dataset):
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected

vectorized_data = tokennize_and_vectorize(dataset)#向量化数据
expected = collect_expected(dataset)#对应标签
#训练集测试集划分
split_point = int(len(vectorized_data)*.8)
x_train = vectorized_data[:split_point]
y_train = expected[:split_point]
x_test = vectorized_data[split_point:]
y_test = expected[split_point:]
print('测试集训练集划分完成')


# 参数设置
maxlen = 400
batchsize = 32
embedding_dims = 300
filters = 250
kernel_size = 3 #卷积核大小：embedding_dims * kernel_size  300*3
hidden_dims = 250
epochs = 2

#截断器 填充器

def pad_trunc(data,maxlen):
    new_data = []
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)
    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data
x_train = pad_trunc(x_train,maxlen)
x_test = pad_trunc(x_test,maxlen)

x_train = np.reshape(x_train,(len(x_train),maxlen,embedding_dims))#训练数据量*400*300
y_train = np.array(y_train)
x_test = np.reshape(x_test,(len(x_test),maxlen,embedding_dims))
y_test = np.array(y_test)
print("截断填充完成")
#-------------------------------

#构建模型
print('build model......')
model = Sequential()
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 input_shape=(maxlen,embedding_dims)
                 ))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#训练
model.fit(x_train,y_train,
          batch_size=batchsize,
          epochs=epochs,
          validation_data=(x_test ,y_test))

#模型保存
model_structure = model.to_json()
with open("cnn_model.json","w") as json_file:
    json_file.write(model_structure)
model.save_weights("cnn_weights.h5")




















