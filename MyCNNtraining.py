import tensorflow
from keras.layers import Embedding, Input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from MyCNN  import *
import pandas as pd

# 特征单词数
max_words = 10000
# 这些单词都属于max_words中的单词，输入维度
maxlen = 20
# 输出维度
embedding_dim = 64

def tokenized(dataframe):
    ## 二、填充缺失值
    train_X = dataframe.fillna("_na_").values

    ## 三、对句子进行分词
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)

    ## 四、填充
    train_X = pad_sequences(train_X, maxlen=maxlen)
    return train_X

def main():
    raw_train = pd.read_csv("materials/train.csv")
    train_s1 = raw_train.iloc[:, 1]
    train_s2 = raw_train.iloc[:, 2]
    train_label = pd.DataFrame(raw_train.iloc[:, -1])
    raw_label = pd.get_dummies(train_label['same_security'])
    train_label = raw_label[True]
    raw_test = pd.read_csv("materials/test.csv")
    test_s1 = raw_test.iloc[:, 1]
    test_s2 = raw_test.iloc[:, 2]
    #句子初始化
    tokenized_train_s1 = tokenized(train_s1)
    tokenized_train_s2 = tokenized(train_s2)
    #句子初始化
    tokenized_test_s1 = tokenized(test_s1)
    tokenized_test_s2 = tokenized(test_s2)

    # #第一层word bedding
    # 定义输入层，确定输入维度
    input1 = Input(shape=(maxlen,))
    input2 = Input(shape=(maxlen,))
    e1 = Embedding(max_words, embedding_dim, input_length=maxlen)(input1)
    e2 = Embedding(max_words, embedding_dim, input_length=maxlen)(input2)
    # #第二层
    c1 = keras.layers.Conv1D(64, 3, strides=1, padding='valid',
                             activation='relu')(e1)
    c2 = keras.layers.Conv1D(64, 3, strides=1, padding='valid',
                             activation='relu')(e2)

    # #第三层
    m1 = keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid', data_format=None)(c1)
    m2 = keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid', data_format=None)(c2)
    #两个结果聚合
    m = keras.layers.concatenate([m1, m2])
    # #第四曾
    h1 = keras.layers.Flatten()(m)
    # #第五层输出层
    output = keras.layers.Dense(1, activation='softmax')(h1)
    model = Model(inputs=[input1, input2], outputs=output)

    model.compile(optimizer="Adam", loss="BinaryCrossentropy", metrics=["BinaryCrossentropy"])
    model.fit([tokenized_train_s1, tokenized_train_s2], train_label, epochs=3)
    # test_score = model.evaluate([tokenized_test_s1, tokenized_test_s2], test_label)
    prection_result = model.predict([tokenized_test_s1 , tokenized_test_s2])
    print(prection_result)

if __name__ == '__main__':
    main()