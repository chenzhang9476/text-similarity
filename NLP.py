from gensim import corpora, models, similarities
from gensim.corpora import Dictionary
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from numpy import *
import jieba
import MyCNN
import ASP

max_feature = 80
def preprocess(URL):
  # preprocess the raw data
  raw_df = pd.read_json(URL)
  raw_data = raw_df.iloc[:, -1]
  questions_list = []
  answers_list = []
  for row in raw_data:
    articles = pd.DataFrame(row)
    paragraphs = articles.iloc[:, 1]
    for par_rpw in paragraphs:
      qass = pd.DataFrame(par_rpw)
      qas = qass.iloc[:, 0]
      for qas_row in qas:
        questions = str(qas_row.get("question"))
        questions_list.append(questions)
        list_answers = qas_row.get("answers")
        if(len(list_answers) != 0):
          answers_list.append(list_answers[0].get('text'))
        else:
          answers_list.append("")
  return pd.DataFrame({'questions': questions_list, 'answers': answers_list})

def dataset(URL):
  data = preprocess(URL)
  questions = data.iloc[:, 0]
  answers = data.iloc[:, 1]
  tokenized_x = tokenized(questions.values.tolist())
  tokenized_y = tokenized(answers.values.tolist())
  return tokenized_x, tokenized_y, questions, answers

def tokenized(data):
  tknizer = Tokenizer()
  tknizer.fit_on_texts(data)
  #生成字典
  word_index = tknizer.word_index
  sentence = tknizer.texts_to_sequences(data)
  sentence = pad_sequences(sentence, value=0, padding='post', maxlen=20)
  return word_index, sentence


def method_1(test_x,test_y):
    query = ASP.inputing()
    texts = [jieba.lcut(text) for text in test_x]
    # 生成字典
    dictionary = Dictionary(texts)
    # 基于字典，将分词列表转换成稀疏向量集，语料库
    corpus = [dictionary.doc2bow(text) for text in texts]
    # print(corpus)
    # 训练tf-idf模型，传入语料库进行训练
    tfidf = models.TfidfModel(corpus)
    # 用训练好的TF-IDF模型处理被检索的文件，语料库
    corpus_tfidf = tfidf[corpus]
    # 句子相似度阈值
    sup = 0.3
    # row_index默认为-1，即未匹配到满足相似度阈值的问题
    row_index = -1
    # 得到tfidf值
    for temp in corpus_tfidf:
        # print(temp)
        vec_bow = dictionary.doc2bow(query.split())
        vec_tfidf = tfidf[vec_bow]
        index = similarities.MatrixSimilarity(corpus_tfidf)
        sims = index[vec_tfidf]
        max_loc = np.argmax(sims)  # 返回最大值的索引
        max_sim = sims[max_loc]
        if max_sim > sup:
            # 相似度最大值对应文件中问题所在的行索引
            row_index = max_loc + 1
    print(test_y[row_index - 1])


def main():
    print("Please waiting for initialization")
    tokenized_train_x, tokenized_train_y, raw_train_x, raw_train_y = dataset("materials/train-v2.0.json")
    #tokenized_test_x, tokenized_test_y, raw_test_x, raw_test_y = dataset("materials/dev-v2.0.json")
    #word_index, tokenized_words = tokenizing(raw_train_x, raw_train_y, raw_test_x, raw_test_y)
    test_x = raw_train_x.values.tolist()[0:100]
    test_y = raw_train_y.values.tolist()[0:100]
    print("Initialization complete")
    method_1(test_x,test_y)
    query = ASP.inputing()
    index = -1
    # for row in test_x:
    #     result = MyCNN(query, row)
    #     print(result)
    #print(row_index)

# where is Normandy located?
if __name__ == '__main__':
    main()