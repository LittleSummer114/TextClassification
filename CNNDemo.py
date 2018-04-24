#encoding:utf-8
# to process data x:['I','Like','eating','apples'] --> y[0]
#输入是一个数据集的名字
#输出是 数据集,训练集 测试集 验证集 还有词汇量 word_index index_word
import numpy as np
np.random.seed(7)
import pickle
import math
import gensim
import pymysql
import json
import pandas
from pandas import DataFrame,read_csv
import jieba
import os
import time
import pandas as pd
import re
from sklearn.utils import shuffle
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
import matplotlib.pyplot as plt

# for neural-network based method
def Sen2Index(data,params):
    def char2idx(name):
        x = []
        y = []
        for sent in data[name+'_x']:
            sentence_char=[]
            len_chars = 0
            for word in sent:
                for char in word:
                    if(len(sentence_char) < params ['max_sent_char_length']):
                        len_chars += 1
                        sentence_char.append(data['char_to_idx'][char])
            sentence_char += [len(data['vocab_char'])+1]*(params['max_sent_char_length']-(len_chars))
            if(params['length_feature']==1):
                sentence_char.append(len_chars)
            x.append(sentence_char)
        for label in data[name+'_y']:
            if(label in data['classes']):
                y.append(data['classes'].index(label))
        return x,y
    def word2idx(name):
        x = []
        y = []
        for sent in data[name+'_x']:
            sentence_word = []
            for word in sent:
                sentence_word.append(data['word_to_idx'][word])
                if (len(sentence_word) == params['max_sent_word_length']):
                    break
            sentence_word += [(len(data['vocab_word']) + 1)] * (params['max_sent_word_length'] - len(sent))
            if (params['length_feature'] == 1):
                sentence_word.append(len(sent))
            x.append(sentence_word)
        for c in data[name+'_y']:
            if(c in data['classes']):
                y.append(data['classes'].index(c))
        return x,y
    data['train_x_word'],data['train_y_word'] = word2idx('train')
    data['dev_x_word'], data['dev_y_word'] = word2idx('dev')
    data['test_x_word'], data['test_y_word'] = word2idx('test')
    data['x_word'] = data['train_x_word'] + data ['test_x_word'] + data ['dev_x_word']
    data['y_word'] = data['train_y_word'] + data ['test_y_word'] + data ['dev_y_word']
    data['train_x_char'], data['train_y_char'] = char2idx('train')
    data['dev_x_char'], data['dev_y_char'] = char2idx('dev')
    data['test_x_char'], data['test_y_char'] = char2idx('test')
    data['x_char'] = data['train_x_char'] + data['test_x_char'] + data['dev_x_char']
    data['y_char'] = data['train_y_char'] + data['test_y_char'] + data['dev_y_char']
    return data

def clear_string(sent):
    sent = sent.replace('< br / > n', ' ')
    sent = sent.replace('<br />rn','')
    sent = sent.replace('& quot;', '')
    sent = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", sent)
    sent = re.sub(r"\'s", " \'s", sent)
    sent = re.sub(r"\'ve", " \'ve", sent)
    sent = re.sub(r"n\'t", " n\'t", sent)
    sent = re.sub(r"\'re", " \'re", sent)
    sent = re.sub(r"\'d", " \'d", sent)
    sent = re.sub(r"\'ll", " \'ll", sent)
    sent = re.sub(r",", " , ", sent)
    sent = re.sub(r"!", " ! ", sent)
    sent = re.sub(r"\(", " \( ", sent)
    sent = re.sub(r"\)", " \) ", sent)
    sent = re.sub(r"\?", " \? ", sent)
    sent = re.sub(r"\s{2,}", " ", sent)
    return sent

def getStopWords():
    stopwords={}
    data=open('data/CH_stopWords.txt',encoding='utf-8').readlines()
    for line in data:
        line=line.strip()
        stopwords[line]=1
    return stopwords

def SplitSentence(words,stopwords,vocab):
    sentence=[]
    for word in words:
        if(vocab ==[]):
            sentence.append(word)
        else:
            if(stopwords.get(word)is None and word in vocab):
                sentence.append(word)
    #string = " ".join(sentence)
    return sentence

def read_EMNLP(data):
    stopwords = getStopWords()
    filename = 'data/EMNLP/hanyu_output_cws_id.txt'
    vocab = []
    if (data != {}):
        vocab = data['vocab_word']
    filedata = open(filename, encoding='utf-8')
    dataset = {}
    for line in filedata:
        line = line.strip()
        line = line.split('|')
        dataset [line[0]] = line[1]
    train_x, train_y, train_Discuss = [], [] , []
    test_x, test_y, test_Discuss = [],[],[]
    for i in range(10):
        filename = 'data/EMNLP/folds/' + 'fold' + str(i) + '.txt'
        labeldata = open(filename, encoding='utf-8')
        for line in labeldata:
            line = line.strip()
            line = line.split('|')
            if(i == data['testfold']):
                test_y.append(line[0])
                sentence = SplitSentence(dataset[line[1]].split(), stopwords, vocab)
                test_Discuss.append(' '.join(sentence))
                test_x.append(sentence)
            else:
                train_y.append(line[0])
                sentence = SplitSentence(dataset[line[1]].split(), stopwords, vocab)
                train_Discuss.append(' '.join(sentence))
                train_x.append(sentence)
    data['train_x'] = train_x
    data['dev_x'] = []
    data['train_Discuss'] = train_Discuss
    data['dev_Discuss'] = []
    data['train_y'] = train_y
    data['dev_y'] = []
    data['test_x'] = test_x
    data['test_y'] = test_y
    data['test_Discuss'] = test_Discuss
    return data

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

#统计词频
def getDFrequency(data):
    print('Getting')
    y = []
    set_data = list(set(data))
    for word in set_data:
        y.append(data.count(word))
    print('Getting')
    return set_data,y

def tocsv(data,vocab_word_file,name):
    # data: 字典的形式,存放的是词语:词语的统计信息(词频,tf*idf)
    temp = sorted(data.items(),key=lambda i:i[1],reverse=True)
    temp_vocab_word = DataFrame(temp,columns=['vocab_'+name,'df'])
    temp_vocab_word.to_csv(vocab_word_file, encoding='utf-8', index=False)
    return temp

#获取词典数据
def getVocab(params):
    vocab_word_file = 'data/' + params['dataset'] + '/testfold' + str(params['testfold']) + '_vocab_word.csv'
    vocab_char_file = 'data/' + params['dataset'] + '/testfold' + str(params['testfold']) + '_vocab_char.csv'
    if (os.path.exists(vocab_word_file)==False and os.path.exists(vocab_char_file)==False):
        data = {}
        data['testfold'] = params['testfold']
        data = eval('read_{}'.format(params['dataset']))(data)
        if(params['cv']==False):
            data['x'] = data['train_x'] + data['dev_x'] + data['test_x']
            data['y'] = data['train_y'] + data['test_y'] + data['dev_y']
        #统计词语的统计信息,如词频,TF*IDF,最后都以字典的形式存在
        vocab_word = {}
        vocab_char = {}
        for sent in data['x']:
            for word in sent:
                vocab_word [word] = vocab_word.get(word,0)+1
                for char in word:
                    vocab_char[char] =vocab_char.get(char,0) + 1
        print('Getting Vocab')
        tocsv(vocab_word,vocab_word_file,'word')
        tocsv(vocab_char,vocab_char_file,'char')
    else:
        print(vocab_word_file)
        print('Vocab_Path exists')
    vocab_word = read_csv(vocab_word_file, encoding='utf-8')['vocab_word'].tolist()
    vocab_char = read_csv(vocab_char_file, encoding='utf-8')['vocab_char'].tolist()
    #print('word',vocab_word)
    #print('char',vocab_char)
    return vocab_word, vocab_char

#统计句子长度数据,主要是RNN做packed用
def getMaxLength(sentences):
    len_word=[]
    len_char=[]
    for sent in sentences:
        len_word.append(len(sent))
        temp_len_char=0
        for word in sent:
            temp_len_char+=len(word)
        len_char.append(temp_len_char)
    return len_word,len_char

#以图表的形式输出数据分析结果
def getHit(train,test,params,name):
    sent_word_file = 'data/' + params['dataset'] + '/'+ '/testfold' + str(params['testfold'])+ '_sent_'+name+'_dis.eps'
    plt.clf()
    plt.figure(1)
    plt.title('sentence_length_'+name)
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.hist(train,bins=100, label='train', lw=1)
    plt.hist(test, bins=100,label='test', lw=1)
    plt.legend(loc='upper right')
    plt.figure(1).savefig(sent_word_file)
    plt.close()

#输出分析数据,目前仅有句子长度分布
def DataAnalysis(data,params):
    train_word, train_char = getMaxLength(data['train_x'] + data['dev_x'])
    test_word, test_char = getMaxLength(data['test_x'])
    getHit(train_word, test_word, params, 'word')
    getHit(train_char, test_char, params, 'char')
    def getMax(name,all):
        print('max_sent_'+name+'_length', max(all))
        print('average_sent_'+name+'_length', np.average(np.array(all)))
    getMax('word',train_word+test_word)
    getMax('char',train_char+test_char)

#以{idx:word}输出数据
def element2idx(data,name):
    data['idx_to_'+name] = {}
    data[name+'_to_idx'] = {}
    print(data.keys())
    for key, word in enumerate(data['vocab_'+name]):
        data['idx_to_'+name][key] = word
        data[name+'_to_idx'][word] = key
    return data

#加载处理好的数据集
def getDataset(params):
    data={}
    data['testfold'] = params['testfold']
    vocab_word, vocab_char = getVocab(params)
    data['vocab_word'] = vocab_word[:params['max_features']]
    data['vocab_char'] = vocab_char
    idx_to_word = {}
    for index, word in enumerate(data['vocab_word']):
        idx_to_word[index] = word
    print(data.keys())
    params['idx_to_word'] = idx_to_word
    data = eval('read_{}'.format(params['dataset']))(data)
    data['x'] = data['train_x'] + data['dev_x'] + data['test_x']
    data['y'] = data['train_y'] + data['test_y'] + data['dev_y']
    data['Discuss']=data['train_Discuss']+data['test_Discuss']+data['dev_Discuss']
    print(data.keys())
    DataAnalysis(data, params)
    classes=list(set(data['y']))
    if('NULL' in classes):
        classes.remove('NULL')
    data['classes']=classes
    print('classes',data['classes'])
    print('label distribution')
    print('label, dataset, train, dev')
    for label in data['classes']:
        print(label,data['y'].count(label),data['train_y'].count(label),data['test_y'].count(label))
    element2idx(data, 'word')
    element2idx(data, 'char')
    params = load_wc(data, params)
    data=Sen2Index(data,params)
    return data,params

#此处加载预训练好的词向量模型,修改相应路径即可 word2vec = json.load(open('word_vectors.json', 'r'))
def load_wc(data,params):
    if(params['type']=='rand'):
        params['wv_maritx'] = []
        print('rand')
        return params
    path = 'models/' + params['dataset'] + '_'+params['level']+'_'+'testfold'+str(params['testfold'])+'_'+str(params['max_features'])+'.pkl'
    if(os.path.exists(path)):
        wc_matrix=pickle.load(open(path,'rb'))
    else:
        wc_matrix = []
        if(params['wv']=='word2vec'):
            word2vec = KeyedVectors.load_word2vec_format('data/EMNLP/zh_wiki_w2v_skigram_300_win5_mincount500_negtive5_iter3.txt',binary=False)
            print('word2vec model saving successfully!')
        for word in data['vocab_word']:
            if (word in word2vec):
                wc_matrix.append(np.array(word2vec[word]).astype('float32'))
            else:
                wc_matrix.append(np.random.uniform(-0.25, 0.25, params['dimension']).astype('float32'))
        # for unk and zero-padding
        wc_matrix.append(np.random.uniform(-0.25, 0.25, params['dimension']).astype('float32'))
        wc_matrix.append(np.zeros(params['dimension']).astype('float32'))
        #wc_matrix.append(np.random.uniform(-0.25, 0.25, 300).astype('float32'))
        print('len(word_matrix):',len(wc_matrix))
        wc_matrix=np.array(wc_matrix)
        pickle.dump(wc_matrix,open(path,'wb'))
    params['wv_maritx']=wc_matrix
    return params

#此处保存训练好的模型
def save_models(model,params):
    path='models/{}_{}_{}_{}.pkl'.format(params['dataset'],params['model'],params['level'],params['time'])
    pickle.dump(model,open(path,'wb'))
    print('successful saved models !')

#输出实验结果,评价指标为RMSE
def getRMSE(prediction,true):
    rmse=0
    assert (len(prediction)==len(true))
    for i in range(len(prediction)):
        rmse+=math.pow(prediction[i]-true[i],2)
    rmse=math.sqrt(rmse/len(prediction))
    rmse=rmse/(1+rmse)
    return rmse

#输出实验结果,评价指标为ACC
def getACC(prediction,true):
    acc=0
    #print('prediction',prediction)
    #print('true',true)
    for i in range(len(prediction)):
        if(prediction[i]==true[i]):
            acc+=1
    acc=acc/len(prediction)
    return acc

#加载预训练好的模型
def load_models(params):
    path = 'models/{}_{}_{}_{}.pkl'.format(params['dataset'], params['model'], params['level'], params['time'])
    print('model path',path)
    if(os.path.exists(path)):
        try:
            model=pickle.load(open(path,'rb'))
            print('loaded model successfully!')
            return model
        except:
            print('error')
    else:
        print('no model finded!')
