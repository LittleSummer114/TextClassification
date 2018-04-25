TM
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

def SplitSentence(sent,stopwords,vocab):
    sent=clear_string(sent)
    words=jieba.cut(sent)
    sentence=[]
    for word in words:
        if(vocab ==[]):
            sentence.append(word)
        else:
            if(stopwords.get(word)is None and word in vocab):
                sentence.append(word)
    #string = " ".join(sentence)
    return sentence

def read_SST_1(data):
    stopwords = getStopWords()
    vocab = []
    if(data!={}):
        vocab = data['vocab_word']
    def read(name):
        path = 'data/AG/' + name + '.txt'
        data = open(path,'r',encoding='utf-8').readlines()
        x = []
        y = []
        Discuss = []
        for line in data:
            line = line.strip()
            line = line.split('\t')
            y.append(line[1])
            x.append(SplitSentence(line[0],stopwords,vocab))
            Discuss.append(line[0])
        return x,y,Discuss
    train_x, train_y, train_Discuss = read('train')
    dev_x, dev_y, dev_Discuss = read('dev')
    test_x, test_y, test_Discuss = read('test')
    data['train_x'] = train_x
    data['dev_x'] = dev_x
    data['train_Discuss'] = train_Discuss
    data['dev_Discuss'] = dev_Discuss
    data['train_y'] = train_y
    data['dev_y'] = dev_y
    data['test_x'] = test_x
    data['test_y'] = test_y
    data['test_Discuss'] = test_Discuss
    return data

def read_AG(data):
    stopwords = getStopWords()
    vocab = []
    if(data!={}):
        vocab = data['vocab_word']
    def read(name):
        path = 'data/AG/' + name + '.txt'
        data = open(path,'r',encoding='utf-8').readlines()
        x = []
        y = []
        Discuss = []
        for line in data:
            line = line.strip()
            line = line.split('\t')
            y.append(line[0])
            x.append(SplitSentence(line[1],stopwords,vocab))
            Discuss.append(line[1])
        return x,y,Discuss
    train_x, train_y, train_Discuss = read('train')
    train_x, train_y, train_Discuss = shuffle(train_x, train_y, train_Discuss)
    test_x, test_y, test_Discuss = read('test')
    train_x_index = len(train_x)
    dev_x_index = train_x_index // 10
    data['train_x'] = train_x[dev_x_index:train_x_index]
    data['dev_x'] = train_x[:dev_x_index]
    data['train_Discuss'] = train_Discuss[dev_x_index:train_x_index]
    data['dev_Discuss'] = train_Discuss[:dev_x_index]
    data['train_y'] = train_y[dev_x_index:train_x_index]
    data['dev_y'] = train_y[:dev_x_index]
    data['test_x'] = test_x
    data['test_y'] = test_y
    data['test_Discuss'] = test_Discuss
    return data

def read_HUAWEI(data):
    '''
    输入华为制造的数据集 格式是 entity1,entity2,relation,origin_data
    :return:
    '''
    stopwords = getStopWords()
    vocab = []
    if(data != {}):
        vocab = data['vocab_word']
    def read(name):
        x = []
        y = []
        Discuss = []
        path = 'data/HUAWEI/'+name+'.csv'
        data = pd.read_csv(path)
        for i in range(len(data['entity1'])):
            # sentence_x = []
            # sentence_x.append(data['entity1'][i])
            # sentence_x.append(data['entity2'][i])
            sentence = SplitSentence(data['origin_data'][i],stopwords,vocab)
            x.append(sentence)
            #x.append(sentence_x)
            Discuss.append(' '.join(sentence))
            y.append(data['relation'][i])
        return x,y,Discuss

    train_x, train_y, train_Discuss = read('train')
    train_x, train_y, train_Discuss = shuffle(train_x, train_y, train_Discuss)
    test_x, test_y, test_Discuss = read('test')
    train_x_index = len(train_x)
    dev_x_index = train_x_index // 10
    data['train_x'] = train_x[dev_x_index:train_x_index]
    data['dev_x'] = train_x[:dev_x_index]
    data['train_Discuss'] = train_Discuss[dev_x_index:train_x_index]
    data['dev_Discuss'] = train_Discuss[:dev_x_index]
    data['train_y'] = train_y[dev_x_index:train_x_index]
    data['dev_y'] = train_y[:dev_x_index]
    data['test_x'] = test_x
    data['test_y'] = test_y
    data['test_Discuss'] = test_Discuss

    return data

def read_TREC(data):
    vocab = []
    if(data!={}):
        vocab = data['vocab_word']
    stopwords = getStopWords()
    def read(name):
        filename='data/TREC/'+name+'.txt'
        data=open(filename,encoding='utf-8')
        x = []
        y = []
        Discuss = []
        for line in data:
            line = line.strip()
            line = line.split(' ')
            sentence = SplitSentence(' '.join(line[1:]),stopwords,vocab)
            y.append(line[0].split(':')[0])
            #print(sentence)
            x.append(sentence)
            Discuss.append(' '.join(sentence))
        return x,y,Discuss
    train_x,train_y,train_Discuss=read('train')
    train_x,train_y,train_Discuss = shuffle(train_x,train_y,train_Discuss)
    test_x,test_y,test_Discuss=read('test')
    train_x_index=len(train_x)
    dev_x_index=train_x_index//10
    data['train_x'] = train_x[dev_x_index:train_x_index]
    data['dev_x']=train_x[:dev_x_index]
    data['train_Discuss'] = train_Discuss[dev_x_index:train_x_index]
    data['dev_Discuss'] = train_Discuss[:dev_x_index]
    data['train_y'] = train_y[dev_x_index:train_x_index]
    data['dev_y'] = train_y[:dev_x_index]
    data['test_x']=test_x
    data['test_y']=test_y
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

def read_Subj(data):
    vocab = []
    if (data != {}):
        vocab = data['vocab_word']
    stopwords = getStopWords()
    x, y ,Discuss = [], [], []

    with open('data/Subj/subjective.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            #相当与原来这里是做NER得到的phrase,那么就需要找到word2vec里面的phrase
            sentence = SplitSentence(line, stopwords, vocab)
            x.append(sentence)
            y.append(1)
            Discuss.append(' '.join(sentence))

    with open('data/Subj/objective.txt', 'r', encoding='utf-8') as f:
        for line in f:
            if line[-1] == '\n':
                line = line[:-1]
            sentence = SplitSentence(line, stopwords, vocab)
            x.append(sentence)
            y.append(0)
            Discuss.append(' '.join(sentence))

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data['train_x'], data['train_y'] , data ['train_Discuss'] = x[:test_idx], y[:test_idx], Discuss[:test_idx]
    data['dev_x'], data['dev_y'], data ['dev_Discuss']  = x[dev_idx:test_idx], y[dev_idx:test_idx],Discuss[dev_idx:test_idx]
    data['test_x'], data['test_y'], data ['test_Discuss']  = x[test_idx:], y[test_idx:],Discuss[test_idx:]

    return data

def read_MRS(data):
    vocab = []
    if(data!={}):
        vocab = data['vocab_word']
    stopwords = getStopWords()
    def read(name):
        filename='data/MRS/'+name+'.txt'
        data=open(filename,encoding='utf-8').readlines()
        x=[]
        y=[]
        Discuss = []
        for line in data:
            line=line.strip().split('\t')
            sentence = SplitSentence(' '.join(line[1].split()), stopwords, vocab)
            y.append(line[0])
            x.append(sentence)
            Discuss.append(line[1])
        return x,y,Discuss
    train_x, train_y, train_Discuss = read('train')
    train_x, train_y, train_Discuss =shuffle(train_x,train_y,train_Discuss)
    test_x, test_y, test_Discuss =read('test')
    train_x_index=len(train_x)
    dev_x_index=train_x_index//10
    data['train_x'] = train_x[dev_x_index:train_x_index]
    data['dev_x']=train_x[:dev_x_index]
    data['train_Discuss'] = train_Discuss[dev_x_index:train_x_index]
    data['dev_Discuss'] = train_Discuss[:dev_x_index]
    data['train_y'] = train_y[dev_x_index:train_x_index]
    data['dev_y'] = train_y[:dev_x_index]
    data['test_x']=test_x
    data['test_y']=test_y
    data['test_Discuss'] = test_Discuss
    return data

def read_MR(data):
    vocab = []
    if(data!={}):
        vocab = data['vocab_word']
    stopwords = getStopWords()
    def read(name):
        filename='data/MR/'+name+'.txt'
        data=open(filename,encoding='utf-8').readlines()
        x=[]
        y=[]
        Discuss = []
        for line in data:
            line=line.strip().split('\t')
            sentence = SplitSentence(' '.join(line[1].split()), stopwords, vocab)
            y.append(line[0])
            x.append(sentence)
            Discuss.append(line[1])
        return x,y,Discuss
    train_x, train_y, train_Discuss = read('train')
    train_x, train_y, train_Discuss =shuffle(train_x,train_y,train_Discuss)
    test_x, test_y, test_Discuss =read('test')
    train_x_index=len(train_x)
    dev_x_index=train_x_index//10
    data['train_x'] = train_x[dev_x_index:train_x_index]
    data['dev_x']=train_x[:dev_x_index]
    data['train_Discuss'] = train_Discuss[dev_x_index:train_x_index]
    data['dev_Discuss'] = train_Discuss[:dev_x_index]
    data['train_y'] = train_y[dev_x_index:train_x_index]
    data['dev_y'] = train_y[:dev_x_index]
    data['test_x']=test_x
    data['test_y']=test_y
    data['test_Discuss'] = test_Discuss
    return data

def list2dic(temp_list):
    temp_dic={}
    for key in temp_list:
        temp_dic[key]=1
    return temp_dic

def read_Travel(data):
    stopwords=getStopWords()
    if(data == {}):
        vocab = []
    else:
        vocab = list2dic(data['vocab_word'])
    def read(name):
        x=[]
        y=[]
        Discuss=[]
        filename = 'data/Travel/' + name + '.csv'
        Travel_data = read_csv(filename, encoding='utf-8')
        if(name=='train'):
            Travel_data = Travel_data.drop_duplicates(['Discuss'])
        result = DataFrame()
        for idx,content in Travel_data.iterrows():
            string=SplitSentence(content['Discuss'],stopwords,vocab)
            Discuss.append(string)
            x.append(string.split())
            if (name == 'test'):
                y.append('NULL')
            else:
                y.append(content['Score'])
        result['Id'] = Travel_data['Id']
        result['Discuss'] = Discuss
        result['Score'] = y
        return x, y,result['Id'],result['Discuss']
    time1 = time.time()
    train_x,train_y ,train_id,train_Discuss=read('train')
    test_x,test_y,test_id,test_Discuss=read('test')
    train_x_index = len(train_x)
    dev_x_index = train_x_index // 10
    train_x,train_y = train_x,train_y
    train_x,train_y = shuffle(train_x,train_y)
    data['train_x'] = train_x[dev_x_index:train_x_index]
    data['dev_x'] = train_x[:dev_x_index]
    data['train_y'] = train_y[dev_x_index:train_x_index]
    data['dev_y'] = train_y[:dev_x_index]
    data['test_x'] = test_x
    data['test_y'] = test_y
    data['train_Id'], data['train_Discuss'] = train_id[dev_x_index:train_x_index], train_Discuss[dev_x_index:train_x_index]
    data['dev_Id'], data['dev_Discuss'] = train_id[:dev_x_index], train_Discuss[:dev_x_index]
    data['test_Id'], data['test_Discuss'] = test_id, test_Discuss
    time2 = time.time()
    print('Load Dataset Time:', str(time2 - time1))
    return data

def dataAugument():
    '''
    数据增强,将类别比较少的数据两两组合
    :return:
    '''
def read_TravelTest():
    data={}
    stopwords=getStopWords()
    def read(name):
        x=[]
        y=[]
        Discuss=[]
        filename = 'data/TravelTest/' + name + '.csv'
        resultname = 'data/TravelTest/' + name + '_split.csv'
        Travel_data = read_csv(filename,encoding='utf-8')
        result = DataFrame()
        for i in range(len(Travel_data['Id'])):
            string=SplitSentence(Travel_data['Discuss'][i],stopwords)
            Discuss.append(string)
            x.append(string.split())
            if(name=='test'):
                y.append('NULL')
            else:
                y.append(Travel_data['Score'][i])
        result['Id']=Travel_data['Id']
        result['Discuss']=Travel_data['Discuss']
        result['Score']=y
        result.to_csv(resultname,encoding='utf-8',index=False)
        return x, y,Travel_data['Id'],Travel_data['Discuss']
    train_x,train_y,train_id,train_Discuss=read('train')
    test_x,test_y,test_id,test_Discuss=read('test')
    train_x_index = len(train_x)
    dev_x_index = train_x_index // 10
    train_x, train_y = train_x, train_y
    train_x, train_y = shuffle(train_x, train_y)
    data['train_x'] = train_x[dev_x_index:train_x_index]
    data['dev_x'] = train_x[:dev_x_index]
    data['train_y'] = train_y[dev_x_index:train_x_index]
    data['dev_y'] = train_y[:dev_x_index]
    data['test_x'] = test_x
    data['test_y'] = test_y
    data['train_Id'], data['train_Discuss'] = train_id[dev_x_index:train_x_index], train_Discuss[
                                                                                   dev_x_index:train_x_index]
    data['dev_Id'], data['dev_Discuss'] = train_id[:dev_x_index], train_Discuss[:dev_x_index]
    data['test_Id'], data['test_Discuss'] = test_id, test_Discuss
    return data

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

def getVocab(params):
    vocab_word_file = 'data/' + params['dataset'] + '/vocab_word.csv'
    vocab_char_file = 'data/' + params['dataset'] + '/vocab_char.csv'
    if (os.path.exists(vocab_word_file)==False and os.path.exists(vocab_char_file)==False):
        data = {}
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

def getHit(train,test,params,name):
    sent_word_file = 'data/' + params['dataset'] + '/' + 'sent_'+name+'_dis.eps'
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

def element2idx(data,name):
    data['idx_to_'+name] = {}
    data[name+'_to_idx'] = {}
    print(data.keys())
    for key, word in enumerate(data['vocab_'+name]):
        data['idx_to_'+name][key] = word
        data[name+'_to_idx'][word] = key
    return data

def getDataset(params):
    data={}
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
        print(label,data['y'].count(label),data['train_y'].count(label),data['dev_y'].count(label))
    element2idx(data, 'word')
    element2idx(data, 'char')
    params = load_wc(data, params)
    params = load_concept(data, params)
    data=Sen2Index(data,params)
    return data,params

def load_concept(data,params):
    concepts = json.load(open('word_concept_vector.json','r'))
    concept_matrix = {}
    for word in data['vocab_word']:
        if(concepts.get(word)is not None):
            concept_matrix[word] = concepts[word]
    params['concept_vectors'] = concept_matrix
    return params

def load_wc(data,params):
    if(params['type']=='rand'):
        params['wv_maritx'] = []
        print('rand')
        return params
    path = 'models/' + params['dataset'] + '_'+params['level']+'_'+params['wv']+'_'+str(params['max_features'])+'.pkl'
    if(os.path.exists(path)):
        wc_matrix=pickle.load(open(path,'rb'))
    else:
        wc_matrix = []
        if(params['wv']=='word2vec'):
            #word2vec = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin',binary=True)
            word2vec = json.load(open('word_vectors.json', 'r'))
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

def save_models(model,params):
    path='models/{}_{}_{}_{}.pkl'.format(params['dataset'],params['model'],params['level'],params['time'])
    pickle.dump(model,open(path,'wb'))
    print('successful saved models !')

def getRMSE(prediction,true):
    rmse=0
    assert (len(prediction)==len(true))
    for i in range(len(prediction)):
        rmse+=math.pow(prediction[i]-true[i],2)
    rmse=math.sqrt(rmse/len(prediction))
    rmse=rmse/(1+rmse)
    return rmse

def getACC(prediction,true):
    acc=0
    #print('prediction',prediction)
    #print('true',true)
    for i in range(len(prediction)):
        if(prediction[i]==true[i]):
            acc+=1
    acc=acc/len(prediction)
    return acc

def getEntityRelation():
    '''
    读取excel中的数据,得到每个缺陷还有部件的层次关系
    :return:
    '''
    data = pd.read_excel('relation_hierarchy.xlsx')
    defect_entities = {}
    defect = ''
    for i in range(len(data['缺陷名称'])):
        if(type(data['缺陷名称'][i]) == str):
            defect = data['缺陷名称'][i]
            defect_entities[defect] = {}
        if(type(data['第一级'][i]) == str):
            if(defect_entities[defect].get(data['第一级'][i]) is None):
                defect_entities[defect][data['第一级'][i]] = ""
        if(type(data['第二级'][i]) == str):
            temp = defect_entities[defect][data['第一级'][i]]
            if(str(data['第二级'][i]) not in temp):
                temp +=' ' + str(data['第二级'][i])
                defect_entities[defect][data['第一级'][i]] = temp
        if(type(data['第三级'][i]) == str):
            if (defect_entities[defect].get(data['第二级'][i]) is None):
                defect_entities[defect][data['第二级'][i]] = ""
            temp = defect_entities[defect][data['第二级'][i]]
            if(str(data['第三级'][i] not in temp)):
                temp += ' ' + str(data['第三级'][i])
                defect_entities[defect][data['第二级'][i]] = temp
    print(defect_entities)
    json.dump(defect_entities,open('relation_hierarchy.json','w',encoding='utf-8'))

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
