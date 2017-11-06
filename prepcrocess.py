#encoding：utf-8
#文本的格式是，id\ttitle\tcontent\tlabel
from import_file import *
#jieba.load_userdict()

def getStopWords(filename):
    stopwords={}
    data = open(filename, encoding='utf-8')
    for line in data:
        line=line.strip()
        stopwords[line]=stopwords.get(line,0)+1
    return stopwords

#按类别输出文档
def getSpeCateDoc(filename,categorylabel,resultname):
    data=open(filename,encoding='utf-8')
    resultfile= open(resultname, 'w',encoding='utf-8')
    for line in data:
        temp=line
        line=line.split('\t')
        label=line[0]
        if(label==categorylabel):
            resultfile.write(temp)

def getDoc(filename):
    labelDict={}
    #为了将label转换成为向量,保留索引
    labels_index=[]
    labels=[]
    docs=[]
    data=open(filename,encoding='utf-8')
    for line in data:
        line=line.strip()
        line=line.split('\t')
        content=line[1]
        label=line[0]
        docs.append(content)
        labelDict[label]=labelDict.get(label,0)+1
        if(label not in labels_index):
            labels_index.append(label)
        labels.append(labels_index.index(label))
    return docs,labels,labelDict

def word2vec(docs,filename):
    docVec=[]
    for doc in docs:
        words=[]
        wordlist=jieba.cut(doc)
        for word in wordlist and stopwords.get(word) is None:
            words.append([word])
        docVec.append(words)
    model=Word2Vec(docVec)
    model.save(filename)
    return model

def getDictFile(Dict,Dictfilename,flag):
    File=open(Dictfilename,'w',encoding='utf-8')
    Dict = sorted(Dict.items(), key=lambda word: word[1], reverse=flag)
    for word in Dict:
        File.write('%s\t%.4f\n' % (word[0],word[1]))

def readDictFile(Dictfilename):
    dict={}
    file = open(Dictfilename,encoding='utf-8')
    data=file.readlines()
    for line in data:
        line=line.strip()
        line=line.split('\t')
        if(len(line)==2):
            if('[' in line[1]):
                dict[line[0]]=list(line[1])
            else:
                dict[line[0]]=float(line[1])
    return dict

#得到传统的字典
def getDict_CH(docs,stopwords):
    #TVac:计算每个词在所有文档出现的次数,word在同一篇doc出现3次，算3次
    #ITVac:模拟IDF
    #DFVac:计算每个词在所有文档出现的次数,word在同一篇doc出现3次，算1次
    #IDFVac:IDF
    #word2vec
    TFVac={}
    DFvac={}
    IDFvac={}
    docVec=[]
    for doc in docs:
        words=[]
        wordlist=jieba.cut(doc)
        for word in wordlist:
            if(stopwords.get(word) is None):
                if (word not in words):
                    DFvac[word] = DFvac.get(word, 0) + 1
                words.append(word)
                TFVac[word] = TFVac.get(word, 0) + 1
        docVec.append(words)
    for word in TFVac:
        IDFvac[word]=IDFvac.get(word,0)+log(len(docVec)/(DFvac[word]+0.01))
    return docVec,TFVac,DFvac,IDFvac

#得到传统的字典
def getDict_EN(docs,stopwords):
    #TVac:计算每个词在所有文档出现的次数,word在同一篇doc出现3次，算3次
    #ITVac:模拟IDF
    #DFVac:计算每个词在所有文档出现的次数,word在同一篇doc出现3次，算1次
    #IDFVac:IDF
    #word2vec
    TFVac={}
    DFvac={}
    IDFvac={}
    docVec=[]
    for doc in docs:
        words=[]
        wordlist=doc.split(' ')
        for word in wordlist:
            if(stopwords.get(word) is None):
                if (word not in words):
                    DFvac[word] = DFvac.get(word, 0) + 1
                words.append(word)
                TFVac[word] = TFVac.get(word, 0) + 1
        docVec.append(words)
    for word in TFVac:
        IDFvac[word]=IDFvac.get(word,0)+log(len(docVec)/(DFvac[word]+0.01))
    return docVec,TFVac,DFvac,IDFvac

#feature list,主要是针对term部分，text maps to a vector space，term是最基本的特征，可以考虑加入unigram+bigram+LDA，然后根据
#实际数据集进行调整，将feature量化,比如时间信息,可以按区间划分成为bin,然后特征可以append(new feature)
#text embedding:tf*idf tf*bdc one-hot word_embedding/average doc2vec,fixed_size->add feature

def getTextVecByTF_IDF(docV,featurelist,IDFdic):
    docVec=[]
    for doc in docV:
        words=[]
        for word in featurelist:
            if word in doc:
                tf = doc.count(word) / len(doc)
                tf_idf = tf * IDFdic[word]
                words.append(tf_idf)
            else:
                words.append(0)
        docVec.append(words)
    return docVec

#将term-category space取平均
def getTextVecByTF_IDF_category(docV,featurelist,IDFdic,term_category):
    docVec=[]
    for doc in docV:
        words=[]
        category_space=[]
        count_term_category=0
        for word in featurelist:
            if word in doc:
                tf = doc.count(word) / len(doc)
                tf_idf = tf * IDFdic[word]
                words.append(tf_idf)
                if(term_category.get(word)is not None):
                    count_term_category+=1
                    category_space+=np.array(term_category[word])
            else:
                words.append(0)
        category_space=category_space/count_term_category
        category_space=category_space.tolist()
        words.append(category_space)
        docVec.append(words)
    return docVec

def getTextVecByTF_BDC(docV,featurelist,Bdcdic):
    docVec=[]
    for doc in docV:
        words=[]
        for word in doc:
            #这里可以考虑加入量化的特征,如LAD等
            if(featurelist.get(word) is not None):
                tf = doc.count(word) / len(doc)
                tf_idf = tf * Bdcdic[word]
                words.append(tf_idf)
            else:
                words.append(0)
        docVec.append(words)
    return docVec

def getTextVecByOne_hot(docV,featurelist):
    docVec=[]
    for doc in docV:
        words=[]
        for word in doc:
            #这里可以考虑加入量化的特征,如LAD等
            if(featurelist.get(word) is not None):
                words.append(1)
            else:
                words.append(0)
        docVec.append(words)
    return docVec

#Wordvec average
def getTextVecByWord2vec(docV,featurelist,word2vec,word2VecSize):
    docVec=[]
    for doc in docV:
        count=1
        words=np.zeros(word2VecSize)
        for word in doc:
            if(featurelist.get(word) is not None):
                words+=word2vec[word]
                count+=1
        words=words/count
        if(count>0):
            docVec.append(words)
    return docVec

def getTextVecByDoc2vec(docV,docID,filename):
    sentences=[]
    for doc in docV:
        sentence=doc2vec.TaggedDocument(words=docV[doc],tags=docID)
        sentences.append(sentence)
    model=doc2vec(sentences)
    model.save(filename)
    return model

#根据DF值选择特征，去掉低频词和高频词
def selectFeatByDF(dfDic,minDF,maxDF):
    featList={}
    for word in dfDic:
        if dfDic[word] > minDF and dfDic[word] < maxDF:
            featList[word]=featList.get(word,0)+1
    return featList