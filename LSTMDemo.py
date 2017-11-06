#encoding：utf-8
#文本的格式是，id\ttitle\tcontent\tlabel
from prepcrocess import *
if __name__=='__main__':
    filename=''
    dictname = '../data/dataset_Dict.txt'
    word2vecfile = '../data/dataset_Word2Vec.txt'
    doc2Vecfilename=''
    #类别标签转换
    labeldic={'positive':1,'negative':-1}
    #word2Vec Dim
    EMBEDDING_DIM=200
    #按类别输出所有的doc及其字典,这里是为了找强特,集中在某几个类别出现的词可以单独作为特征append
    stopWfilenam = ''
    stopwords = getStopWords(stopWfilenam)
    docs,docID,labels,labelDict=getDoc(filename,labeldic)
    stopwords=getStopWords(stopWfilenam)
    docV, AllTFdic, DFdic, IDFdic,word2vec= getFeature(docs,stopwords,dictname,word2vecfile,word2VecSize)
    bdc={}
    bdcFile=open('','w',encoding='utf-8')
    for categorylabel in labeldic:
        resultname='../data/dataset_'+categorylabel+'.txt'
        getSpeCateDoc(filename, categorylabel, resultname)
        docs, docID,labels, labelDict = getDoc(resultname, labeldic)
        dictname = '../data/dataset_' + categorylabel + '_Dict.txt'
        dictname = '../data/dataset_' + categorylabel + '_Word2Vec.txt'
        docV, AllTFdic, DF_category_dic, IDFdic, word2vec= getFeature(docs,stopwords,dictname,word2vecfile,word2VecSize)
        for word in DF_category_dic:
            bdc[word]+=DF_category_dic[word]/DFdic[word]*log(DF_category_dic[word]/DFdic[word])
    for word in DFdic:
        bdc[word]=1+bdc[word]
    bdcDict=sorted(bdc.items(),key=lambda word:word[1],reversed=False)
    for word in bdcDict:
        bdcFile.write('%s,%d'%(word,bdcDict[word]))
    #这里是用AllTF/DF做特征选择
    minDF=0
    maxDF=10000
    featurelist=selectFeatByDF(DFdic,minDF,maxDF)
    max_words=len(featurelist)
    # text embedding,TF*IDF/bdc,fasttext,bag of words,此处用tf*idf
    #docVec=getTextVecByTF_BDC(docV,featurelist,bdc)
    #docVec=getTextVecByOne_hot(docV,featurelist,bdc)
    #word2Vec就不用np.array()
    #docVec=getTextVecByWord2vec(docV,featurelist,word2vec,word2VecSize)
    #docVec=getTextVecByDoc2vec(docV,docID,doc2Vecfilename)
    docVec=getTextVecByTF_IDF(docV,featurelist,IDFdic)
    max_words=len(featurelist)
    datasetX = np.array(docVec)
    datasetY = to_categorical(labels,len(set(labels)))
    kf = StratifiedKFold(n_splits=n_split)
    num_classes = np.max(labels) + 1
    print(num_classes, 'classes')
    print('Building model...')
    batch_size = 32
    epochs = 5
    model = Sequential()
    model.add(Embedding(len(DFdic), EMBEDDING_DIM,
                        input_length=len(featurelist)))
    model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(datasetX,'datasetX')
    print('datasetY',datasetY)
    train_X, test_X, train_y, test_y = train_test_split(datasetX,datasetY,train_size=0.8)
    model.fit(train_X, train_y,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1)
    score=model.evaluate(test_X, test_y,batch_size=batch_size, verbose=1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])