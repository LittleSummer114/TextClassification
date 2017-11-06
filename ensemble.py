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
    #K折交叉验证
    n_split=4
    #word2Vec Dim
    word2VecSize=200
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
    # text embedding,TF*IDF/bdc,fasttext,bag of words,此处用tf*idf
    #docVec=getTextVecByTF_BDC(docV,featurelist,bdc)
    #docVec=getTextVecByOne_hot(docV,featurelist,bdc)
    #word2Vec就不用np.array()
    #docVec=getTextVecByWord2vec(docV,featurelist,word2vec,word2VecSize)
    #docVec=getTextVecByDoc2vec(docV,docID,doc2Vecfilename)
    docVec=getTextVecByTF_IDF(docV,featurelist,IDFdic)
    datasetX = np.array(docVec)
    datasetY = np.array(labels)
    kf = StratifiedKFold(n_splits=n_split)
    #model=RandomForestClassifier()
    model = GradientBoostingClassifier()
    #model = LGBMClassifier()
    #model=XGBClassifier()
    for train_index, test_index in kf.split(datasetX, datasetY):
        train_X = datasetX[train_index]
        train_y = datasetY[train_index]
        test_X = datasetX[test_index]
        test_y = datasetY[test_index]
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        print('micro_f1', f1_score(test_y, pred_y, average='micro'))
        print('macro_f1', f1_score(test_y, pred_y, average='macro'))