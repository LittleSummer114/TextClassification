#encoding：utf-8
#文本的格式是,label\tcontent
from prepcrocess import *
from import_file import *

if __name__=='__main__':
    filename='data/train.txt'
    stopWfilenam = 'data/EN_stopWords.txt'
    AllTFdicfilename='data/TfDict.txt'
    DFdicfilename='data/DfDict.txt'
    IDFdicfilename='data/IdfDict.txt'
    BDCdicfilename='data/BdcDict.txt'
    CategoryDictfilename='data/CategoryDict.txt'
    #K折交叉验证
    n_split=4
    #word2Vec Dim
    word2VecSize=200
    #按类别输出所有的doc及其字典,这里是为了找强特,集中在某几个类别出现的词可以单独作为特征append
    stopwords = getStopWords(stopWfilenam)
    time1=time.time()
    docs,labels,labelDict=getDoc(filename)
    category_size=len(labelDict)
    time2=time.time()
    print('Getting Data:',str(time2-time1))
    print('labelDict',labelDict)
    time3=time.time()

    docV,AllTFdic,DFdic,IDFdic= getDict_EN(docs, stopwords)
    getDictFile(AllTFdic,AllTFdicfilename,True)
    getDictFile(DFdic, DFdicfilename, True)
    getDictFile(IDFdic, IDFdicfilename,False)
    bdcDict={}
    #将每个term映射到category space
    term_category={}
    for categorylabel in labelDict:
        resultname = 'data/dataset_' + categorylabel + '.txt'
        getSpeCateDoc(filename, categorylabel, resultname)
        specialdocs,speciallabels,speciallabelDict = getDoc(resultname)
        specialdocV, AllTFdic, DF_category_dic, IDFdic = getDict_EN(specialdocs, stopwords)
        for word in DFdic:
            if(word not in DF_category_dic):
                if (term_category.get(word) is None):
                    term_category[word] = [0]
                else:
                    term_category[word].append(0)
                continue
            bdcDict[word]=bdcDict.get(word,0)+DF_category_dic[word] / DFdic[word] * log(DF_category_dic[word] / DFdic[word])
            if(term_category.get(word) is None):
                term_category[word]=[DF_category_dic[word]/DFdic[word]]
            else:
                term_category[word].append(DF_category_dic[word]/DFdic[word])
    #用一个文件把term-category space存储起来
    CategoryDictfile = open(CategoryDictfilename, 'w', encoding='utf-8')
    for word in term_category:
        CategoryDictfile.write('%s\t%s\n'% (word,term_category[word]))
    for word in bdcDict:
        bdcDict[word] = 1 + bdcDict[word]/log(category_size)
    getDictFile(bdcDict,BDCdicfilename,False)

    bdcDict=readDictFile(BDCdicfilename)
    term_category=readDictFile(CategoryDictfilename)
    AllTFdic = readDictFile(AllTFdicfilename)
    DFdic = readDictFile(DFdicfilename)
    IDFdic = readDictFile(IDFdicfilename)
    time4 = time.time()
    print('Getting Dict:', str(time4 - time3))
    print('term_category',term_category)
    # 这里是用AllTF/DF做特征选择
    minDF = 0
    maxDF = 10000
    k = 10
    print('DFdic',DFdic)
    featurelist = selectFeatByDF(DFdic, minDF, maxDF)
    time5 = time.time()
    print('Getting FeatureList:', str(time5 - time4))
    #text embedding,TF*IDF/bdc,fasttext,bag of words,此处用tf*idf
    #docVec=getTextVecByTF_BDC(docV,featurelist,bdcDict)
    # # docVec=getTextVecByOne_hot(docV,featurelist,bdc)
    # # word2Vec就不用np.array()
    # # docVec=getTextVecByWord2vec(docV,featurelist,word2vec,word2VecSize)
    # # docVec=getTextVecByDoc2vec(docV,docID,doc2Vecfilename)
    docVec = getTextVecByTF_IDF(docV, featurelist, IDFdic)
    time6 = time.time()
    print('Getting Text:', str(time6 - time5))
    #docVec = getTextVecByTF_IDF_category(docV, featurelist, IDFdic,term_category)
    datasetX = np.array(docVec)
    datasetY = np.array(labels)
    print(datasetX)
    print(datasetY)
    kf = StratifiedKFold(n_splits=n_split)
    model = KNeighborsClassifier(n_neighbors=k)

    result_micro_f1=[]
    result_macro_f1=[]
    step=0
    model.fit(datasetX,datasetY)
    time7 = time.time()
    print('Finished Training:', str(time7 - time6))
    joblib.dump(model,'data/train_model.m')

    # #加载验证集
    # clf = joblib.load('data/train_model.m')
    # pred_y = model.predict(datasetX)
    # micro_f1=f1_score(datasetY, pred_y, average='micro')
    # macro_f1 = f1_score(datasetY, pred_y, average='macro')
    # result_micro_f1.append(micro_f1)
    # result_macro_f1.append(macro_f1)
    # print('micro_f1:%s'%(step,micro_f1))
    # print('macro_f1:%s' % (step, macro_f1))

    #交叉验证结果
    # for train_index, test_index in kf.split(datasetX, datasetY):
    #     train_X = datasetX[train_index]
    #     train_y = datasetY[train_index]
    #     test_X = datasetX[test_index]
    #     test_y = datasetY[test_index]
    #     model.fit(train_X, train_y)
    #     pred_y = model.predict(test_X)
    #     micro_f1=f1_score(test_y, pred_y, average='micro')
    #     macro_f1 = f1_score(test_y, pred_y, average='macro')
    #     result_micro_f1.append(micro_f1)
    #     result_macro_f1.append(macro_f1)
    #     print('第%s次迭代,micro_f1:%s'%(step,micro_f1))
    #     print('第%s次迭代,macro_f1:%s' % (step, macro_f1))
    #     step+=1

    #可视化结果
    # x=[]
    # for word in result_micro_f1:
    #     x.append(result_micro_f1.index(word))
    #
    # plt.subplot(2, 1, 1)  # 面板设置成2行1列，并取第一个（顺时针编号）
    # plt.plot(x, result_micro_f1, 'yo-')  # 画图，染色
    # plt.title('A tale of 2 subplots')
    # plt.ylabel('micro-f1')
    #
    # plt.subplot(2, 1, 2)  # 面板设置成2行1列，并取第二个（顺时针编号）
    # plt.plot(x,result_macro_f1, 'r.-')  # 画图，染色
    # plt.xlabel('K-fold')  # x轴标签
    # plt.ylabel('macro-f1')  # y轴标签
    # plt.show()