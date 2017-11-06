#encoding：utf-8
#文本的格式是，id\ttitle\tcontent\tlabel
from prepcrocess import *
if __name__=='__main__':
    filename=''
    dictname = 'data/dataset_Dict.txt'
    word2vecfile = 'data/dataset_Word2Vec.txt'
    doc2Vecfilename=''
    #类别标签转换
    labeldic={'positive':1,'negative':-1}
    #word2Vec Dim
    word2VecSize=200
    #按类别输出所有的doc及其字典,这里是为了找强特,集中在某几个类别出现的词可以单独作为特征append
    #stopWfilenam = ''
    #stopwords = getStopWords(stopWfilenam)
    stopwords={}
    docs,docID,labels,labelDict=getDoc(filename,labeldic)
    print(labelDict)
