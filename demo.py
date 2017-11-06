#encoding：utf-8
#文本的格式是，id\ttitle\tcontent\tlabel
from import_file import *

if __name__=='__main__':
    max_words=3
    n_split=2
    EMBEDDING_DIM=3
    MAX_SEQUENCE_LENGTH=3
    docVec=[[[1,2,4],[1,2,4],[4,3,6]],[[1,2,4],[1,2,4],[4,3,6]],[[1,2,4],[1,2,4],[4,3,6]],[[1,2,4],[1,2,4],[4,3,6]],[[1,2,4],[1,2,4],[4,3,6]]]
    #docVec = [[1, 2, 4], [1, 2, 4], [4, 3, 6], [1, 2, 4], [4, 3, 6]]
    labels=[1,0,1,0,1]
    datasetX = np.array(docVec)
    datasetY = to_categorical(labels,len(set(labels)))
    kf = StratifiedKFold(n_splits=n_split)
    num_classes = np.max(labels) + 1
    print(num_classes, 'classes')
    print('Building model...')
    batch_size = 32
    epochs = 5
    print(docVec[0])
    model = Sequential()
    model.add(Embedding(5,3,input_length=3))
    model.add(Dropout(0.5))
    model.add(Conv1D(2,3, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(1))
    model.add(Flatten())
    model.add(Dense(EMBEDDING_DIM, activation='relu'))
    model.add(Dense(len(set(labels)), activation='softmax'))
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