#encoding:utf-8

import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
'''
模型部分:标准CNN模型 
'''
class CNN(nn.Module):
    def __init__(self,**kwargs):
        super(CNN,self).__init__()
        self.name='CNN'
        self.concept_matrix = kwargs['concept_vectors']
        self.dimension = kwargs['dimension']
        self.batch_size = kwargs['batch_size']
        self.type = kwargs['type']
        self.classes=kwargs['classes']
        self.number_of_filters = kwargs['number_of_filters']
        self.filter_size = kwargs['filter_size']
        self.epoch=kwargs['epoch']
        self.wv = kwargs['wv_maritx']
        self.dropout_rate = kwargs['dropout']
        self.level = kwargs['level']
        self.VOCABULARY_SIZE = kwargs['VOCABULARY_'+self.level+'_SIZE']
        self.max_sent_length = kwargs['max_sent_'+self.level+'_length']
        self.length_feature = kwargs['length_feature']
        self.max_sent_length+=self.length_feature
        self.embedding=nn.Embedding(self.VOCABULARY_SIZE+2,self.dimension,padding_idx=self.VOCABULARY_SIZE+1)
        #self.embedding.weight=nn.Parameter(torch.LongTensor(self.wv))
        self.relu=nn.ReLU()
        self.channel=1
        if(self.type=='static'):
            self.embedding.weight.requires_grad = False
        if(self.type=='multichannel'):
            self.channel=2
        if(self.type=='non-static' and self.level=='word'):
            self.embedding.weight.data.copy_(torch.from_numpy(self.wv))
        self.dropout=nn.Dropout(p=self.dropout_rate)
        for i in range(len(self.number_of_filters)):
            con=nn.Conv1d(self.channel,self.number_of_filters[i],self.dimension*self.filter_size[i],stride=self.dimension)
            setattr(self,'con_{}'.format(i),con)
        self.fc1=nn.Linear(sum(self.number_of_filters),100)
        self.fc2=nn.Linear(100,len(self.classes))
        self.softmax=nn.Softmax(dim=1)

    def con(self,i):
        return getattr(self,'con_{}'.format(i))

    def forward(self,input):
        x = self.embedding(input).view(-1, 1, self.dimension * self.max_sent_length)
        conv=[]
        for i in range(len(self.number_of_filters)):
            temp=self.con(i)(x)
            temp=self.relu(temp)
            temp=nn.MaxPool1d(self.max_sent_length-self.filter_size[i]+1)(temp).view(-1,self.number_of_filters[i])
            conv.append(temp)
        x=torch.cat(conv,1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

#这里用CNN_attention_来获取概念
class BiGRU_CNN(nn.Module):
    def __init__(self, **kwargs):
        super(BiGRU_CNN, self).__init__()
        self.name = 'BiGRU_CNN'
        self.BATCH_SIZE = kwargs['batch_size']
        self.level = kwargs['level']
        self.number_of_filters = kwargs['number_of_filters']
        self.filter_size = kwargs['filter_size']
        self.dropout_rate = kwargs['dropout']
        self.number_layers = kwargs['num_layers']
        self.length_feature = kwargs['length_feature']
        self.MAX_SENT_LEN = kwargs['max_sent_' + self.level + '_length']
        if(self.length_feature == 1):
             self.MAX_SENT_LEN += 1
        self.VOCAB_SIZE = kwargs['VOCABULARY_'+self.level+'_SIZE']
        self.classes = kwargs['classes']
        self.CLASS_SIZE = len(self.classes)
        self.relu = nn.ReLU()
        self.channel = 1
        self.type = kwargs['type']
        self.WORD_DIM = kwargs['dimension']
        self.DROPOUT_PROB = kwargs['dropout']
        self.WV_MATRIX = kwargs['wv_maritx']
        self.FILTERS = kwargs['filter_size']
        self.FILTER_NUM = kwargs['number_of_filters']
        self.GPU = kwargs['gpu']
        self.idx_to_word = kwargs['idx_to_word']
        self.hidden_size = kwargs['hidden_size']
        self.concept_vectors = kwargs['concept_vectors']
        self.IN_CHANNEL = 1
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if (self.type != 'rand'):
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        self.attend = nn.Linear(self.WORD_DIM, 1)

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.hidden_size * self.FILTERS[i] * 2, stride=self.hidden_size * 2)
            setattr(self, 'conv_{}'.format(i), conv)
        self.gru = nn.GRU(input_size=self.WORD_DIM, hidden_size=self.hidden_size, num_layers=self.number_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        #这个是为了concept能够进行attention做的，其实就相当于一个双曲线，先把所有的都转换成 1*concept_dim
        self.fc1_concept = nn.Linear(2 * self.hidden_size, self.WORD_DIM)
        self.softmax_word = nn.Softmax(dim=0)
        self.fc2_attention = nn.Linear(self.WORD_DIM*2,1)
        self.fc_output = nn.Linear(self.WORD_DIM*2,len(self.classes))
        self.soft_output = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc1 = nn.Linear(sum(self.number_of_filters), 100)
        self.fc2 = nn.Linear(100, len(self.classes))
        self.softmax = nn.Softmax(dim=1)

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))

    def forward(self, inp):
        x = self.embedding(inp)
        output, hidden = self.gru(x)
        print(output)
        #可以说进行了拼接操作 获得了一个词语的当前context的表示加上他的concept表示
        output = torch.cat(output)
        output = output.view(-1,1,self.hidden_size*2*self.MAX_SENT_LEN)
        # output_shape:batch_size*seq_len*2*self.word_embedding_dim
        conv = []
        for i in range(len(self.number_of_filters)):
            temp = self.get_conv(i)(output)
            temp = self.relu(temp)
            temp = nn.MaxPool1d(self.MAX_SENT_LEN - self.filter_size[i] + 1)(temp).view(-1,
                                                                                           self.number_of_filters[i])
            conv.append(temp)
        x = torch.cat(conv, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

#RCNN,先用rnn获得词语的新的表示,然后再做卷积
class BiGRU_CONCEPT_CNN(nn.Module):
    def __init__(self, **kwargs):
        super(BiGRU_CONCEPT_CNN, self).__init__()
        self.name = 'BiGRU_CONCEPT_CNN'
        self.BATCH_SIZE = kwargs['batch_size']
        self.level = kwargs['level']
        self.number_of_filters = kwargs['number_of_filters']
        self.filter_size = kwargs['filter_size']
        self.dropout_rate = kwargs['dropout']
        self.number_layers = kwargs['num_layers']
        self.length_feature = kwargs['length_feature']
        self.MAX_SENT_LEN = kwargs['max_sent_' + self.level + '_length']
        if(self.length_feature == 1):
             self.MAX_SENT_LEN += 1
        self.VOCAB_SIZE = kwargs['VOCABULARY_'+self.level+'_SIZE']
        self.classes = kwargs['classes']
        self.CLASS_SIZE = len(self.classes)
        self.relu = nn.ReLU()
        self.channel = 1
        self.type = kwargs['type']
        self.WORD_DIM = kwargs['dimension']
        self.DROPOUT_PROB = kwargs['dropout']
        self.WV_MATRIX = kwargs['wv_maritx']
        self.FILTERS = kwargs['filter_size']
        self.FILTER_NUM = kwargs['number_of_filters']
        self.GPU = kwargs['gpu']
        self.idx_to_word = kwargs['idx_to_word']
        self.hidden_size = kwargs['hidden_size']
        self.concept_vectors = kwargs['concept_vectors']
        self.IN_CHANNEL = 1
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if (self.type != 'rand'):
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        self.attend = nn.Linear(self.WORD_DIM, 1)

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i] * 2, stride=self.WORD_DIM * 2)
            setattr(self, 'conv_{}'.format(i), conv)
        self.gru = nn.GRU(input_size=self.WORD_DIM, hidden_size=self.hidden_size, num_layers=self.number_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        #这个是为了concept能够进行attention做的，其实就相当于一个双曲线，先把所有的都转换成 1*concept_dim
        self.fc1_concept = nn.Linear(2 * self.hidden_size, self.WORD_DIM)
        self.softmax_word = nn.Softmax(dim=0)
        self.fc2_attention = nn.Linear(self.WORD_DIM*2,1)
        self.fc_output = nn.Linear(self.WORD_DIM*2,len(self.classes))
        self.soft_output = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc1 = nn.Linear(sum(self.number_of_filters), 100)
        self.fc2 = nn.Linear(100, len(self.classes))
        self.softmax = nn.Softmax(dim=1)

    def get_concept(self, matrix, context):
        #matrix: batch_size*seq_len  context:batch_size*seql_len*embedding_dim
        context_np = context.cpu().data.numpy()
        concept_matrix = []
        for index, sent in enumerate(matrix):
            concept_sent = []
            for index_word,word in enumerate(sent):
                # 每个词语自己的hidden state vector作为它的context
                context_word = context[index][index_word]
                context_word = context_word.view(300,1)
                if int(word) < self.VOCAB_SIZE:
                    word = self.idx_to_word[int(word)]
                    word = word.lower()
                    if word in self.concept_vectors.keys():
                        concepts = self.concept_vectors[word]
                        #如果一個詞沒有concept
                        if len(concepts) == 0:
                            concept = np.random.uniform(-0.01, 0.01, self.WORD_DIM).astype('float')
                        else:
                            first = concepts[:300]
                            word_concepts = []
                            word_concepts.append(first)
                            word_concepts+=concepts[300:]
                            #这里应该获得每个词语的context
                            # word_concepts: n*word_embedding,有可能只有一個
                            if(self.GPU!=-1):
                                word_concepts = Variable(torch.FloatTensor(word_concepts)).cuda(self.GPU)
                            else:
                                word_concepts = Variable(torch.FloatTensor(word_concepts))
                            # word context_word concepts: word embedding_dim*1
                            attention_word_concept = torch.matmul(word_concepts,context_word)
                            attention_word_concept = self.softmax_word(attention_word_concept).view(1, -1)
                            # attention_word_concept: n * 1 --> 1 * n * n * embedding_dim == 1 * embedding_dim,再转换成embedding_dim即可
                            concept = torch.matmul(attention_word_concept, word_concepts).view(self.WORD_DIM)
                            concept = concept.cpu().data.numpy().astype('float32')
                    else:
                        concept = np.random.uniform(-0.01, 0.01, self.WORD_DIM).astype('float32')
                elif int(word) == self.VOCAB_SIZE:
                    concept = np.random.uniform(-0.01, 0.01, self.WORD_DIM).astype('float32')
                else:
                    concept = np.zeros(self.WORD_DIM).astype('float32')
                concept_sent.append(concept)
            # batch_size * seq_len * concept_dim
            concept_matrix.append(concept_sent)
        concept_matrix = np.array(concept_matrix)
        #concept_matrix = np.array(concept_matrix)
        if(self.GPU == -1):
            concept_matrix = Variable(torch.from_numpy(concept_matrix))
        else:
            concept_matrix = Variable(torch.from_numpy(concept_matrix)).cuda(self.GPU)
        return concept_matrix

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))

    def forward(self, inp):
        x = self.embedding(inp)
        output, hidden = self.gru(x)
        output = self.fc1_concept(output)
        concepts = self.get_concept(inp,output)
        #可以说进行了拼接操作 获得了一个词语的当前context的表示加上他的concept表示
        output = torch.cat([output,concepts],2).view(-1,1,self.WORD_DIM*2*self.MAX_SENT_LEN)
        # output_shape:batch_size*seq_len*2*self.word_embedding_dim
        conv = []
        for i in range(len(self.number_of_filters)):
            temp = self.get_conv(i)(output)
            temp = self.relu(temp)
            temp = nn.MaxPool1d(self.MAX_SENT_LEN - self.filter_size[i] + 1)(temp).view(-1,
                                                                                           self.number_of_filters[i])
            conv.append(temp)
        x = torch.cat(conv, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
'''
模型部分:上下两个部分的拼接:word-concept + char-level
'''
class BiGRU(nn.Module):
    def __init__(self, **kwargs):
        super(BiGRU_Attention, self).__init__()
        self.name = 'BiGRU_Attention'
        self.BATCH_SIZE = kwargs['batch_size']
        self.level = kwargs['level']
        self.length_feature = kwargs['length_feature']
        self.MAX_SENT_LEN = kwargs['max_sent_' + self.level + '_length']
        if(self.length_feature == 1):
             self.MAX_SENT_LEN += 1
        self.VOCAB_SIZE = kwargs['VOCABULARY_'+self.level+'_SIZE']
        self.classes = kwargs['classes']
        self.CLASS_SIZE = len(self.classes)
        self.WORD_DIM = kwargs['dimension']
        self.DROPOUT_PROB = kwargs['dropout']
        self.WV_MATRIX = kwargs['wv_maritx']
        self.FILTERS = kwargs['filter_size']
        self.FILTER_NUM = kwargs['number_of_filters']
        self.GPU = kwargs['gpu']
        self.type = kwargs['type']
        self.num_layers = kwargs['num_layers']
        self.idx_to_word = kwargs['idx_to_word']
        self.hidden_size = kwargs['hidden_size']
        self.concept_vectors = kwargs['concept_vectors']
        self.IN_CHANNEL = 1
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if(self.type !='rand'):
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        self.attend = nn.Linear(self.WORD_DIM, 1)

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i] * 2, stride=self.WORD_DIM * 2)
            setattr(self, 'conv_{}'.format(i), conv)

        self.gru = nn.GRU(input_size=self.WORD_DIM, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        #这个是为了concept能够进行attention做的，其实就相当于一个双曲线，先把所有的都转换成 1*concept_dim
        self.softmax_word = nn.Softmax(dim=0)
        self.fc2_attention = nn.Linear(self.hidden_size*2,1)
        self.fc_output = nn.Linear(self.hidden_size*2,len(self.classes))
        self.soft_output = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))

    def forward(self, inp):
        x = self.embedding(inp)
        output, hidden = self.gru(x)
        # output_shape:batch*seq_len*2*hidden_size
        #之后不用CNN直接加Attention看能不能取得很好的效果
        attention = self.fc2_attention(output)
        # output_shape:batch*seql_len*1
        attention = self.tanh(attention)
        attn_weights = self.soft_output(attention).view(-1,1,self.MAX_SENT_LEN)
        # attn_weights:batch*1*seq_len
        output = torch.bmm(attn_weights, output)
        # output:batch*1*2*hidden_size
        output = output.view(-1,self.hidden_size*2)
        output = self.fc_output(output)
        output = self.soft_output(output)
        return output

class BiGRU_Attention(nn.Module):
    def __init__(self, **kwargs):
        super(BiGRU_Attention, self).__init__()
        self.name = 'BiGRU_Attention'
        self.BATCH_SIZE = kwargs['batch_size']
        self.level = kwargs['level']
        self.length_feature = kwargs['length_feature']
        self.MAX_SENT_LEN = kwargs['max_sent_' + self.level + '_length']
        if(self.length_feature == 1):
             self.MAX_SENT_LEN += 1
        self.VOCAB_SIZE = kwargs['VOCABULARY_'+self.level+'_SIZE']
        self.classes = kwargs['classes']
        self.CLASS_SIZE = len(self.classes)
        self.WORD_DIM = kwargs['dimension']
        self.DROPOUT_PROB = kwargs['dropout']
        self.WV_MATRIX = kwargs['wv_maritx']
        self.FILTERS = kwargs['filter_size']
        self.FILTER_NUM = kwargs['number_of_filters']
        self.GPU = kwargs['gpu']
        self.num_layers = kwargs['num_layers']
        self.idx_to_word = kwargs['idx_to_word']
        self.type = kwargs['type']
        self.hidden_size = kwargs['hidden_size']
        self.concept_vectors = kwargs['concept_vectors']
        self.IN_CHANNEL = 1
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if (self.type != 'rand'):
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        self.attend = nn.Linear(self.WORD_DIM, 1)

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i] * 2, stride=self.WORD_DIM * 2)
            setattr(self, 'conv_{}'.format(i), conv)

        self.gru = nn.GRU(input_size=self.WORD_DIM, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        #这个是为了concept能够进行attention做的，其实就相当于一个双曲线，先把所有的都转换成 1*concept_dim
        self.softmax_word = nn.Softmax(dim=0)
        self.fc2_attention = nn.Linear(self.hidden_size*2,1)
        self.fc_output = nn.Linear(self.hidden_size*2,len(self.classes))
        self.soft_output = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))

    def forward(self, inp):
        x = self.embedding(inp)
        output, hidden = self.gru(x)
        # output_shape:batch*seq_len*2*hidden_size
        #之后不用CNN直接加Attention看能不能取得很好的效果
        attention = self.fc2_attention(output)
        # output_shape:batch*seql_len*1
        attention = self.tanh(attention)
        attn_weights = self.soft_output(attention).view(-1,1,self.MAX_SENT_LEN)
        # attn_weights:batch*1*seq_len
        output = torch.bmm(attn_weights, output)
        # output:batch*1*2*hidden_size
        output = output.view(-1,self.hidden_size*2)
        output = self.fc_output(output)
        output = self.soft_output(output)
        return output

class BiGRU_CONCEPT_Attention(nn.Module):
    def __init__(self, **kwargs):
        super(BiGRU_CONCEPT_Attention, self).__init__()
        self.name = 'BiGRU_CONCEPT_Attention'
        self.BATCH_SIZE = kwargs['batch_size']
        self.level = kwargs['level']
        self.number_layers = kwargs['num_layers']
        self.length_feature = kwargs['length_feature']
        self.MAX_SENT_LEN = kwargs['max_sent_' + self.level + '_length']
        if(self.length_feature == 1):
             self.MAX_SENT_LEN += 1
        self.VOCAB_SIZE = kwargs['VOCABULARY_'+self.level+'_SIZE']
        self.classes = kwargs['classes']
        self.CLASS_SIZE = len(self.classes)
        self.type = kwargs['type']
        self.WORD_DIM = kwargs['dimension']
        self.DROPOUT_PROB = kwargs['dropout']
        self.WV_MATRIX = kwargs['wv_maritx']
        self.FILTERS = kwargs['filter_size']
        self.FILTER_NUM = kwargs['number_of_filters']
        self.GPU = kwargs['gpu']
        self.idx_to_word = kwargs['idx_to_word']
        self.hidden_size = kwargs['hidden_size']
        self.concept_vectors = kwargs['concept_vectors']
        self.IN_CHANNEL = 1
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if (self.type != 'rand'):
            self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
        self.attend = nn.Linear(self.WORD_DIM, 1)

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i] * 2, stride=self.WORD_DIM * 2)
            setattr(self, 'conv_{}'.format(i), conv)
        self.gru = nn.GRU(input_size=self.WORD_DIM, hidden_size=self.hidden_size, num_layers=self.number_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        #这个是为了concept能够进行attention做的，其实就相当于一个双曲线，先把所有的都转换成 1*concept_dim
        self.fc1_concept = nn.Linear(2 * self.hidden_size, self.WORD_DIM)
        self.softmax_word = nn.Softmax(dim=0)
        self.fc2_attention = nn.Linear(self.WORD_DIM*2,1)
        self.fc_output = nn.Linear(self.WORD_DIM*2,len(self.classes))
        self.soft_output = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def get_concept(self, matrix, context):
        #matrix: batch_size*seq_len  context:batch_size*seql_len*embedding_dim
        context_np = context.cpu().data.numpy()
        concept_matrix = []
        for index, sent in enumerate(matrix):
            concept_sent = []
            for index_word,word in enumerate(sent):
                # 每个词语自己的hidden state vector作为它的context
                context_word = context[index][index_word]
                context_word = context_word.view(300,1)
                if int(word) < self.VOCAB_SIZE:
                    word = self.idx_to_word[int(word)]
                    word = word.lower()
                    if word in self.concept_vectors.keys():
                        concepts = self.concept_vectors[word]
                        #如果一個詞沒有concept
                        if len(concepts) == 0:
                            concept = np.random.uniform(-0.01, 0.01, self.WORD_DIM).astype('float')
                        else:
                            first = concepts[:300]
                            word_concepts = []
                            word_concepts.append(first)
                            word_concepts+=concepts[300:]
                            #这里应该获得每个词语的context
                            # word_concepts: n*word_embedding,有可能只有一個
                            if(self.GPU!=-1):
                                word_concepts = Variable(torch.FloatTensor(word_concepts)).cuda(self.GPU)
                            else:
                                word_concepts = Variable(torch.FloatTensor(word_concepts))
                            # word context_word concepts: word embedding_dim*1
                            attention_word_concept = torch.matmul(word_concepts,context_word)
                            attention_word_concept = self.softmax_word(attention_word_concept).view(1, -1)
                            # attention_word_concept: n * 1 --> 1 * n * n * embedding_dim == 1 * embedding_dim,再转换成embedding_dim即可
                            concept = torch.matmul(attention_word_concept, word_concepts).view(self.WORD_DIM)
                            concept = concept.cpu().data.numpy().astype('float32')
                    else:
                        concept = np.random.uniform(-0.01, 0.01, self.WORD_DIM).astype('float32')
                elif int(word) == self.VOCAB_SIZE:
                    concept = np.random.uniform(-0.01, 0.01, self.WORD_DIM).astype('float32')
                else:
                    concept = np.zeros(self.WORD_DIM).astype('float32')
                concept_sent.append(concept)
            # batch_size * seq_len * concept_dim
            concept_matrix.append(concept_sent)
        concept_matrix = np.array(concept_matrix)
        #concept_matrix = np.array(concept_matrix)
        if(self.GPU == -1):
            concept_matrix = Variable(torch.from_numpy(concept_matrix))
        else:
            concept_matrix = Variable(torch.from_numpy(concept_matrix)).cuda(self.GPU)
        return concept_matrix

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))

    def forward(self, inp):
        x = self.embedding(inp)
        output, hidden = self.gru(x)
        output = self.fc1_concept(output)
        concepts = self.get_concept(inp,output)
        #可以说进行了拼接操作 获得了一个词语的当前context的表示加上他的concept表示
        output = torch.cat([output,concepts],2)
        #之后不用CNN直接加Attention看能不能取得很好的效果
        attention = self.fc2_attention(output)
        attention = self.tanh(attention)
        attn_weights = self.soft_output(attention).view(-1,1,self.MAX_SENT_LEN)
        output = torch.bmm(attn_weights, output)
        output = output.view(-1,self.WORD_DIM*2)
        output = self.fc_output(output)
        output = self.soft_output(output)
        return output
