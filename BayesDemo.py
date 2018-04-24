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
        self.tanh = nn.Tanh()
        self.softmax=nn.Softmax(dim=1)

    def con(self,i):
        return getattr(self,'con_{}'.format(i))

    def forward(self,input):
        print(input.shape())
        x = self.embedding(input).view(-1, 1, self.dimension * self.max_sent_length)
        conv=[]
        for i in range(len(self.number_of_filters)):
            temp=self.con(i)(x)
            temp=self.relu(temp)
            temp=nn.MaxPool1d(self.max_sent_length-self.filter_size[i]+1)(temp).view(-1,self.number_of_filters[i])
            conv.append(temp)
        x = torch.cat(conv,1)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

'''
模型部分:标准RNN模型
'''

class RNN(nn.Module):
    def __init__(self, **kwargs):
        super(RNN, self).__init__()
        self.name = 'RNN'
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

        self.gru = nn.RNN(input_size=self.WORD_DIM, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        #这个是为了concept能够进行attention做的，其实就相当于一个双曲线，先把所有的都转换成 1*concept_dim
        self.softmax_word = nn.Softmax(dim=0)
        self.fc2_attention = nn.Linear(self.hidden_size*2,1)
        self.fc_output = nn.Linear(self.hidden_size,len(self.classes))
        self.soft_output = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

    def get_conv(self, i):
        return getattr(self, 'conv_{}'.format(i))

    def forward(self, inp):
        x = self.embedding(inp)
        output, hidden = self.gru(x)
        # output_shape:batch*seq_len*2*hidden_size
        output = output.permute(1,0,2)
        # output_shape:seq_len*batch_size*2*hidden_size
        output = output[-1]
        output = self.fc_output(output)
        output = self.soft_output(output)
        return output
