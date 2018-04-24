#encoding:utf-8
import os
'''
统一进行调参 有新的参数往这里添加
'''
hidden_sizes = [
    256,
   # 128,
   # 64
]
number_layers = [
    1
]
models=[
    'CNN',
]
modes=[
    'train'
]
datasets=[
'AG'
]
max_features=[
   5000,
    8000,
   # 15000,
   # 5000,
   # 1000,
]
levels=[
  # 'char',
    'word',
]
length_features=[
    #0,
    1,
]
wvs=[
    'word2vec',
]
epochs=[
50
]
batch_sizes=[
    # 50000,
    # 30000,
   # 10000,
 20,
             ]
learning_rates=[
   # 0.01,
   # 0.03,
   # 0.001,
   # 0.003,
    #0.1,
    0.3,
    1,
]
metrics=['ACC']
dimensions=[
    300,
]
max_sent_word_lengths=[
    20,
    40,
    30
]
max_sent_char_lengths=[
    150,
   # 100,
   # 50,
]
filter_sizes=[
    ['3','4','5','6']
]
num_of_filters=[
    ['100','100','100','100']
]
gpu={'AG':0,'Subj':1,'MRS':2,'TREC':3}
optimizers=[
    'Adadelta',
    #'SGD'
]

if __name__=='__main__':
    for i in range(10):
        for dataset in datasets:
            for model in models:
                for mode in modes:
                    for wv in wvs:
                        for batch_size in batch_sizes:
                            for learning_rate in learning_rates:
                                for epoch in epochs:
                                    for dimension in dimensions:
                                        for max_sent_word_length in max_sent_word_lengths:
                                            for filter_size in filter_sizes:
                                                for num_of_filter in num_of_filters:
                                                    for optimizer in optimizers:
                                                        for max_sent_char_length in max_sent_char_lengths:
                                                            for level in levels:
                                                                for length_feature in length_features:
                                                                    for max_feature in max_features:
                                                                        for hidden_size in hidden_sizes:
                                                                            for number_layer in number_layers:
                                                                                os.system('python3 run.py --model {} '
                                                                                          '--mode {} '
                                                                                          '--wv {} '
                                                                                          '--batch_size {} '
                                                                                          '--learning_rate {} '
                                                                                          '--epoch {} '
                                                                                          '--dimension {} '
                                                                                          '--max_sent_word_length {} '
                                                                                          '--max_sent_char_length {} '
                                                                                          '--filter_size {} '
                                                                                          '--number_of_filters {} '
                                                                                          '--optimizer {} '
                                                                                          '--level {} '
                                                                                          '--length_feature {} '
                                                                                          '--gpu {} '
                                                                                          '--dataset {} '
                                                                                          '--max_features {} '
                                                                                          '--hidden_size {} '
                                                                                          '--num_layers {} '
                                                                                    .format(
                                                                                    model,
                                                                                    mode,
                                                                                    wv,
                                                                                    batch_size,
                                                                                    learning_rate,
                                                                                    epoch,
                                                                                    dimension,
                                                                                    max_sent_word_length,
                                                                                    max_sent_char_length,
                                                                                    ' '.join(filter_size),
                                                                                    ' '.join(num_of_filter),
                                                                                    optimizer,
                                                                                    level,
                                                                                    length_feature,
                                                                                    gpu[dataset],
                                                                                    dataset,
                                                                                    max_feature,
                                                                                    hidden_size,
                                                                                    number_layer
                                                                                ))
