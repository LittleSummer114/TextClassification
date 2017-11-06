from __future__ import print_function
import time
import numpy as np
import pandas as pd
from math import *
import re
from sklearn.datasets import *
import matplotlib.pyplot as plt
from sklearn.model_selection import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.cluster import *
from sklearn.svm import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import *
from sklearn.naive_bayes import *
from sklearn.externals import joblib
import jieba
# import jieba.posseg as pg

# import keras
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# import tensorflow as tf
from gensim.models import *
# import pynlpir
# from lightgbm import LGBMClassifier
# from xgboost.sklearn import XGBClassifier