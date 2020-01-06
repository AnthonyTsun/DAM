import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Flatten,Dense,Input,Dropout,multiply, LSTM,Bidirectional,Permute, Concatenate
from keras.models import Model
from keras.optimizers import Adam
import scipy.io as sc 
from sklearn import preprocessing
from keras import regularizers
import keras
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from itertools import cycle
from scipy import interp
import matplotlib.pyplot as plt
import csv
from sklearn import metrics
import tensorflow as tf
from keras import backend as K

Delta = 		4
lstm_units = 	128
Sclass = 		5
no_fea=			64
epochs=			30 
#for sample dataset setting
batch_size=		64
middle=			21000
final=			28000

def one_hot(y_):
    y_=np.array(y_)
    y_ = y_.reshape(len(y_))
    n_values = np.max(y_) + 1
    return np.eye(int(n_values))[np.array(y_, dtype=np.int32)]


#EEG eegmmidb person dependent raw data mixed read
feature = sc.loadmat("sample.mat")					##10 person, 5 class(0-7), person independent, 8*20W, samples, 0-51 features, 52: class label
all = feature['filtered']


np.random.shuffle(all)  # mix eeg_all
all=all[0:29783,:193]

no_fea=all.shape[-1]-1
label_all = all[:, no_fea:no_fea+1]
label_all=np.squeeze(label_all.reshape(1, -1))
result = [a - 1 for a in label_all]
label_all=result

no_fea=64
feature_training_task1 =all[:,0:no_fea]          	#Sub-tasks - Alpha
feature_training_task2 =all[:,no_fea:no_fea*2]   	#Sub-tasks - Beta
feature_training_3 =all[:,no_fea*2:no_fea*3]  		#3-tasks - Raw

feature_normalized_task1=preprocessing.scale(feature_training_task1)
feature_normalized_task2=preprocessing.scale(feature_training_task2)
feature_normalized_3=preprocessing.scale(feature_training_3)


feature1=feature_normalized_task1
feature2=feature_normalized_task2
feature3=feature_normalized_3
label=label_all


INPUT_DIM = int(no_fea/Delta)
x_train1=feature1[0:middle]
x_train1 = x_train1.reshape(21000,Delta,int(no_fea/Delta))

x_train2=feature2[0:middle]
x_train2 = x_train2.reshape(21000,Delta,int(no_fea/Delta))

x_train3=feature3[0:middle]
x_train3 = x_train3.reshape(21000,Delta,int(no_fea/Delta))

x_test1=feature1[middle:final]
x_test1 = x_test1.reshape(7000,Delta,int(no_fea/Delta))

x_test2=feature2[middle:final]
x_test2 = x_test2.reshape(7000,Delta,int(no_fea/Delta))

x_test3=feature3[middle:final]
x_test3 = x_test3.reshape(7000,Delta,int(no_fea/Delta))

y_train=label[0:middle]
y_train =one_hot(y_train)

print(y_train)

y_test=label[middle:final]
y_test = one_hot(y_test) 
print(y_test)
nb_classes = y_test.shape[1]


def attention(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(Delta, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul
 

inputs1 = Input(shape=(Delta,INPUT_DIM))
inputs2 = Input(shape=(Delta,INPUT_DIM))
inputs3 = Input(shape=(Delta,INPUT_DIM))

lstm_out1 = Bidirectional(LSTM(lstm_units, activation='tanh', return_sequences=True, name='bilstm1'))(inputs1)
lstm_out2 = Bidirectional(LSTM(lstm_units, activation='tanh', return_sequences=True, name='bilstm2'))(inputs2)
lstm_out3 = Bidirectional(LSTM(lstm_units, activation='tanh', return_sequences=True, name='bilstm2'))(inputs3)

attention_mul1 = attention(lstm_out1)
attention_mul2 = attention(lstm_out2)
attention_mul3 = attention(lstm_out3)

attention_mul = attention(keras.layers.Add()([attention_mul1, attention_mul2, attention_mul3]))
attention_flatten = Flatten()(attention_mul)


output = Dense(5, activation='softmax',kernel_regularizer=regularizers.l2(0.004))(attention_flatten)

    
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=output)
adam = Adam(lr=1e-4)
model.compile(optimizer='adam', loss='logcosh',  metrics=['accuracy'])

print(model.summary())


while(1):
	print('Training------------')
	model.fit([x_train1, x_train2, x_train3], y_train,	epochs=epochs, batch_size=batch_size)
	print('Testing--------------')
	loss, accuracy = model.evaluate([x_test1, x_test2, x_test3], y_test,)
	Y_pred = model.predict([x_test1, x_test2, x_test3])
	Y_pred = [np.argmax(y) for y in Y_pred] 
	Y_valid = [np.argmax(y) for y in y_test]





