from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras import backend as K
from sklearn.preprocessing import minmax_scale
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Flatten
import numpy as np
import h5py
import numpy as np
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
#import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import numpy as np;
import pandas as pd
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
#import pydot
from tensorflow.keras import datasets, layers, models
#import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.externals import joblib
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, LSTM
import keras
from keras import backend as K
from keras.models import Sequential


def ReadFastaFile(FileName):
    File = open(pwd+FileName,"r")
    Dic = {}
    count =0
    for sLine in File:
        if count%2 == 0:
            sKey = sLine.strip()
            #List.append(sKey)
        else:
            Dic[sKey] = sLine.strip()
        count +=1  
    File.close()   
    return Dic    

def one_hot_encode(Dic):
    nSample = len(Dic.keys())
    nSeqTotal = len(Dic[list(Dic.keys())[0]])
    arr_matrix = np.zeros((nSample, nSeqTotal,3))  # sequence data >1:30000_35000.txt
    Sample_count = 0
    for sSample in Dic.keys():       
        line = Dic[sSample]
        line_character_count = 0
        for c in line.strip():
            if c == "A":
                arr_matrix[Sample_count][line_character_count][0] = 1
            if c == "C":
                arr_matrix[Sample_count][line_character_count][1] = 1
            if c == "G":
                arr_matrix[Sample_count][line_character_count][2] = 1
            line_character_count += 1
        Sample_count +=1

    arr_matrix = arr_matrix.astype('int')
    return (arr_matrix)

def Label_Sample(nPeak,nNonpeak):
    nTotal = nPeak+nNonpeak
    y = []
    for i in range(0,nPeak):
        y.append(1)
    for j in range(nPeak,nTotal):
        y.append(0)
    y= pd.Series(y)
    y = y.astype('int')
    return y
    
def plotit(history, series, xlabel = 'Epoch', ylabel='', loc='lower right' ):
    for s in series:
        plt.plot(history.history[s], label=s)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=loc)
    return plt

def Main(nARF): 
    PeakDic = ReadFastaFile("ARF"+nARF+"_bin125_Peak.fa")
    arr_matrix1 = one_hot_encode(PeakDic)
    NonPeakDic = ReadFastaFile("ARF"+nARF+"_bin125_NonPeak_SameNumbPeak.fa")
    arr_matrix2 = one_hot_encode(NonPeakDic)

    X = np.concatenate((arr_matrix1,arr_matrix2))

    nPeak = len(PeakDic.keys())
    nNonpeak = len(NonPeakDic.keys())
    y = np.array(Label_Sample(nPeak,nNonpeak))
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)
    #print(X_test)
    print("Training set size:", len(y_train))
    print("Testing set size:", len(y_test))
    print("Input attributes:",X.shape[1])    
    model = Sequential()
    conv_layer = Conv1D(input_shape = X_train.shape[1:3],filters=320,kernel_size=32,padding='valid',activation='relu')
    model.add(conv_layer)
    #model.add(layers.Conv1D(32,3, activation='relu'))
    #model.Dense(32, activation='relu')
    model.add(MaxPooling1D(pool_size=50,strides=2,padding='valid'))
    model.summary()
    model.add(Flatten())
    model.add(Dense(1, activation = 'sigmoid'))
    sgd = SGD(lr = 0.1, momentum = 0.9, decay = 0, nesterov = False)
    model.compile(optimizer=sgd,
              loss='binary_crossentropy',
              metrics=['accuracy'])
              #metrics=['FalseNegatives'])
    
    #print(y_test[0:10])
    #print(np.array(y_test))
    history = model.fit(X_train, y_train, epochs=20, 
                    validation_data=(X_test, y_test))
    #print(y_test)
    #print(y_train)
    pred = model.predict_classes(X_test)
    print(pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    sensitivity = recall_score(y_test, pred)
    cm = metrics.confusion_matrix(y_test, pred)
    print(cm)
    print(accuracy)

pwd = "/scratch/sb14489/1-2.ML_NewTry/6.AllSimulations_Maize/2.Output/"
#ARFNumbList = ["18","29","4","27","16","34","39","36","14","7","10","25"]
#Outfile = open("Result_Onehotencoding_SequantialModel.txt","w")
ARFNumbList = ["Simulated"]
for nARF in ARFNumbList:
    Main(nARF)
