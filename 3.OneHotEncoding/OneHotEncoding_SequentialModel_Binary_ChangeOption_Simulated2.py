import statistics
from multiprocessing import *
from sklearn.model_selection import StratifiedKFold, KFold
#from __future__ import print_function
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
    return plti

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

def WriteMeanSD(List3):
    Average = truncate(statistics.mean(List3)*100,2)
    Sd = truncate(statistics.stdev(List3)*100,2)
    return str(Average)+"+-"+str(Sd)

def GetMean(List3):
    Average = truncate(statistics.mean(List3),2)
    #print(Average)
    return Average

def CalculateScore(CM):
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    FPR = float(FP)/(float(FP)+float(TN))
    FNR = float(FN)/(float(FN)+float(TP))
    ACC = (float(TN)+float(TP))/(float(FP)+float(FN)+float(TN)+float(TP))
    return ACC,FPR,FNR

def SeparateList(List):
    ACC = []
    FPR =[]
    FNR = []
    for i in List:
        ACC.append(i[0][0])
        FPR.append(i[0][1])
        FNR.append(i[0][2])
    return ACC,FPR,FNR

def Model(X,y,train,test,ACC_list,manager,i,*args):
    X_train = X[train]
    X_test = X[test]
    y_train = y[train]
    y_test =  y[test]
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

    pred = model.predict_classes(X_test)
    print(pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    sensitivity = recall_score(y_test, pred)
    CM = metrics.confusion_matrix(y_test, pred)
    #print(cm)
    #print(accuracy)
    ACC,FPR,FNR = CalculateScore(CM)
    print("Print what I want")
    print(ACC)
    sub_1 = manager.list(ACC_list[i])
    sub_1.append([ACC,FPR,FNR])
    
    ACC_list[i] = sub_1

def Main(nARF): 
    PeakDic = ReadFastaFile("ARF"+nARF+"_bin125_Peak.fa")
    arr_matrix1 = one_hot_encode(PeakDic)
    NonPeakDic = ReadFastaFile("ARF"+nARF+"_bin125_NonPeak_SameNumbPeak.fa")
    arr_matrix2 = one_hot_encode(NonPeakDic)
    #BorderDic = ReadFastaFile("ARF"+nARF+"_bin125_Border_SameNumbPeak.fa")
    #arr_matrix3 = one_hot_encode(BorderDic) 
 
    X = np.concatenate((arr_matrix1,arr_matrix2))

    nPeak = len(PeakDic.keys())
    nNonpeak = len(NonPeakDic.keys())
    y = np.array(Label_Sample(nPeak,nNonpeak))


    #Stratified k-fold
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    ACC_list =[]

    #MultiProcessing
    Train_list = []
    Test_list =[]

    for train, test in skf.split(X, y): 
        Train_list.append(train)
        Test_list.append(test)
    #print(Train_list)
    manager=Manager()
    nCross = 5
    ACC_list = manager.list([[]]*nCross)

    #print(ACC_list)
    p =[]

    for m in range(nCross):
        train = Train_list[m]
        test = Train_list[m]
        p.append(Process(target=Model, args=(X,y,train,test,ACC_list,manager,m)))    
        
        p[m].start()
    for m in range(nCross):
        p[m].join()

    return ACC_list



    
if __name__ == '__main__':
    pwd = "/scratch/sb14489/1-2.ML_NewTry/6.AllSimulations_Maize/2.Output/"
    nRepeat=3
    
    Acc_re = []
    FPR_re = []
    FNR_re =[]
    for nRe in range(nRepeat):
        ACC_list = Main("Simulated")
        ACC_list, FPR_list, FNR_list = SeparateList(ACC_list)
        Acc_re.append(float(GetMean(ACC_list)))
        FPR_re.append(float(GetMean(FPR_list)))
        FNR_re.append(float(GetMean(FNR_list)))
    
    Line1 = "OneHotEncoding\t"+WriteMeanSD(Acc_re)+"\t"+WriteMeanSD(FPR_re)+"\t"+WriteMeanSD(FNR_re)+"\n"
    #print(Line1)
    outfile = open("Result.txt","w")
    outfile.write(Line1)
    outfile.close()
