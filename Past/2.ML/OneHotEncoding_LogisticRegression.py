from sklearn.feature_extraction.text import CountVectorizer
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
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import random
from itertools import product
from sklearn import metrics
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

def ReadFastaFile(FileName):
    File = open("../1.MakeFASTA/2.Output/"+FileName,"r")
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
    NonPeakDic = ReadFastaFile("ARF"+nARF+"_bin125_NonPeak.fa")
    arr_matrix2 = one_hot_encode(NonPeakDic)
    
    X = np.concatenate((arr_matrix1,arr_matrix2))

    nPeak = len(PeakDic.keys())
    nNonpeak = len(NonPeakDic.keys())
    y = Label_Sample(nPeak,nNonpeak)
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2)
    #print(X_test)
    print("Training set size:", len(y_train))
    print("Testing set size:", len(y_test))
    print("Input attributes:",X.shape[1])     
    ##### Logistic
    TFIDF_LR = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)
    #TFIDF_LR = LogisticRegression()
    #X_train_flattened = [e.ravel() for e in X_train]
    #X_test_flattened = [e.ravel() for e in X_test]
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    X_train_counts.shape
    X_test_counts = count_vect.fit_transform(X_test)
    TFIDF_LR.fit(X_train_counts, y_train)	
    LR_hold_TFIDF_pred = TFIDF_LR.predict(X_test_counts)
    #print(metrics.classification_report(y_test, LR_hold_TFIDF_pred)) #y_true, y_pred
    accuracy = accuracy_score(y_test, LR_hold_TFIDF_pred)
    precision = precision_score(y_test, LR_hold_TFIDF_pred)
    sensitivity = recall_score(y_test, LR_hold_TFIDF_pred)
    return accuracy, precision, sensitivity

ARFNumbList = ["18","29","4","27","16","34","39","36","14","7","10","25"]
Outfile = open("Result_Onehotencoding_LogisticRegression","w")
Outfile.write("ARF\tValue\tKind\n")
for ARF in ARFNumbList:
    start = time.time()
    accuracy, precision, sensitivity = Main(ARF)
    print(time.time() - start)
    Outfile.write("ARF"+ARF+"\t"+str(accuracy)+"\tAccuracy\n")
    Outfile.write("ARF"+ARF+"\t"+str(precision)+"\tPrecision\n")
    Outfile.write("ARF"+ARF+"\t"+str(sensitivity)+"\tSensitivity\n")
Outfile.close()

