 #!/usr/bin/env python3

# -------------------------------------------------------------------------
# BIG DISCLAIMER! ONLY TESTED FOR python3
# @author: Katherine Mejia-Guerra (mm2842@cornell.edu)
# Copyright (C) 2016 Katherine Mejia-Guerra
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# -------------------------------------------------------------------------

import os
import sys
import numpy as np
#import pandas as pd
import sqlalchemy
import logging
import time
from math import log
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from pandas import Series, DataFrame
#from random import *
import random
from itertools import product
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
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
#######################################################
#######################################################
## Compute_kmer_entropy < Make_stopwords
#######################################################
#######################################################

def compute_kmer_entropy(kmer):
    '''
    compute shannon entropy for each kmer
    :param kmer: string
    :return entropy: float
    '''
    prob = [float(kmer.count(c)) / len(kmer) for c in dict.fromkeys(list(kmer))]
    entropy = - sum([ p * log(p) / log(2.0) for p in prob ])
    return round(entropy, 2)


def make_stopwords(kmersize):
    '''
    write filtered out kmers
    :param kmersize: integer, 8
    :return stopwords: list of sorted low-complexity kmers
    '''
    kmersize_filter = {5:1.3, 6:1.3, 7:1.3, 8:1.3, 9:1.3, 10:1.3}
    limit_entropy = kmersize_filter.get(kmersize)
    print(limit_entropy)
    kmerSet = set()
    nucleotides = ["a", "c", "g", "t"]    
    kmerall = product(nucleotides, repeat=kmersize)
    for n in kmerall:
        #print(n)
        kmer = ''.join(n)
        #print(kmer)
        if compute_kmer_entropy(kmer) < limit_entropy:
            kmerSet.add(make_newtoken(kmer))
        else:
            continue
    stopwords = sorted(list(kmerSet))
    return stopwords

##############################################################
############################################################## 
####   CreateKmerSet < CreateNewtokenSet
##############################################################
##############################################################
def createKmerSet(kmersize):
    '''
    write all possible kmers
    :param kmersize: integer, 8
    :return uniq_kmers: list of sorted unique kmers
    '''
    kmerSet = set()
    nucleotides = ["a", "c", "g", "t"]
    kmerall = product(nucleotides, repeat=kmersize)
    for i in kmerall:
        kmer = ''.join(i)
        kmerSet.add(kmer)
    uniq_kmers = sorted(list(kmerSet))
    return uniq_kmers

def createNewtokenSet(kmersize):
    '''
    write all possible newtokens
    :param kmersize: integer, 8
    :return uniq_newtokens: list of sorted unique newtokens
    ''' 
    newtokenSet = set()
    uniq_kmers = createKmerSet(kmersize)
    for kmer in uniq_kmers:
        newtoken = make_newtoken(kmer)
        newtokenSet.add(newtoken)  
    uniq_newtokens = sorted(list(newtokenSet))
    return uniq_newtokens      

##########################################################
##########################################################
#### Make_newtoken < Write_ngrams
##########################################################
##########################################################
def make_newtoken(kmer):
    '''
    write a collapsed kmer and kmer reverse complementary as a newtoken
    :param kmer: string e.g., "AT"
    :return newtoken: string e.g., "atnta"
    :param kmer: string e.g., "TA"
    :return newtoken: string e.g., "atnta"
    '''
    kmer = str(kmer).lower()
    newtoken = "n".join(sorted([kmer,kmer.translate(str.maketrans('tagc', 'atcg'))[::-1]]))
    return newtoken

def write_ngrams(sequence):
    '''
    write a bag of newtokens of size n
    :param sequence: string e.g., "ATCG"
    :param (intern) kmerlength e.g., 2
    :return newtoken_string: string e.g., "atnta" "gatc" "cgcg" 
    '''
    seq = str(sequence).lower()
    finalstart = (len(seq)-kmerlength)+1
    allkmers = [seq[start:(start+kmerlength)] for start in range(0,finalstart)]
    tokens = [make_newtoken(kmer) for kmer in allkmers if len(kmer) == kmerlength and "n" not in kmer]
    newtoken_string = " ".join(tokens)
    return newtoken_string

##############################################################
##############################################################
##############################################################
def save_plot_prc(precision,recall, avg_prec, figure_file, name):
    '''
    make plot for precission recall
    :param precission: precission
    :param recall: recall
    :param avg_prec: avg_prec
    :param figure_file: figure_file
    :param name: name
    :return plot precission recall curve
    '''    
    plt.clf()
    title = 'Precision Recall Curve - double strand '+ name
    plt.title(title)
    plt.plot(recall, precision, label='Precission = %0.2f' % avg_prec )
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(figure_file)
            
def save_plot_roc(false_positive_rate, true_positive_rate, roc_auc, figure_file, name):
    '''
    make plot for roc_auc
    :param false_positive_rate: false_positive_rate
    :param true_positive_rate: true_positive_rate
    :param roc_auc: roc_auc
    :param figure_file: figure_file
    :param name: name
    :return roc_auc
    '''
    plt.clf()
    title = 'Receiver Operating Characteristic - double strand '+ name
    plt.title(title)
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(figure_file)    

#kmerlength=7
#print(len(createNewtokenSet(7)))
#print(len(make_stopwords(7)))
## Start!!!
################################################################
################################################################
def Bringfiles(InfileName):
    infile = open(InfileName,"r")
    Label=[]
    Seq=[]
    for sLine in infile:
        if sLine.startswith(">"):
            Label.append(sLine.strip())
        else:
            Seq.append(sLine.strip())

    return Label, Seq 

def CountNumber_Split_Training_Test(Seq,TrainingRatio):
    nNumberOfInput = len(Seq)
    nNumberTrain = int(nNumberOfInput*TrainingRatio)
    nNumberTest = nNumberOfInput-nNumberTrain
    ## 0 is Train 1 is Test. 
    #1) Generate Array
    Array = np.zeros(nNumberOfInput)
    #2) Define Random number
    RandomNumberforTest = random.sample(list(range(0,nNumberOfInput)), nNumberTest)
    #3) Put random Number in array
    Array[RandomNumberforTest] = 1
    #4) Return Test and Training Seq set.
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    RandomNumberforTrain = get_indexes(0,Array)
    #Seq_train = np.array(Seq)[RandomNumberforTrain]
    #Seq_test = np.array(Seq)[RandomNumberforTest]
    return RandomNumberforTest,RandomNumberforTrain

def CountNumber_Split_Training_Test_DownSampling(Seq,TrainingRatio,nNon,nPeak):
    
    nNumberOfInput = len(Seq)
    nNumberTrain = int(nNumberOfInput*TrainingRatio)
    nNumberTest = nNumberOfInput-nNumberTrain
    #######****************************************
    RandomNumberforTrain_NonPeak = random.sample(list(range(0,nNon)), int(nNon/2))
    RandomNumberforTrain_NonPeak_Sub = random.sample(RandomNumberforTrain_NonPeak,int(nPeak/2))
    RandomNumberforTrain_Peak = random.sample(list(range(nNon,nNon+nPeak)), int(nPeak/2))
    RandomNumberforTrain = RandomNumberforTrain_Peak+RandomNumberforTrain_NonPeak_Sub

    ## 0 is Train 1 is Test.
    #1) Generate Array
    Array = np.ones(nNumberOfInput)
    #3) Put random Number in array
    Array[RandomNumberforTrain_NonPeak+RandomNumberforTrain_Peak] = 0
    #4) Return Test and Training Seq set.
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    RandomNumberforTest = get_indexes(1,Array)
    #Seq_train = np.array(Seq)[RandomNumberforTrain]
    #Seq_test = np.array(Seq)[RandomNumberforTest]
    return RandomNumberforTest,RandomNumberforTrain


def Split_SameNumberwithPeak(PeakSeq,NonPeakSeq,BorderSeq):
    PeakNumber = len(PeakSeq)
    NonPeakNumber=len(NonPeakSeq)
    BorderNumber = len(BorderSeq)

    MinN = int(min([PeakNumber,NonPeakNumber,BorderNumber])/2)

    RN_Train_NonPeak,RN_Test_NonPeak = Sub_for_SplitSameNumberwithPeak(NonPeakNumber,MinN,0)
    RN_Train_Peak,RN_Test_Peak = Sub_for_SplitSameNumberwithPeak(PeakNumber,MinN,NonPeakNumber)
    RN_Train_Border,RN_Test_Border = Sub_for_SplitSameNumberwithPeak(BorderNumber,MinN,(NonPeakNumber+PeakNumber))
    
    RandomNumberforTest=RN_Test_NonPeak+RN_Test_Peak+RN_Test_Border
    RandomNumberforTrain=RN_Train_NonPeak+RN_Train_Peak+RN_Train_Border
    return RandomNumberforTest,RandomNumberforTrain

def Sub_for_SplitSameNumberwithPeak(TotalNumber,TrainNumber,PlusNumber):
    TestNumber = TotalNumber-TrainNumber
    #1) Generate Array
    Array = np.zeros(TotalNumber)
    #2) Define Random number
    RandomNumberforTest_B = random.sample(list(range(0,TotalNumber)), TestNumber)
    #3) Put random Number in array
    Array[RandomNumberforTest_B] = 1
    #4) Return Test and Training Seq set.
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    RandomNumberforTrain_B = get_indexes(0,Array)
    ##put to the below function
    RandomNumberforTest = Add_Number_inList(RandomNumberforTest_B,PlusNumber)
    RandomNumberforTrain = Add_Number_inList(RandomNumberforTrain_B,PlusNumber)
    return RandomNumberforTrain, RandomNumberforTest

def Add_Number_inList(List,N):
    NewList =[]
    for i in List:
        NewList.append(i+N)
    return NewList

def SubSampling(nNonPeak_New,nNonPeak_Ori,NonPeakLabel_Ori,NonPeakSeq_Ori):
    nSampleNumber= random.sample(list(range(0,nNonPeak_Ori)), nNonPeak_New)
    NonPeakLabel=[]
    NonPeakSeq=[]
    for i in nSampleNumber:
        NonPeakLabel.append(NonPeakLabel_Ori[i])
        NonPeakSeq.append(NonPeakSeq_Ori[i])
    print("Done with pepare Nonpeak!")
    return NonPeakLabel,NonPeakSeq

###############################################################
###############################################################
def Running_ML(PeakLabel,PeakSeq,BorderLabel,BorderSeq,NonPeakLabel,NonPeakSeq,nBorder_Choice):

    ## MakeYFirst 
    Y_NonPeak  = np.zeros(len(NonPeakSeq))
    Y_Peak = np.ones(len(PeakSeq))

    if nBorder_Choice==3:
        Y_Total  = np.concatenate([Y_NonPeak,Y_Peak])
        X_Total = np.concatenate([NonPeakSeq,PeakSeq])
    else:
        Y_Border = np.zeros(len(BorderSeq))
        Y_Border[0:len(BorderSeq)]=nBorder_Choice

        Y_Total  = np.concatenate([Y_NonPeak,Y_Peak,Y_Border])
        X_Total = np.concatenate([NonPeakSeq,PeakSeq,BorderSeq])
   ##############################################################
   ## Split it into test and  training set
   ##############################################################

    TrainingRatio = 0.7

    RN_Test,RN_Train = CountNumber_Split_Training_Test(X_Total,TrainingRatio)
    #RN_Test,RN_Train = CountNumber_Split_Training_Test_DownSampling(X_Total,TrainingRatio,len(Y_NonPeak),len(Y_Peak))
    #RN_Test,RN_Train = Split_SameNumberwithPeak(PeakSeq,NonPeakSeq,BorderSeq)

    X_train = np.array(X_Total)[RN_Train]
    X_test = np.array(X_Total)[RN_Test]

    Y_train = np.array(Y_Total)[RN_Train]
    Y_test = np.array(Y_Total)[RN_Test]

    print("I came to here!")
    ## Apply write ngrams using "map"
    Apply_ngrams  = lambda x: write_ngrams(x)
    Tokens_Train = list(map(Apply_ngrams, X_train))
    Tokens_Test = list(map(Apply_ngrams, X_test))
    #print(Gram_Train[0:3]) --> ['atagatcngatctat tagatcantgatcta agatcacngtgatct gatcacantgtgatc atcacatnatgtgat aatgtgantcacatt cacattcngaatgtg acattctnagaatgt aagaatgncattctt attcttgncaagaat ccaagaanttcttgg tccaagantcttgga atccaagncttggat aatccaanttggatt aaatccantggattt gaaatccnggatttc agaaatcngatttct atttctgncagaaat ccagaaantttctgg accagaanttctggt','...']

    #####################################
    ##  Building a vocabulary from tokens
    #####################################
    tmpvectorizer = TfidfVectorizer(min_df = 1 , max_df = 1.0, sublinear_tf=True,use_idf=True)
    X_TFIDF_ALL =  tmpvectorizer.fit_transform(all_tokens) #newtoken sequences to numeric index.
    vcblry = tmpvectorizer.get_feature_names()

    ## Doing Something for stop word
    print("removing %d low-complexity k-mers" % len(stpwrds))
    kmer_names = [x for x in vcblry if x not in stpwrds]
    feature_names = np.asarray(kmer_names) #key transformation to use the fancy index into the report
    #print(feature_names) ['aaaaccgncggtttt' 'aaaacctnaggtttt' 'aaaacgcngcgtttt' ...'ttggaaantttccaa' 'ttgtaaantttacaa' 'tttcaaantttgaaa']
    # All kinds of tokens
    print("The number of All tokens %d"%len(all_tokens))
    # Aftere Stop words 
    print("The number of  tokens %d"%len(kmer_names))

    #######################################
    #### feature of Tokens (['ggttcngaacc ttcgangacact ..',...]   --> vectorize 0,1
    #####################################
    print("Extracting features from the training data using TfidfVectorizer")
    vectorizer = TfidfVectorizer(min_df = 1 , max_df = 1.0, sublinear_tf=True,use_idf=True,vocabulary=kmer_names) #vectorizer for kmer frequencies

    # 1) Train
    X_TFIDF_DEV =  vectorizer.fit_transform(Tokens_Train).toarray()
    print("train_samples: %d, n_features: %d" % X_TFIDF_DEV.shape)
    print("NonPeak n_labels: %d Peak n_labels: %d Border n_labels: %d" % ((Y_train==0).sum(),(Y_train==1).sum(),(Y_train==2).sum()))
    # 2) Test
    X_TFIDF_test =  vectorizer.fit_transform(Tokens_Test).toarray()
    print("train_samples: %d, n_features: %d" % X_TFIDF_test.shape)
    print("NonPeak n_labels: %d Peak n_labels: %d Border n_labels: %d" % ((Y_test==0).sum(),(Y_test==1).sum(),(Y_test==2).sum()))
    #####################################
    ## fiting a LogisticRegression (LR) model to the training set
    ####################################
    #TFIDF_LR = LogisticRegression(random_state=0)
    #TFIDF_LR = OneVsRestClassifier(LinearSVC(random_state=0))
    TFIDF_LR = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)

    #TFIDF_LR = LogisticRegression(C=1.0, class_weight={0:0.1,1:0.9}, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l2', random_state=None, solver='liblinear', tol=0.0001,verbose=0, warm_start=False)
    #TFIDF_LR = RandomForestClassifier()
    #TFIDF_LR = RandomForestClassifier(max_depth=100)
    #print("SVC")
    #TFIDF_LR = SVC(kernel='linear', class_weight='balanced', probability=True)

    TFIDF_LR.fit(X_TFIDF_DEV, Y_train)
    print("Predicting labels for holdout set")
    LR_hold_TFIDF_pred = TFIDF_LR.predict(X_TFIDF_test) # y_pred
    #print(LR_hold_TFIDF_pred)
    outfile = open("Result_Check.txt","w")
    for b in range(len(LR_hold_TFIDF_pred)):
        print(b)
        outfile.write(str(LR_hold_TFIDF_pred[b])+"\t"+X_TFIDF_test[b]+"\n")
    print("Evaluating model")
    print(metrics.classification_report(Y_test, LR_hold_TFIDF_pred)) #y_true, y_pred
    print("Contigency_matrix from Sohyun")
    print(metrics.confusion_matrix(Y_test, LR_hold_TFIDF_pred,labels=[0,1,2])) #y_true, y_pred
    print(metrics.classification_report(Y_test, LR_hold_TFIDF_pred)) #y_true, y_pred
    print("Accuracy")
    print(accuracy_score(Y_test, LR_hold_TFIDF_pred))
    print("Precision")
    print(precision_score(Y_test, LR_hold_TFIDF_pred))
    print("Sensitivity")
    print(recall_score(Y_test, LR_hold_TFIDF_pred))
#########################################################################################
logging.basicConfig(level=logging.INFO, filename= "./Log.txt", filemode="a+",format="%(asctime)-15s %(levelname)-8s %(message)s")

filtered = True
full = False
kmerlength = 7
newtoken_size = 1+(kmerlength*2)
all_tokens = createNewtokenSet(kmerlength)
stpwrds = make_stopwords(kmerlength)
expected_tokens = len(all_tokens)


DataPath = "/scratch/sb14489/1.ML_DAPSeq/13.AllCombi/1.MakeFASTA/2.Output/"

def Main(nARF,sBinSize,Ratio):
    '''
    print("bin"+sBinSize)
    PeakLabel,PeakSeq = Bringfiles(DataPath+"ARF4_bin"+sBinSize+"_Peak.fa")
    NonPeakLabel,NonPeakSeq = Bringfiles(DataPath+"ARF4_bin"+sBinSize+"_NonPeak_2XNumbPeak.fa")
    BorderLabel,BorderSeq = Bringfiles(DataPath+"ARF4_bin"+sBinSize+"_Border_SameNumbPeak.fa")

    print("Peak: "+str(len(PeakLabel)))
    print("Non Peak: "+str(len(NonPeakLabel)))
    print("Border: "+str(len(BorderLabel)))

    ## 1) No Border 
    print("No Border")
    Running_ML(PeakLabel,PeakSeq,BorderLabel,BorderSeq,NonPeakLabel,NonPeakSeq,3)
    '''
    #### 2)3)4)
    print("bin"+sBinSize)
    if Ratio != "Original": 
        PeakLabel,PeakSeq = Bringfiles(DataPath+"ARF"+nARF+"_bin"+sBinSize+"_Peak.fa")
        NonPeakLabel,NonPeakSeq = Bringfiles(DataPath+"ARF"+nARF+"_bin125_NonPeak_Ratio1:"+str(Ratio)+".fa")
        BorderLabel,BorderSeq = Bringfiles(DataPath+"ARF"+nARF+"_bin125_Border_Ratio1:"+str(Ratio)+".fa")
    elif Ratio == "Original" :
        PeakLabel,PeakSeq = Bringfiles(DataPath+"ARF"+nARF+"_bin"+sBinSize+"_Peak.fa")
        NonPeakLabel,NonPeakSeq = Bringfiles(DataPath+"ARF"+nARF+"_bin125_NonPeak.fa")
        BorderLabel,BorderSeq = Bringfiles(DataPath+"ARF"+nARF+"_bin125_Border.fa")	
    #nPeak_Ori= len(PeakLabel)
    #nNonPeak_Ori= len(NonPeakLabel_Ori)
    #nBorder_Ori= len(BorderLabel)
    print("Peak: "+str(len(PeakLabel)))
    print("Non Peak: "+str(len(NonPeakLabel)))
    print("Border: "+str(len(BorderLabel)))

    
    ## 3) Nonpeak
    #print("ARF"+nARF)
    Running_ML(PeakLabel,PeakSeq,BorderLabel,BorderSeq,NonPeakLabel,NonPeakSeq,0)
    

#RatioList = [1,2,3,4,5,10,20,30,40,50,60,70,80,90]
#RatioList = [110]
#RatioList = ["Original"]
#ARFNumbList = ["10","13","14","16","18","25","27","29","34","35","36","39","4","7"]
ARFNumbList = ["4"]
for i in ARFNumbList:
    print("############################")
    print("ARF"+i)
    Main(i,"125","1")
