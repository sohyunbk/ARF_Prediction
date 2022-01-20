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
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

def ReadFastaFile(FileName):
	File = open(FileName,"r")
	Dic = {}
	count =0
	for sLine in File:
		if count%2 == 0:
			sKey = sLine.strip()
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

