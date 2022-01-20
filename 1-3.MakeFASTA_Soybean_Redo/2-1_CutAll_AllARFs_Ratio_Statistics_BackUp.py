import os, sys, glob
import numpy as np


def MakePeakDic(InfileName):
    Dic = {}
    infile = open(InfileName,"r")
    for sLine in infile:
        sChr,nStart,nEnd = Split_Bed(sLine)
        Dic.setdefault(sChr,[])
        Dic[sChr].append(sLine)
    infile.close()
    return Dic

def Split_Bed(sLine):
    sList = sLine.strip().split("\t")
    sChr = sList[0]
    nStart = int(sList[1])
    nEnd = int(sList[2])
    return sChr,nStart,nEnd

def Split_Head(sLine):
    sList = sLine.strip().split(":")
    sChr = sList[0].replace(">","")
    nStart = int(sList[1].split("-")[0])
    nEnd = int(sList[1].split("-")[1])
    return sChr,nStart,nEnd


def Main(PeakFile,Cut_Length):
	PeakDic = MakePeakDic(pwd_IF+PeakFile)
	############################################
	infile = open(pwd_IF+"UMR.fa","r")
	PeakName = os.path.split(PeakFile)[1].split(".")[0]
	OutName = PeakName+"_bin"+str(Cut_Length)+"_"
	#Peak_Outfile = open(pwd_OF+OutName+"Peak.fa","w")
	#NonPeak_Outfile = open(pwd_OF+OutName+"NonPeak.fa","w")
	#Border_Outfile = open(pwd_OF+OutName+"Border.fa","w")
	##########################################
	Count_bp = 0
	Count =0
	for sLine in infile:
		if sLine.startswith(">"):
			sChr,nStart,nEnd = Split_Head(sLine)
			PeakLines = PeakDic[sChr]
			PeakList = []
			for PeakLine in PeakLines:
				PeakChr,PeaknS,PeaknE = Split_Bed(PeakLine)
				if nStart < PeaknS and PeaknE < nEnd:
					PeakList.append(PeakLine)
		else:
			sSeq = sLine.strip()
			nLength = len(sSeq)
			## Make np array of peaks  for overlap 
			#PeakArray_list = []
			Array = np.zeros(nLength)
			#print(PeakList)
			for sPeakLine in PeakList:
				PeakChr,PeaknS,PeaknE = Split_Bed(sPeakLine)
				#Array = np.zeros(nLength)
				Array[PeaknS-nStart:PeaknE-nStart] = 1
				Count_bp += ((PeaknE-nStart)-(PeaknS-nStart))
				Count+=1
				#PeakArray_list.append(Array)	
	print(PeakFile+"\t"+str(Count)+"\t"+str(Count_bp))
	infile.close()
	#Peak_Outfile.close()
	#NonPeak_Outfile.close()
	#Border_Outfile.close()	


pwd_IF ="/scratch/sb14489/1.ML_ARF/2.ML/1-3.MakeFASTA_Soybean_Redo/1.InputFile/"
pwd_OF = "/scratch/sb14489/1.ML_ARF/2.ML/1-3.MakeFASTA_Soybean_Redo/2.OutputFile/"

ARFNumbList = ["4","16","18","27","29","34","7","10","14","25","36","39"]

for i in ARFNumbList:
	Main("ZmARF"+i+"_Soybean.GEM_events.narrowPeak",125)
	

