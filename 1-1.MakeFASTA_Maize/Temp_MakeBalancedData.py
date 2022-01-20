import os, sys, random

wd = "/scratch/sb14489/1.ML_DAPSeq/12.200kb/1.MakeFASTA/ARF4/"

def MakdeDic(FileName):
    infile = open(FileName,"r")
    nLine = 0
    Dic = {}
    for sLine in infile:
        if sLine.startswith(">"):
            sKey = sLine
            nLine+=1
        else:
            Dic[sKey] = sLine
    infile.close()
    return Dic,nLine

def WriteFile(OutfileName,Dic,nOriginal,nNew):
    RandomN = random.sample(list(range(0,nOriginal)), nNew)
    outfile = open(OutfileName,"w")
    for i in RandomN :
         Header = Dic.keys()[i]
         Seq = Dic[Header]
         outfile.write(Header)
         outfile.write(Seq)

    outfile.close()

def Make_RatioData_withoutBorder(NonPeakFile,PeakFile,Ratio):
    if Ratio ==1:
        RatioName = "1to"+str(Ratio)
        NonPeakName = os.path.split(NonPeakFile)[1].split("_NonPeakRemoveR")[0]
        
        NonPeak, nNonPeak = MakdeDic(NonPeakFile)
        Peak, nPeak = MakdeDic(PeakFile)
        nNewNonPeak = nPeak

        WriteFile(wd+NonPeakName+"_NotConsiderBor_"+RatioName+".fa",NonPeak,nNonPeak,nNewNonPeak)

def Make_RatioData_withBorder(NonPeakFile,BorderFile,Ratio):
    NonPeakName = os.path.split(NonPeakFile)[1].split("_NonPeakRemoveR")[0]
    NonBorderName = os.path.split(BorderFile)[1].split(".")[0]
  
    NonPeak, nNonPeak = MakdeDic(wd+"ConsiderBor_"+Ratio+".fa")
    Border, nBorder = MakdeDic(wd+"Border.fa")
    Peak, nPeak = MakdeDic(wd+"Peak.fa")
    
    nNewBorder = (nPeak*nBorder)/nNonPeak
    nNewNonPeak = nPeak-nNewBorder

    WriteFile(wd+"NonPeak_Bal.fa",NonPeak,nNonPeak,nNewNonPeak)
    WriteFile(wd+"Border_Bal.fa",Border,nBorder,nNewBorder)




Ratio =1


Make_RatioData_withoutBorder(NonPeakFile,Ratio)
