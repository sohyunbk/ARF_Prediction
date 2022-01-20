import os, sys, random

wd = "/scratch/sb14489/1.ML_DAPSeq/13.AllCombi/1.MakeFASTA/2.Output/"


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

NonPeak, nNonPeak = MakdeDic(wd+"ARF4_bin125_NonPeakRemoveR.fa")
Border, nBorder = MakdeDic(wd+"ARF4_bin125_Border.fa")
Peak, nPeak = MakdeDic(wd+"ARF4_bin125_Peak.fa")

print(nNonPeak)
print(nBorder)
print(nPeak)

nTwicePeak = nPeak*2

#WriteFile(wd+"ARF4_bin125_NonPeak_SameNumbPeak.fa",NonPeak,nNonPeak,nPeak)
#WriteFile(wd+"ARF4_bin125_Border_SameNumbPeak.fa",Border,nBorder,nPeak)
WriteFile(wd+"ARF4_bin125_NonPeak_2XNumbPeak.fa",NonPeak,nNonPeak,nTwicePeak)
