import os, glob, sys

def CountLine(Infile):
	infile = open(Infile,"r")
	j = 0 
	for k in infile:
		j+=1
	infile.close()
	return (j/2)


outfile = open("Soybean_CountNumber.txt","w")

ARFNumbList = ["4","16","18","27","29","34","7","10","14","25","36","39"]

nNonTotal = 0
nPeakTotal = 0
nRatioTotal = 0
i =0 
for ARF in ARFNumbList:
	outfile.write("ARF"+ARF+"\t")
	nPeak = CountLine("ZmARF"+ARF+"_Soybean_bin125_Peak.fa")
	nBorder = CountLine("ZmARF"+ARF+"_Soybean_bin125_Border.fa")
	nNonPeak = CountLine("ZmARF"+ARF+"_Soybean_bin125_NonPeakRemoveR.fa")
	nNonPeak = nBorder+nNonPeak
	nNonTotal += nNonPeak
	nPeakTotal += nPeak
	nRatioTotal += float(nNonPeak)/float(nPeak)
	outfile.write(str(nPeak)+"\t"+str(nNonPeak)+"\t1:"+str(float(nNonPeak)/float(nPeak))+"\n")
	i+=1
outfile.write("Average\t"+str(float(nPeakTotal)/float(i))+"\t"+str(float(nNonTotal)/float(i))+"\t1:"+str(float(nRatioTotal)/float(i))+"\n")

outfile.close()
