import os, glob, sys

def CountLine(Infile):
	infile = open(Infile,"r")
	i = 0 
	for j in infile:
		i+=1
	return (i/2)


outfile = open("Soybean_CountNumber.txt","w")

ARFNumbList = ["4","16","18","27","29","34","7","10","14","25","36","39"]

for ARF in ARFNumbList:
	outfile.write("ARF"+ARF+"\t")
	nPeak = CountLine("ZmARF"+ARF+"_Soybean_bin125_Peak.fa")
	nBorder = CountLine("ZmARF"+ARF+"_Soybean_bin125_Border.fa")
	nNonPeak = CountLine("ZmARF"+ARF+"_Soybean_bin125_NonPeakRemoveR.fa")
	nNonPeak = nBorder+nNonPeak
	outfile.write(str(nPeak)+"\t"+str(nNonPeak)+"\t1:"+str(float(nNonPeak)/float(nPeak))+"\n")
