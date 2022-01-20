import os, sys, glob
## Count Zea_mays genome Length


infile = open("UMR.fa","r")

Count = 0
for sLine in infile:
	if not sLine.startswith(">"):
		nLine = len(sLine.strip())
		Count += nLine

print(Count)

infile.close()


infile2= open("UMR.bed","r")
Count2 = 0
for sLine2 in infile2:
	sList = sLine2.strip().split("\t")
	nLength = int(sList[2].strip()) - int(sList[1].strip())
	Count2 += nLength


print(Count2)

infile2.close()
