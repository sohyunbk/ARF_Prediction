import os, sys, glob
## Count Zea_mays genome Length


infile = open("Gmax_508_v4.0_OnlyChr.fa","r")

Count = 0
for sLine in infile:
	if not sLine.startswith(">"):
		nLine = len(sLine.strip())
		Count += nLine


print(Count)

