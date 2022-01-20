import os, sys, glob
## Count Zea_mays genome Length


infile = open("Zea_mays.AGPv4.dna.toplevel_OnlyChr.fa","r")

Count = 0
for sLine in infile:
	if not sLine.startswith(">"):
		nLine = len(sLine.strip())
		Count += nLine


print(Count)

