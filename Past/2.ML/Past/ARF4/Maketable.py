import os, sys

infile = open("allARFs.314012.out","r")
outfile = open("Table.txt","w")
Switch = 0
for sLine in infile:
	if sLine.startswith("ARF"):
		outfile.write(sLine.strip()+"\t")
	if Switch == 1:
		outfile.write(sLine.strip()+"\t")
	if Switch == 2:
		outfile.write(sLine)
	
	if sLine.startswith("Accuracy"):
		Switch = 1
	elif sLine.startswith("Precision"):
		Switch = 1

	elif sLine.startswith("Sensitivity"):
		Switch = 2

	else:
		Switch =0


infile.close()
outfile.close()
