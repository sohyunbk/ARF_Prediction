def DeleteChr(Infile,Outfile):
	infile = open(Infile,"r")
	outfile = open(Outfile,"w")

	for sLine in infile:
		outfile.write(sLine.replace("chr",""))

	infile.close()
	outfile.close()	

DeleteChr("UMR.txt","NonChr_UMR.txt")
