infile = open("allc_Gmax.tsv","r")
outfile =open("allc_Gmax_AddOneCol.tsv","w")
for sLine in infile:
    sList = sLine.strip().split("\t")
    sNewLine = "\t".join(sList[0:2])+"\t"+str(int(sList[1])+1)+"\t"+"\t".join(sList[2:])+"\n"

    outfile.write(sNewLine)

infile.close()
outfile.close()
