infile = open("UMR_Maize.txt","r")
outfile = open("UMR_Maize.bed","w")
for sLine in infile:
    NewLine = sLine.replace("chr","")
    sList = NewLine.strip().split("\t")
    New2 = "\t".join(sList[0:3])
    outfile.write(New2+"\n")

infile.close()
outfile.close()
