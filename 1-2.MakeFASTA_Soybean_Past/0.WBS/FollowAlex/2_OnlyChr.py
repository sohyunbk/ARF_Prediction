#infile = open("allc_Gmax_AddOneCol.tsv","r")
#outfile = open("allc_Gmax_AddOneCol_OnlyChr.tsv","w")
infile = open("allc_Gmax.tsv","r")
outfile = open("allc_Gmax_OnlyChr.tsv","w")

Dic = {}
for sLine in infile:
	#sList = sLine.strip().split("\t")
    if (sLine.startswith("Gm")) and ("scaffold" not in sLine):
        outfile.write(sLine)
        sList = sLine.strip().split("\t")
        Dic.setdefault(sList[0],"")

infile.close()
outfile.close()
print(Dic.keys())
