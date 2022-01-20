infile = open("UMR_Soybean.txt","r")
outfile = open("UMR_Soybean.bed","w")
for sLine in infile:
    sList = sLine.strip().split("\t")
    New2 = "\t".join(sList[0:3])
    #Gm01    241686  242063  ATAC_Soybean_10days_leaf.macs2.all_peak_1       170     .       6.85857 20.0189 17.002  279
    outfile.write(New2+"\tINfo\t100\t.\t10\t10\t10\t10\n")

infile.close()
outfile.close()
