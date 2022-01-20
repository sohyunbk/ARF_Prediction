infile = open("Gmax_508_v4.0_OnlyChr_tiles.condensed.classified.bed","r")
outfile = open("Gmax_508_v4.0_OnlyChr_tiles.condensed.classified_OnlyUMR.bed","w")
for sLine in infile:
    sList = sLine.strip().split("\t")
    if sList[3] == "UMR":
        outfile.write(sLine)

infile.close()
outfile.close()
        
