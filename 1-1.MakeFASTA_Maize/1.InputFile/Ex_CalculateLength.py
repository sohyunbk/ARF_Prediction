infile = open("ARF10.GEM_events.narrowPeak","r")

nTotal = 0 
for sLine in infile:
    sList = sLine.strip().split("\t")
    nLength = int(sList[2]) - int(sList[1])
    nTotal += nLength

print(nTotal)
