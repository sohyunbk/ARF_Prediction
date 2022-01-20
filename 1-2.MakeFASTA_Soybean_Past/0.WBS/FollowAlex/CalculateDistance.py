infile = open("UMR.txt","r")

j = 0
for sLine in infile:
    sList = sLine.strip().split("\t")
    nLength = int(sList[2])-int(sList[1])
    if nLength > 300:
        j +=1

print(j)
