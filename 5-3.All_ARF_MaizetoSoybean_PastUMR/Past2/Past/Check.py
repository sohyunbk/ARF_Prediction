infile = open("ARFTest_PredictionInfo.txt","r")

TN = 0
TP = 0
for sLine in infile:
    sList = sLine.strip().split("\t")
    Correct = sList[2]
    Pre = sList[3]
    if Correct == "0.0" and Pre =="0.0":
        TN+=1
    if Correct == "1.0" and Pre =="1.0":
        TP +=1        


print(TN)
print(TP)
