import os, glob, sys

outfile = open("Summary_ClassweightNone_maize.txt","w")
outfile.write("ARFName\tAccuracy\tFPR\tFNR\n")
for sFiles in glob.glob("Maize_Final_ClassWeighNone2.*.out"):
    infile = open(sFiles,"r")
    infile.readline()
    ARFName = infile.readline().strip()
    k =0 
    for sLine in infile:
        if sLine.startswith("[["):
            List = sLine.replace("[[","").split(" ")
            j =0
            for i in List:
                if i.isnumeric() == True and j ==0:
                    TN = i
                    j +=1 
                elif i.isnumeric() == True and j !=0:
                    FP = i
        if sLine.startswith(" [") and k ==0:
            List = sLine.replace("[","").split(" ")
            print(List)
            j =0
            k+=1
            for i in List:
                if i.isnumeric() == True and j ==0:
                    FN = i
                    j +=1
                elif i.isnumeric() == True and j !=0:
                    TP = i


    #print(TN)
    #print(FP)
    #print(FN)
    #print(TP)
    infile.close()
    FPR = float(FP)/(float(FP)+float(TN))
    FNR = float(FN)/(float(FN)+float(TP))
    ACC = (float(TN)+float(TP))/(float(FP)+float(FN)+float(TN)+float(TP))
    print(ACC)
    outfile.write(ARFName + "\t"+ str(ACC*100) + "\t"+ str(FPR*100) + "\t"+ str(FNR*100) +"\n")

outfile.close()
