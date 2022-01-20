import os, glob, sys

outfile = open("Summary.txt","w")
outfile.write("ARFName\tAccuracy\tFPR\tFNR\n")
ARFNumbList = ["18","29","4","27","16","34","39","36","14","7","10","25"]

infile = open("Adjust_10_1.3573243.out","r")

Switch = "OFF" 
m = 0
for sLine in infile:
    if "Contigency_matrix from Sohyun" in sLine:
        Switch = "ON"
        k = 0 

    if Switch == "ON":
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
                    Switch == "OFF"

    if sLine.startswith("Accuracy"):
        ARFName = ARFNumbList[m]
        FPR = float(FP)/(float(FP)+float(TN))
        FNR = float(FN)/(float(FN)+float(TP))
        ACC = (float(TN)+float(TP))/(float(FP)+float(FN)+float(TN)+float(TP))
        #print(ACC)
        m+=1
        outfile.write(ARFName + "\t"+ str(ACC*100) + "\t"+ str(FPR*100) + "\t"+ str(FNR*100) +"\n")

infile.close()
outfile.close()
