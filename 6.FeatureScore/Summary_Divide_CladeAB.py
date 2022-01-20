import statistics

def truncate(num, n):
    integer = int(num * (10**n))/(10**n)
    return float(integer)

def WriteMeanSD(List3):
    Average = statistics.mean(List3)
    Sd = statistics.stdev(List3)
    return str(Average)+"\t"+str(Sd)



def Main(ARFList, OutfileName):
    
    Dic = {}
    for ARF in ARFList:
        infile = open("ARF"+ARF+"_FeatureScore.txt","r")
        for sLine in infile:
            sList = sLine.strip().split("\t")
            Dic.setdefault(sList[0],[])
            Dic[sList[0]].append(float(sList[1]))
         

        infile.close()

    outfile = open(OutfileName,"w")
    outfile.write("ARF\tMean\tSd\n")
    for sKey in Dic.keys():
        List3 = Dic[sKey]
        #print(List3)
        outfile.write(sKey+"\t"+WriteMeanSD(List3)+"\n")            
    outfile.close()




ARFA = ["4","16","18","27","29","34"]
ARFB = ["7","10","14","25","36","39"]
Main(ARFA, "CladeA_Average_Sd.txt")
Main(ARFB, "CladeB_Average_Sd.txt")
