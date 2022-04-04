from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand

def lineToPC(line, K=-1, S='all'):
    data = line.split(',')
    if int(data[3])>0 and (S == 'all' or S == data[7]):
        if K == -1:
            return [(data[1],data[6])]
        else:
            return[(rand.randint(0,K-1),(data[1],data[6]))]
    return []

def removeDuplicate(pairs):
    pList = []
    for p in pairs[1]:
        if p not in pList:
            pList.append(p)
    return [(pairs[0],x) for x in pList]

        
def main():

    assert len(sys.argv) == 5, "Usage: python WordCountExample.py <K> <file_name>"

    conf = SparkConf().setAppName('G040HW1').setMaster("local[*]")
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")

    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    H = sys.argv[2]
    assert H.isdigit(), "H must be an integer"
    H = int(H)

    S = sys.argv[3]

    data_path = sys.argv[4]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path,minPartitions=K).cache()
    rawData.repartition(numPartitions=K)

    num = rawData.count()
    print("Number of lines = ", num)

    productCustomer = (rawData.flatMap(lambda x: lineToPC(x,S=S))
                    .groupByKey()
                    .flatMap(removeDuplicate))
    
    print("Porduct-Customer Pairs = ", productCustomer.count())

    productPopularity1 = (productCustomer
                        .groupByKey()
                        .mapValues(len))

    productPopularity2 = (productCustomer
                        .map(lambda x: (x[0],1))
                        .reduceByKey(lambda x,y: x+y))

    if H>0:
        pList = productPopularity1.takeOrdered(H, key=lambda x: -x[1])
        print(f"First {H} popular product(s): ", pList)
    elif H==0:
        pList1 = productPopularity1.collect()
        pList2 = productPopularity2.collect()
        pList1.sort()
        pList2.sort()
        print(f"All product(s) popularity (pP1): ", pList1)
        print(f"All product(s) popularity (pP2): ", pList2)


if __name__ == "__main__":
	main()