from tokenize import group
from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand

def pid_cid_per_tansaction(transaction, S):
    pairs_dict = {}
    values = transaction.split(',')
    if int(values[3]) > 0 and (S == 'all' or str(values[7]) == S):
        pairs_dict[values[1]] = values[6]
    return [((key, pairs_dict[key]), None) for key in pairs_dict.keys()]

def gather_pairs_partitions(pairs):
    pairs_dict = {}
    for p in pairs:
        if p[0] not in pairs_dict.keys():
            pairs_dict[p[0]] = 1
        else:
            pairs_dict[p[0]] += 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

def gather_pairs(pairs):
	pairs_dict = {}
	for p in pairs[1]:
		if p[0] not in pairs_dict.keys():
			pairs_dict[p[0]] = 1
		else:
			pairs_dict[p[0]] += 1
	return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

def online_retailer_1(docs, S):
    productCustomer = (docs.flatMap(lambda x: pid_cid_per_tansaction(x, S)) # <-- MAP PHASE (R1)
        .reduceByKey(lambda x, y: x)            # <-- SHUFFLE + REDUCE PHASE (R1)
        .map(lambda x: x[0]))                   # <-- MAP PHASE (R2)
    return productCustomer

def online_retailer_partition_1(productCustomer):
    productPopularity = (productCustomer.mapPartitions(gather_pairs_partitions)    # <-- REDUCE PHASE (R2)
        .groupByKey()                              # <-- SHUFFLE + GROUPING
        .mapValues(lambda vals: sum(vals)))        # <-- MAP PHASE (R3)
    return productPopularity

def online_retailer_partition_2(productCustomer, K):
    productPopularity = (productCustomer.map(lambda x: (rand.randint(0,K-1), x))    # <-- MAP PHASE (R2)
        .groupByKey()                                   # <-- SHUFFLE + GROUPING
        .flatMap(gather_pairs)                          # <-- REDUCE PHASE (R2)
        .reduceByKey(lambda x, y: x + y))               # <-- REDUCE PHASE (R2)
    return productPopularity

def online_retailer_best_prods(productPopularity, H):
    bestProds = productPopularity.top(H, key=lambda x: x[1])
    return bestProds

def main():
# CHECKING NUMBER OF CMD LINE PARAMTERS
    assert len(sys.argv) == 5, "Usage: python G040HW1.py <K> <H> <S> <file_name>"

    # SPARK SETUP
    conf = SparkConf().setAppName('onlineRetailer').setMaster("local[*]")
    sc = SparkContext(conf=conf)

    # INPUT READING

    # 1. Read number of partitions
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # 2. Read number of results to show
    H = sys.argv[2]
    assert H.isdigit(), "H must be an integer"
    H = int(H)

    # 3. Read state for which to show the result
    S = sys.argv[3]
    assert isinstance(S, str), "S must be a string"
    S = str(S)

    # 4. Read input file and subdivide it into K random partitions
    data_path = sys.argv[4]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path,minPartitions=K).cache()
    rawData.repartition(numPartitions=K)

    # SETTING GLOBAL VARIABLES
    numtransictions = rawData.count()
    print("\nNumber of rows = ", numtransictions)

    # DIFFERENTS (P, C) PAIRS without using distinct
    productCustomer = online_retailer_1(rawData, S)
    print("Product-Customer Pairs =", productCustomer.count())

    # DIFFERENTS (P, Popularity) with mapPartitions
    productPopularity1 = online_retailer_partition_1(productCustomer)
    numprods = productPopularity1.count()
    print("\nNumber of distinct products in the transictions using mapPartitions =", numprods)

    # DIFFERENTS (P, Popularity) without mapPartitions
    productPopularity2 = online_retailer_partition_2(productCustomer, K)
    numprods = productPopularity2.count()
    print("\nNumber of distinct products in the transictions without using mapPartitions =", numprods)

    if H > 0:
        print("\nTop %d most popular products for productPopularity1" % (H))
        mostPopularProds = online_retailer_best_prods(productPopularity1, H)
        for i in range(len(mostPopularProds)):
            print("Product: %s Popularity: %s" %(mostPopularProds[i][0], mostPopularProds[i][1]), end ="; ")
        print()
    
    elif H == 0:
        print("\nOrdered products and popularity for productPopularity1")
        sorted_prods = productPopularity1.sortByKey().collect()
        for i in range(len(sorted_prods)):
            print("Product: %s Popularity: %s" %(sorted_prods[i][0], sorted_prods[i][1]), end ="; ")
        print()
        print("\nOrdered products and popularity for productPopularity2")
        sorted_prods = productPopularity2.sortByKey().collect()
        for i in range(len(sorted_prods)):
            print("Product: %s Popularity: %s" %(sorted_prods[i][0], sorted_prods[i][1]), end ="; ")
        print()
    
if __name__ == "__main__":
    main()
