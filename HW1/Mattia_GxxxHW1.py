from sys import argv
from pyspark import SparkContext
from operator import add
from collections import Counter

def split_and_filter(S):
    def _(line):
        _, pid, _, qu, _, _, cid, country = line.split(b',')

        if int(qu) > 0 and (S.value == "all" or country.decode('ascii') == S.value):
            yield (pid, int(cid)), None

    return _

def count(kv):
    return Counter(k for k, v in kv).items()

def dump(pp, colon=True):
    colon = ':' if colon else ''

    for pid, pop in pp:
        print(f"Product{colon} {pid.decode('ascii')} Popularity{colon} {pop}; ", end="")

def main():
    K, H, S, dataset = argv[1:]
    K = int(K)
    H = int(H)

    sc = SparkContext()
    sc.setLogLevel("ERROR")
    S = sc.broadcast(S)

    rawData = sc.textFile(dataset, minPartitions=K, use_unicode=False) \
                .repartition(K)

    print(f"Number of rows = {rawData.count()}")

    productCustomer = rawData.flatMap(split_and_filter(S)) \
                             .reduceByKey(lambda X, Y: None) \
                             .keys() \
                             .partitionBy(K) # <-- this step could be removed
                                             #     if we only cared about pP2

    print(f"Product-Customer Pairs = {productCustomer.count()}")

    productPopularity1 = productCustomer.mapPartitions(count)

    productPopularity2 = productCustomer.map(lambda X: (X[0], 1)) \
                                        .reduceByKey(add)

    if H > 0:
        print(f"Top {H} Products and their Popularities")
        dump(productPopularity1.top(H, key=lambda X: X[1]), colon=False)
    else:
        print("productPopularity1: ")
        dump(productPopularity1.sortByKey().collect())
        print("\nproductPopularity2: ")
        dump(productPopularity2.sortByKey().collect())
        print()

if __name__ == "__main__":
    main()
