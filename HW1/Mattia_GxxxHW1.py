from sys import argv
from pyspark import SparkContext
from operator import add

def split_and_filter(S):
    def _(line):
        _, pid, _, qu, _, _, cid, country = line.split(b',')

        if int(qu) > 0 and (S.value == "all" or country.decode('ascii') == S.value):
            yield (pid, int(cid)), None

    return _

def count(kv):
    out = {}

    for k, v in kv:
        if k in out:
            out[k] += 1
        else:
            out[k] = 1

    return out.items()

def main():
    K, H, S, dataset = argv[1:]
    K = int(K)
    H = int(H)

    sc = SparkContext()
    S = sc.broadcast(S)

    rawData = sc.textFile(dataset, minPartitions=K, use_unicode=False) \
                .repartition(K)

    productCustomer = rawData.flatMap(split_and_filter(S)) \
                             .reduceByKey(lambda X, Y: None) \
                             .keys() \
                             .partitionBy(K) # <-- this step could be removed
                                             #     if we only cared about pP2

    productPopularity1 = productCustomer.mapPartitions(count)

    productPopularity2 = productCustomer.map(lambda X: (X[0], 1)) \
                                        .reduceByKey(add)

    if H > 0:
        for pair in productPopularity1.top(H, key=lambda X: X[1]):
            print(pair)
    else:
        for pair in productPopularity1.collect():
            print(pair)

        print("---")

        for pair in productPopularity2.collect():
            print(pair)

if __name__ == "__main__":
    main()
