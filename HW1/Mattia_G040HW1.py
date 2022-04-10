from sys import argv, stderr
from os import path
from pyspark import SparkConf, SparkContext
from operator import add
from collections import Counter

def split_and_filter(S):
    """
    Given `Broadcast` object `S` of type `bytes`, returns a generator.
    This generator must be given a dataset line; it parses the line as a
    transaction record, and may yield a single tuple `(pid, cid)`, where
    `pid` is the product ID of type `bytes` and `cid` is the customer ID
    of type `int`.  This tuple is yielded only if the recorded quantity
    is positive and the value of `S` is either a. equal to the recorded
    country or b. equal to `b"all"`, which acts as a wildcard.
    """
    def _(line):
        _, pid, _, qu, _, _, cid, country = line.split(b',')

        if int(qu) > 0 and (S.value == b"all" or country == S.value):
            yield (pid, int(cid)), None

    return _

def count(kv):
    """
    Given a list of pairs `kv`, returns a list of tuples.
    Each element in the output list is of the form `(k, v)`.  `k` is unique,
    and `c`, of type `int`, is the number of pairs in the original list with
    `k` as first element.
    """
    return Counter(k for k, v in kv).items()

def dump(pp, colon=True):
    """
    Given a list of pairs `pp`, prints it out in an idiosyncratic format.
    Each pair must be of the form `(pid, pop)`, where `pid` is a product ID of
    type `bytes` and `pop` its popularity, stored as an `int`.  If `colon` is
    `True`, a variant format is used.
    """
    colon = ':' if colon else ''

    for pid, pop in pp:
        print(f"Product{colon} {pid.decode('ascii')} Popularity{colon} {pop}; ", end="")

def main(K, H, S, dataset):
    K = int(K)
    H = int(H)

    conf = SparkConf().setAppName("ProductPopularity").setMaster("local[*]")
    sc = SparkContext.getOrCreate(conf=conf)
    S = sc.broadcast(S.encode())

    rawData = (
        sc.textFile(dataset, minPartitions=K, use_unicode=False)
          .repartition(K)
    )

    print(f"Number of rows = {rawData.count()}")

    # Note that the call to keys() effectively makes the partitioning random
    # with respect to pid, as the original partitioning was induced by
    # (pid, cid) through a hash function; for any sufficiently good hash there
    # ought not be any correlation between either partitioning.
    productCustomer = (
        rawData.flatMap(split_and_filter(S))      # R1 MAP
               .reduceByKey(lambda X, Y: None, K) # R1 SHUFFLE + REDUCE
               .keys()                            # R2 MAP
               .cache()
    )

    print(f"Product-Customer Pairs = {productCustomer.count()}")

    # As per the note above, in the R3 REDUCE below we may regard the
    # key-value pairs as being randomly distributed among the K partitions;
    # thus, despite the fact that the number of pairs with the same pid may
    # be O(N) (since (pid, cid) pairs can be assumed to be O(1)), we may
    # safely assume that, with high probability, the R3 REDUCE will only have
    # to deal with O(N/K) items.

    productPopularity1 = (
        productCustomer.mapPartitions(count) # R2 REDUCE
                       .groupByKey(K)        # R3 SHUFFLE
                       .mapValues(sum)       # R3 REDUCE
    )

    # Note that reduceByKey reduces inside each partition first, so the
    # argument above holds for reduceByKey as well.
    # A call to reduceByKey is in fact equivalent to a call to mapPartitions
    # followed by a shuffle (partitionBy) and another call to mapPartitions;
    # indeed this makes the following essentially call-by-call equivalent to
    # the above "hand-crafted" version, as mapValues is ultimately implemented
    # through mapPartitions{ByIndex} and groupByKey through partitionBy.

    productPopularity2 = (
        productCustomer.map(lambda X: (X[0], 1)) # R2 MAP (cont.)
                       .reduceByKey(add, K)      # R2 SHUFFLE + REDUCE
    )

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
    if len(argv) != 5:
        print(f"Usage: {argv[0]} K H S dataset", file=stderr)
        exit()

    K, H, S, dataset = argv[1:]

    try:
        K, H = int(K), int(H)
    except ValueError:
        print(f"K and H must be integers, not {K!r} and {H!r}", file=stderr)
        exit(-1)

    if not path.isfile(dataset):
        print(f"file not found: {dataset}", file=stderr)
        exit(-1)

    main(*argv[1:])
