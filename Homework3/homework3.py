'''
@Author: Shihan Ran
@Date: 2019-11-07 12:54:16
@LastEditors: Shihan Ran
@LastEditTime: 2019-11-08 20:06:39
@Email: rshcaroline@gmail.com
@Software: VSCode
@License: Copyright(C), UCSD
@Description: 
'''

import gzip
import random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        yield l.strip().split(',')


def readBaseline():
    """ 
    Would-read baseline: just rank which books are popular and which are not, and return '1' if a book is among the top-ranked
    """
    bookCount = defaultdict(int)
    totalRead = 0

    for user, book, _ in readCSV("./data/train_Interactions.csv.gz"):
        bookCount[book] += 1
        totalRead += 1

    mostPopular = [(bookCount[x], x) for x in bookCount]
    mostPopular.sort()
    mostPopular.reverse()

    return1 = set()
    count = 0
    for ic, i in mostPopular:
        count += ic
        return1.add(i)
        if count > totalRead/1.9:
            break

    predictions = open("predictions_Read.txt", 'w')
    for l in open("./data/pairs_Read.txt"):
        if l.startswith("userID"):
            # header
            predictions.write(l)
            continue
        u, b = l.strip().split('-')
        if b in return1:
            predictions.write(u + '-' + b + ",1\n")
        else:
            predictions.write(u + '-' + b + ",0\n")

    predictions.close()


def ratingBaseline():
    """
    Rating baseline: compute averages for each user, or return the global average if we've never seen the user before
    """
    allRatings = []
    userRatings = defaultdict(list)

    for user, book, r in readCSV("./data/train_Interactions.csv.gz"):
        r = int(r)
        allRatings.append(r)
        userRatings[user].append(r)

    globalAverage = sum(allRatings) / len(allRatings)
    userAverage = {}
    for u in userRatings:
        userAverage[u] = sum(userRatings[u]) / len(userRatings[u])

    predictions = open("predictions_Rating.txt", 'w')
    for l in open("./data/pairs_Rating.txt"):
        if l.startswith("userID"):
            # header
            predictions.write(l)
            continue
        u, b = l.strip().split('-')
        if u in userAverage:
            predictions.write(u + '-' + b + ',' + str(userAverage[u]) + '\n')
        else:
            predictions.write(u + '-' + b + ',' + str(globalAverage) + '\n')

    predictions.close()


def main():
    readBaseline()


if __name__ == "__main__":
    main()