
"""ID3 Decision tree induction algorithm
Author: Radeeb Bashir, JunHyeong Lee
"""
from math import log

def createData():
    inputData = [['senior', 'java', 'no', 'no', 'false'],
               ['senior', 'java', 'no', 'yes', 'false'],
               ['mid', 'python', 'no', 'no', 'true'],
               ['junior', 'python', 'no', 'no', 'true'],
               ['junior', 'R', 'yes', 'no', 'true'],
               ['junior', 'R', 'yes', 'yes', 'false'],
               ['mid', 'R', 'yes', 'yes', 'true'],
               ['senior', 'python', 'no', 'no', 'false'],
               ['senior', 'R', 'yes', 'no', 'true'],
               ['junior', 'python', 'yes', 'no', 'true'],
               ['senior', 'python', 'yes', 'yes', 'true'],
               ['mid', 'python', 'no', 'yes', 'true'],
               ['mid', 'java', 'yes', 'no', 'true'],
               ['junior', 'python', 'no', 'yes', 'false']]

    labels = ['level', 'language', 'tweets', 'phd']
    # change to discrete values
    return inputData, labels
