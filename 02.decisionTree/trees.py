
from math import log
import operator
from collections import Counter

training_data = [
({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'no'}, False),
({'level':'Senior', 'lang':'Java', 'tweets':'no', 'phd':'yes'}, False),
({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, True),
({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Junior', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, False),
({'level':'Mid', 'lang':'R', 'tweets':'yes', 'phd':'yes'}, True),
({'level':'Senior', 'lang':'Python', 'tweets':'no', 'phd':'no'}, False),
({'level':'Senior', 'lang':'R', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Junior', 'lang':'Python', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Senior', 'lang':'Python', 'tweets':'yes', 'phd':'yes'}, True),
({'level':'Mid', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, True),
({'level':'Mid', 'lang':'Java', 'tweets':'yes', 'phd':'no'}, True),
({'level':'Junior', 'lang':'Python', 'tweets':'no', 'phd':'yes'}, False)
]

def createDataSet(data):

    dataTable = []

    for tup in data:
        dataRow = list(tup[0].values())
        dataRow.append(tup[1])
        dataTable.append(dataRow)
        labels = list(tup[0].keys())
        
    return dataTable, labels


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)


        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        """
        print("feature : " + str(i))
        print("baseEntropy : "+str(baseEntropy))
        print("newEntropy : " + str(newEntropy))
        print("infoGain : " + str(infoGain))
        """
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


def majority_voting(decision):
    return Counter(decision).most_common()[0][1] > (len(decision) // 2)


def createTree(dataTable, labels):
    decision = [row[-1] for row in dataTable]  #[y, n n n n n n n]
    if decision.count(decision[0]) == len(decision):
        return decision[0]              # return when all of the decision in the dataTable is same
    if len(dataTable[0]) == 1:
        return majority_voting(decision)    # return if there is only one remaining feature in dataTable

    #### Changed up to this part.

    bestFeat = chooseBestFeatureToSplit(dataTable)
    bestFeatLabel = labels[bestFeat]

    #build a tree recursively
    myTree = {bestFeatLabel: {}}
    #print("myTree : "+labels[bestFeat])
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataTable]
    #print("featValues: "+str(featValues))
    uniqueVals = set(featValues)
    #print("uniqueVals: " + str(uniqueVals))
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        #print("subLabels"+str(subLabels))
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataTable, bestFeat, value), subLabels)
        #print("myTree : " + str(myTree))
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    #print("fistStr : "+firstStr)
    secondDict = inputTree[firstStr]
    #print("secondDict : " + str(secondDict))
    featIndex = featLabels.index(firstStr)
    #print("featIndex : " + str(featIndex))
    key = testVec[featIndex]
    #print("key : " + str(key))
    valueOfFeat = secondDict[key]
    #print("valueOfFeat : " + str(valueOfFeat))
    if isinstance(valueOfFeat, dict):
        #print("is instance: "+str(valueOfFeat))
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        #print("is Not instance: " + valueOfFeat)
        classLabel = valueOfFeat
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)



if __name__ == "__main__":

    myDat, labels = createDataSet(training_data)
    mytree = createTree(myDat, labels)
    print(mytree)

