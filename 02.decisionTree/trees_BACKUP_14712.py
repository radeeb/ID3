
from math import log
import operator
from collections import Counter
import pandas as pd

training_data1 = pd.read_csv('iris.csv') #read the csv file into a dataframe

training_data2 = [
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

#similar method but for extracting from the dataFrame
def createDataSetCsv(data):
    dataLabels= ['sepal length', 'sepal width', 'petal length', 'petal width']

    dataTable = training_data1.values.tolist() #Use .values to get a numpy.array and then .tolist() to get a list.
    #dataTable.append(dataLabels)
    #print(dataTable[0])

    return dataTable, dataLabels

def entropy(dataSet):
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) /  len(dataSet)
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


def NodeSelection(dataTable):
    '''
    Calculates the entropy of the whole system first, then iterates o
    '''
    entropy_for_system = entropy(dataTable)
    highest_info_gain = 0.0;
    selected_attribute = 0
    attributesCount = len(dataTable[0]) - 1

    for i in range(attributesCount):
        attributes_all = [row[i] for row in dataTable]
        attributes = set(attributes_all)
        finalEntropy = 0.0
        for att in attributes:
            subTable = splitDataSet(dataTable, i, att)
            prob = len(subTable) / float(len(dataTable))
            finalEntropy += prob * entropy(subTable)


        information_gain = entropy_for_system - finalEntropy  # calculate the info gain; ie reduction in entropy

        if (information_gain > highest_info_gain):  # compare this to the best gain so far
            highest_info_gain = information_gain  # if better than current best, set to best
            selected_attribute = i
    return selected_attribute  # returns an integer


def majority_voting(decision):
    '''
    returns the most popular element from the list given
    '''

    return Counter(decision).most_common()[0][1] > (len(decision) // 2)


def createTree(dataTable, labels):
    decision = [row[-1] for row in dataTable]  #[y, n n n n n n n]
    if decision.count(decision[0]) == len(decision):
        return decision[0]              # return when all of the decision in the dataTable is same
    if len(dataTable[0]) == 1:
        return majority_voting(decision)    # return if there is only one remaining feature in dataTable

    #### Changed up to this part.

    bestFeat = NodeSelection(dataTable)
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataTable]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataTable, bestFeat, value), subLabels)
    return myTree




def classify(inputTree, test_data):
    labels = list(test_data.keys())
    test_values = list(test_data.values())
    root_node = list(inputTree.keys())[0]
    branches = inputTree[root_node]
    labels_index = labels.index(root_node)
    key = test_values[labels_index]
    value = branches[key]

    if isinstance(value, dict):
        classLabel = classify(value, test_data)
    else:
        classLabel = value
    return classLabel

if __name__ == "__main__":

<<<<<<< HEAD
    ######testing for csv data######
    #myDat, labels = createDataSetCsv(training_data1) #for given training data
    #mytree1 = createTree(myDat, labels)
    #print(mytree1)
    #answer = classify(mytree1, ['sepal length', 'sepal width', 'petal length', 'petal width'],
                      #[4.6, 3.4, 1.4, 0.2])

    ####testing for builtin data######
    myDat1, labels1 = createDataSet(training_data2)  # for given training data
    mytree1 = createTree(myDat1, labels1)
    print(mytree1)
    answer = classify(mytree1, ['level', 'lang', 'tweets', 'phd'],
                      ['Senior', 'Java', 'no', 'no'])
=======
    # FROM CSV
    myDat, labels = createDataSetCsv(training_data1) #for given training data
    mytree1 = createTree(myDat, labels)
    #print(mytree1)

    test_plant = {'sepal length':4.6, 'sepal width':3.4, 'petal length':1.4, 'petal width':0.2}
    answer = classify(mytree1, test_plant)
    print(answer)

    # FOR HW

    myDat1, labels1 = createDataSet(training_data2)  # for given training data
    mytree2 = createTree(myDat1, labels1)
    #print(mytree2)

>>>>>>> dce79768be85f545348177a4cf49ef93aac00630
    #print((answer)
    test1 = {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "no"}  #True
    test2 = {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "yes"} #False
    answer1 = classify(mytree2, test1)
    answer2 = classify(mytree2, test2)
    print(answer1, answer2)

    #print(createDataTable(training_data2))
    #print(createDataSetCsv(training_data1))

