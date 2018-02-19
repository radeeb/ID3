
from math import log
from collections import Counter
import pandas as pd

training_data1 = pd.read_csv('iris.csv') #read the csv file into a dataframe

training_data2 = [
({'level':'Intern', 'lang':'Java', 'tweets':'no', 'phd':'no'}, False),
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

#first one senio
def createDataTableCsv(data):
    '''
    Takes data in form of pandas dataFrame that was extracted from .csv file. Note that the labels are manually
    typed in. As long as the data from .csv can return list of dataTable and list of dataLabels, the algorithm should work.
    '''

    dataLabels= ['sepal length', 'sepal width', 'petal length', 'petal width']
    dataTable = training_data1.values.tolist() # Use .values to get a numpy.array and then .tolist() to get a list.
    dataTable.append(dataLabels)
    return dataTable, dataLabels

def createDataTable(data):
    '''
    Takes data in form of tuple that contains dictionary of attribute:value format and the classification.
    returns tuple of dataTable list and labels list so that it can be used for the algorithm.
    '''

    dataTable = []
    for tup in data:
        dataRow = list(tup[0].values())
        dataRow.append(tup[1])
        dataTable.append(dataRow)
        labels = list(tup[0].keys())
    return dataTable, labels

def createSubtable(dataTable, index, value):
    '''
    returns sub-dataTable based on the given dataTable, index of the attribute and its value.
    '''

    subDataTable = []
    for row in dataTable:
        if row[index] == value:
            chopped_row = row[:index]  # setting the values that are BEFORE the index
            chopped_row.extend(row[index + 1:])  # setting the value that are AFTER the index
            subDataTable.append(chopped_row) # adding both of them, eventually returning reduced dataTable.
    return subDataTable

def createTree(dataTable, labels):
    '''
    Generates tree based on the dataTable given. dataTable shrinks over time which allows recursive algorithm.
    '''

    decision = [row[-1] for row in dataTable]
    if decision.count(decision[0]) == len(decision):
        return decision[0]              # return when all of the decision in the dataTable is same
    if len(dataTable[0]) == 1:
        return majority_voting(decision)    # return if there is only one remaining feature in dataTable
    root_attribute = NodeSelection(dataTable)
    root_attributeLabel = labels[root_attribute]
    idTree = {root_attributeLabel: {}}
    del (labels[root_attribute])  # reducing the dataTable so the recursion will work
    attributeValuesAll = [row[root_attribute] for row in dataTable]
    attribute_values = set(attributeValuesAll)  # finding the unique values
    attribute_values.add('None')
    for value in attribute_values:
        if value == 'None':
            idTree[root_attributeLabel][value] = {'None': 'True'}
        else:
            sub_labels = labels[:]
            idTree[root_attributeLabel][value] = createTree(createSubtable(dataTable, root_attribute, value), sub_labels)
    return idTree

def entropy(dataTable):
    '''
    Calculates entropy based on a given table. Iterates through dataTable gets the number for each attribute.
    Then goes through calculation and eventually returns the entropy.
    '''

    decisionCounts = {}
    for row in dataTable:
        currentDecision = row[-1]
        if currentDecision not in decisionCounts.keys(): decisionCounts[currentDecision] = 0
        decisionCounts[currentDecision] += 1  # Common way of counting things in a form of dictionary
    entropy = 0.0
    for decision in decisionCounts:
        probability = float(decisionCounts[decision]) / len(dataTable)
        entropy -= probability * log(probability,2)
    return entropy

def majority_voting(decision):
    '''
    returns the most popular element from the list given
    '''

    return Counter(decision).most_common()[0][1] > (len(decision) // 2)

def NodeSelection(dataTable):
    '''
    Calculates the entropy of the whole system first, then calcaultes the entropy of the certain attribute
    which then is used to calculate the information gain. Index of attribute with highest
    information gain is returned.
    '''

    entropy_for_system = entropy(dataTable)
    highest_info_gain = 0.0
    selected_attribute = 0

    for i in range(len(dataTable[0]) - 1):
        attributes_all = [row[i] for row in dataTable] # adding all attributes
        attributes = set(attributes_all)  # store the unique attributes
        finalEntropy = 0.0
        for att in attributes:
            subTable = createSubtable(dataTable, i, att)
            probability = len(subTable) / float(len(dataTable))
            finalEntropy += probability * entropy(subTable)
        information_gain = entropy_for_system - finalEntropy  # calculate the info gain; ie reduction in entropy
        if information_gain > highest_info_gain:  # compare this to the best gain so far
            highest_info_gain = information_gain  # if better than current best, set to best
            selected_attribute = i
    return selected_attribute  # returns an integer

def classify(inputTree, test_data):
    '''gut

    '''

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

    # FROM CSV
    #myDat, labels = createDataTableCsv(training_data1) #for given training data
    #mytree1 = createTree(myDat, labels)
    #print(mytree1)

    #test_plant = {'sepal length':4.6, 'sepal width':3.4, 'petal length':1.4, 'petal width':0.2}
    #answer = classify(mytree1, test_plant)
    #print(answer)

    # FOR HW
    myDat1, labels1 = createDataTable(training_data2)  # for given training data
    mytree2 = createTree(myDat1, labels1)
    print(mytree2)

    #print((answer)
    test1 = {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "no"}  #True
    test2 = {"level" : "Junior","lang" : "Java","tweets" : "yes","phd" : "yes"} #False
    answer1 = classify(mytree2, test1)
    answer2 = classify(mytree2, test2)
    print(answer1, answer2)

    #print(createDataTable(training_data2))
    #print(createDataTableCsv(training_data1))
