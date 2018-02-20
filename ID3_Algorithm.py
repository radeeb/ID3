"""
CSC 635 Data Mining Assignment 1
Written by Radeeb Bashir(radeeb102) and JunHyeong Lee(junhyeong123)
Trace Folder Name: Radeeb102
2/19/2018
"""

from math import log
from collections import Counter
import pandas as pd
from random import shuffle

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

dataTable_testing = []  # Stores the 10% of the sample after the randomization.
dataLabelsCSV_copy = ['sepal length', 'sepal width', 'petal length', 'petal width']


# first one senio
def createDataTableCsv(data):
    '''
    Takes data in form of pandas dataFrame that was extracted from .csv file. Note that the labels are manually
    typed in. As long as the data from .csv can return list of dataTable and list of dataLabels, the algorithm should work.
    '''

    global dataTable_testing
    dataLabelsCSV = ['sepal length', 'sepal width', 'petal length', 'petal width']
    dataTable = training_data1.values.tolist()  # Use .values to get a numpy.array and then .tolist() to get a list.
    shuffle(dataTable)
    training_size = int(len(dataTable) * .9)  # 90 percent of the sample after the randomization.
    dataTable_training = list(dataTable[0:training_size])
    dataTable_testing = list(dataTable[training_size:])

    return dataTable_training, dataLabelsCSV


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
            chopped_row.extend(row[index+1:])  # setting the value that are AFTER the index
            subDataTable.append(chopped_row) # adding both of them, eventually returning reduced dataTable.
    return subDataTable

def createTree(dataTable, labels):
    '''
    Generates tree based on the dataTable given. dataTable shrinks over time which allows recursive algorithm.
    '''

    decision = [row[-1] for row in dataTable]

    dataCounter = Counter(decision) #to find the most common class
    majorityClass = dataCounter.most_common(1)[0][0] #set the value to majorityClass

    if decision.count(decision[0]) == len(decision):
        return decision[0]              # return when all of the decision in the dataTable is same
    if len(dataTable[0]) == 1:
        return majority_voting(decision)    # do the majority voting if there is only one remaining feature in dataTable
    root_attribute = NodeSelection(dataTable)
    root_attributeLabel = labels[root_attribute]
    tree = {root_attributeLabel: {}}
    del (labels[root_attribute])  # reducing the dataTable so the recursion will work
    attributeValuesAll = [row[root_attribute] for row in dataTable]
    attribute_values = set(attributeValuesAll)  # finding the unique values
    attribute_values.add(None)
    for value in attribute_values:
        if value == None:
            tree[root_attributeLabel][value] = majorityClass #set the None branch to majority class
        else:
            sub_labels = labels[:]
            tree[root_attributeLabel][value] = createTree(createSubtable(dataTable, root_attribute, value), sub_labels)
    return tree

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

    return Counter(decision).most_common()[0][1]

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
        information_gain = entropy_for_system - finalEntropy  # info gain calculation
        if information_gain > highest_info_gain:  
            highest_info_gain = information_gain  
            selected_attribute = i       # choosing the best information gain
    return selected_attribute  # returns an index of an attribute

def classify(inputTree, test_data):
    '''Takes as an argument a new sample and the decision tree represented as a dictionary.
    Returns the class classification of the new sample. Also takes care of missing and unexpected attribute values'''

    labels = list(test_data.keys())
    test_values = list(test_data.values())
    root_node = list(inputTree.keys())[0]
    branches = inputTree[root_node]
    labels_index = labels.index(root_node)
    key = test_values[labels_index]
    if key not in branches.keys():
        key = None
    value = branches[key]
    if isinstance(value, dict):
        classLabel = classify(value, test_data)
    else:
        classLabel = value
    return classLabel


def main():
    ####### FROM CSV FILE #######
    global dataLabelsCSV_copy  # Copy of the dataLable used for testing purpose.
    myDat, labels = createDataTableCsv(training_data1)  # for given training data
    mytree1 = createTree(myDat, labels)  # tree creation for the data from the web
    # print(mytree1)

    test_plant1 = {'sepal length': 4.6, 'sepal width': 3.4, 'petal length': 1.4, 'petal width': 0.2}
    test_plant2 = {'sepal width': 3.4, 'petal length': 1.4,
                   'petal width': 0.2}  # testing missing values (sepal length is missing)

    answerPlant1 = classify(mytree1, test_plant1)
    answerPlant2 = classify(mytree1, test_plant2)
    print("Test_plant1 is: " + answerPlant1)
    print("Test_plant2 is: " + answerPlant2)

    ##### testing random 10% of the sample #####

    for test_data in dataTable_testing:
        # print("correct classification: ", test_data[-1])
        temp_dict = dict(zip(dataLabelsCSV_copy, test_data[:-1]))
        print("is prediection correct? ", classify(mytree1, temp_dict) == test_data[-1])

    ######## FROM HW SAMPLE #######
    myDat1, labels1 = createDataTable(training_data2)  # for given training data
    mytree2 = createTree(myDat1, labels1)  # tree creation for the data from the assignment
    # print(mytree2)

    # print(answer)
    candidate_1 = {"level": "Junior", "lang": "Java", "tweets": "yes", "phd": "no"}  # True
    candidate_2 = {"level": "Junior", "lang": "Java", "tweets": "yes", "phd": "yes"}  # False

    # testing missing/incorrect values
    candidate_3 = {"level": "Intern", "lang": "Java", "tweets": "yes",
                   "phd": "yes"}  # incorrect value: Intern (Should return True)

    answer1 = classify(mytree2, candidate_1)
    answer2 = classify(mytree2, candidate_2)
    answer3 = classify(mytree2, candidate_3)
    print("Candidate_1:  ", answer1, "\nCandidate_2: ", answer2, "\nCandidate_3: ", answer3)


main()
