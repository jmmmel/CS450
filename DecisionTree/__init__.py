import numpy
import math
import operator
import random as rand

from collections import Counter
from collections import defaultdict
from sklearn import datasets


class node:
    def __init__(self,children, searchedArray):
        self.response = -1
        self.attributeValue = -1
        self.nodeList = []
        if len(children) == 0:
            return

        if all_same(children[:,len(children[0,:])-1]):
            self.response = children[0,len(children[0,:])-1]
            return
        self.children = children
        sortedArray = {}
        for attribute in searchedArray:
            sortedArray[attribute] = {}
            for value in range(int(numpy.min(self.children[:,attribute])), int(numpy.max(self.children[:,attribute]))):
                sortedArray[attribute][value] = 0
                for row in range(0,len(self.children[:,attribute])):
                    if self.children[row, attribute] == value:
                        sortedArray[attribute][value] += 1
        entropyMeasure = {}
        for index in sortedArray:
            entropyMeasure[index] = get_entropy(sortedArray[index], len(self.children))
        indexToFollow = min(entropyMeasure.items(), key=operator.itemgetter(1))[0]
        searchedArray = numpy.delete(searchedArray, numpy.where(searchedArray == indexToFollow))
        if len(searchedArray) == 0:
            self.response = self.children[0, len(self.children[0]) - 1]
            return
        self.attributeValue = indexToFollow
        for attributeValue in range(0, int(numpy.max(self.children[:,self.attributeValue]))):
            newList = self.children[numpy.logical_or.reduce([self.children[:,self.attributeValue] == attributeValue])]
            newNode = node(newList, searchedArray)
            if len(newList) == 0:
                newNode.set_Response(attributeValue)
            self.nodeList.append( newNode )

    def set_Response(self, val):
            self.response = val

    def get_value(self, searchValue):
        if self.response != -1:
            return  self.response
        attributeSearch = searchValue[int(self.attributeValue)]
        if attributeSearch >= len(self.nodeList):
            attributeSearch = len(self.nodeList) - 1
        return self.nodeList[int(attributeSearch)].get_value(searchValue)

    def display_value(self):
        myStr = "<node attributeSelected='" + str(self.attributeValue) + "' prediction='" + str(self.response) + "'>\n"
        myStr += "<children>\n"
        for child in self.nodeList:
            myStr += child.display_value()
        myStr += "</children>\n"
        myStr += "</node>\n"
        return  myStr



class myTree:
    def __init__(self, learnData):
        searchLength = len(learnData[0,:]) - 1
        searchArray = numpy.r_[0:searchLength]
        self.root = node(learnData, searchArray)

    def get_Type(self, searchVal):
        return self.root.get_value(searchVal)

    def Display_Tree(self):
        myStr = "<DecisionTree>\n"
        myStr += self.root.display_value()
        myStr += "</DecisionTree>"
        return myStr
    #found at http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

def all_same(items):
    return all(x == items[0] for x in items)

def get_entropy(values, size):
    total = 0.0
    for answerSize in values:
        prob = values[answerSize]/size
        if prob != 0:
            total += (prob * (math.log(prob, 2)))
    total *= -1
    return total

#gets iris dataset and shuffles it
iris = datasets.load_iris()
rand.seed()
shuffle_in_unison(iris.target, iris.data)
iris.data = iris.data - iris.data.min(axis=0)
iris.data = iris.data // ((iris.data.min(axis=0) + iris.data.max(axis=0)) / 10)
testData = numpy.hstack((iris.data,numpy.atleast_2d(iris.target).T))
decisionTree = myTree(testData[:30])
total = 0
correct = 0
for row in testData[30:]:
    guessType = decisionTree.get_Type(row)
    total += 1
    if guessType == row[len(row)-1]:
        correct += 1


percent = correct/total
print(decisionTree.Display_Tree())
print("This is accurate to the ", percent*100, "%")