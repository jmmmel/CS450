import numpy
import csv
import operator
import random as rand
import math
from sklearn import datasets

class neurode:
    def __init__(self, rowCount):
        self.weights = []
        for value in range(0, rowCount + 1):
            self.weights.append(rand.uniform(-1,1))

    def checkInputs(self, inputs):
        total = 0
        for index in range(0, len(inputs)):
            total += (self.weights[index] * inputs[index])
        calculatedVal = 1 / (1 + math.exp(-total))
        return calculatedVal


class network:
    def __init__(self, numberOfNodes, initialInputs, numberOfOutputs):
        self.learning = 1
        self.nodeSets = []
        nodes = []
        for x in range(0, numberOfNodes[0]):
            nodes.append(neurode(initialInputs))
        self.nodeSets.append(nodes)
        for value in range(1, len(numberOfNodes) ):
                nodes = []
                for index in range(0, numberOfNodes[value]):
                    nodes.append(neurode(numberOfNodes[value-1]))
                self.nodeSets.append(nodes)
        nodes = []
        for index in range(0, numberOfOutputs):
            nodes.append(neurode(numberOfNodes[len(numberOfNodes)-1]))
        self.nodeSets.append(nodes)


    def checkInputs(self, inputs):
        outputs = inputs
        for nodeIndex in range(0, len(self.nodeSets) ):
            biasInput = [-1]
            biasInput.extend(outputs)
            outputs = []
            for value in range(0, len(self.nodeSets[nodeIndex]) ):
                outputs.append(self.nodeSets[nodeIndex][value].checkInputs(biasInput))
        return outputs


def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

iris = datasets.load_iris()
rand.seed()
shuffle_in_unison(iris.target, iris.data)
iris.data = iris.data / iris.data.max(axis=0)
testData = numpy.hstack((iris.data,numpy.atleast_2d(iris.target).T))

newNeurode = network([2,3,5], len(testData[0,:]) - 1 , 3)

total = 0
correct = 0
for flower in testData:
    total += 1
    output = newNeurode.checkInputs(flower[0:4])
    maxIndex = 0
    for index in range(1, len(output)):
        if output[index] > output[maxIndex]:
            maxIndex = index
    if flower[4] == maxIndex:
        correct += 1
flowerCorrectness = correct/total
print("This is ", flowerCorrectness, "% correct\n")

diabetesData = []
iFile = open('pimaIndians.data', 'r')
reader = csv.reader(iFile)
for row in reader:
    diabetesData.append(list(row))

numpyData = numpy.asarray(diabetesData)
numpyData = numpyData.astype(float)
numpyData = numpyData / (numpyData.max(axis=0) + numpy.spacing(0))
numpy.random.shuffle(numpyData)

pimaNetwork = network([2,3,4], len(numpyData[0,:]) - 1, 2)

total = 0
correct = 0
for person in numpyData:
    total += 1
    output = pimaNetwork.checkInputs(person[0:7])
    maxIndex = 0
    for index in range(1, len(output)):
        if output[index] > output[maxIndex]:
            maxIndex = index
    if person[len(person)-1] == maxIndex:
        correct += 1
flowerCorrectness = correct/total
print("This is ", flowerCorrectness, "% correct\n")

