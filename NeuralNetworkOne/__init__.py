import numpy
import csv
import operator
import random as rand
from sklearn import datasets

class neurode:
    def __init__(self, rowCount):
        self.weights = []
        for value in range(1, rowCount + 1):
            self.weights.append(rand.uniform(-1,1))
    def printWeight(self):
        for value in range(0, len(self.weights)):
            print("Weight " + str(value) + ": " + str(self.weights[value])  )

    def checkInputs(self, inputs):
        print ("Input received:")
        for input in inputs:
            print(input)
        print ("\n")
        return 1

class network:
    def __init__(self, numberOfNodes, columnCount):
        self.learning = 1
        self.dataLength = numberOfNodes
        self.nodes = []
        for value in range(0, numberOfNodes ):
                self.nodes.append(neurode(columnCount))

    def printWeight(self):
        for value in range(0, len(self.nodes)):
            print("Node " + str(value) + ":" )
            self.nodes[value].printWeight()
            print("\n")

    def checkInputs(self, inputs):
        biasInput = [-1]
        biasInput.extend(inputs)
        outputs = []
        for value in range(0, len(self.nodes) ):
            outputs.append(self.nodes[value].checkInputs(biasInput))
        return outputs

def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

iris = datasets.load_iris()
rand.seed()
shuffle_in_unison(iris.target, iris.data)
iris.data = iris.data / iris.data.min(axis=0)
testData = numpy.hstack((iris.data,numpy.atleast_2d(iris.target).T))

newNeurode = network(10, 5)
x = 1;
for flower in iris.data:
    outputs = newNeurode.checkInputs(flower)
    print("Outputs for " + str(x) + ":" )
    for output in outputs:
        print(output)
    x += 1
    print("\n")

diabetesData = []
iFile = open('pimaIndians.data', 'r')
reader = csv.reader(iFile)
for row in reader:
    diabetesData.append(list(row))
numpyData = numpy.asarray(diabetesData)
numpyData = numpyData.astype(float)
numpyData = numpyData / (numpyData.max(axis=0) + numpy.spacing(0))

numpy.random.shuffle(numpyData)

pimaNetwork = network(5,len(numpyData[0]))

x = 1;
for person in numpyData:
    outputs = pimaNetwork.checkInputs(person)
    print("Outputs for " + str(x) + ":" )
    for output in outputs:
        print(output)
    x += 1
    print("\n")
