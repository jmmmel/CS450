import random as rand
import numpy
from sklearn import datasets

class HardCoded:
    def train(self, valuesArray):
        return valuesArray
    def predict(self, valuesArray):
        total = 150-105
        a = []
        for index in range(total):
            a.append(0)
        return a
#found at http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)

#gets iris dataset and shuffles it
iris = datasets.load_iris()
rand.seed()
shuffle_in_unison(iris.target, iris.data)

#Stores parts of iris dataset into comparable arrays
trainArray = iris.data[:105], iris.target[:105]
predictArray = iris.target[105:]

#Sends data to the training then sends predicted object
blackBox = HardCoded()
blackBox.train(trainArray)
predictedValues = blackBox.predict(predictArray)

count = 0
for index in range(len(predictArray)):
    if predictArray[index] == predictedValues[index]:
        count = count+1

print("The percent correct is: " + str(count / len(predictArray)))