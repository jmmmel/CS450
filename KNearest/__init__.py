import random as rand
import numpy
import math
import random as rand
from sklearn import datasets

class KNNeighbors:
    internalLearningSet = numpy.array
    def learn(self, learningData):
        self.internalLearningSet  = learningData
    def predict(self, predictData, nearestK):
        distanceArray = [];
        for row in range(0, len(self.internalLearningSet[:,0])):
            distance = 0;
            for index in range(0, len(predictData)-1):
                distance += (predictData[index]-self.internalLearningSet[row,index])**2
            distance = math.sqrt(distance)
            distanceArray.append(distance)
        return self.internalLearningSet[distanceArray.index(min(distanceArray)),len(self.internalLearningSet[0])-1]

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

iris.data = iris.data / iris.data.max(axis=0)
testData = numpy.hstack((iris.data,numpy.atleast_2d(iris.target).T))
#Sends data to the training then sends predicted object
blackBox = KNNeighbors()
blackBox.learn(testData[:30])
findData = testData[30:]
total = 0
correct = 0
for index in range(0, len(findData[:,0])):
    predicted = blackBox.predict(findData[index],1)
    total += 1
    if predicted == findData[index, len(findData[index])-1]:
        correct += 1

print ("Accuracy: ", (correct/total) )