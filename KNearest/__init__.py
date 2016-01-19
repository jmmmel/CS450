import csv
import numpy as np

def getCarCsv():
    dataArray = []
    with open('car.csv') as csvFile:
      reader = csv.reader(csvFile)
      for row in reader:
          if row[0] == 'vhigh':
              row[0] = 3
          if row[0] == 'high':
              row[0] = 2
          if row[0] == 'med':
              row[0] = 1
          if row[0] == 'low':
              row[0] = 0
          if row[1] == 'vhigh':
              row[1] = 3
          if row[1] == 'high':
              row[1] = 2
          if row[1] == 'med':
              row[1] = 1
          if row[1] == 'low':
              row[1] = 0
          if row[2] == '5more':
              row[2] = 5
          if row[3] == 'more':
              row[3] = 6
          dataArray.append(row)
    return dataArray

class KNNeighbors:
    internalLearningSet = []
    def learn(self, learningData):
        standardized = np.asarray(learningData)
    def predict(self, predictData):
        return predictData

import random as rand
import numpy
from sklearn import datasets

#found at http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
def shuffle_in_unison(a, b):
    rng_state = numpy.random.get_state()
    numpy.random.shuffle(a)
    numpy.random.set_state(rng_state)
    numpy.random.shuffle(b)


car = getCarCsv()
rand.seed()
numpy.random.shuffle(car)

#Stores parts of iris dataset into comparable arrays
trainArray = car.data[:105], car.target[:105]
predictArray = car.target[105:]

#Sends data to the training then sends predicted object
blackBox = KNNeighbors()
blackBox.learn(trainArray)
predictedValues = blackBox.predict(predictArray)

count = 0
for index in range(len(predictArray)):
    if predictArray[index] == predictedValues[index]:
        count = count+1

print("The percent correct is: " + str(count / len(predictArray)))