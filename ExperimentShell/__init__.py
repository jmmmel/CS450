class HardCoded:
    def train(self, valuesArray):
        return valuesArray
    def predict(self, valuesArray):
        total = 150-105
        a = []
        for index in range(total):
            a.append(0)
        return a

import random as rand
from sklearn import datasets
iris = datasets.load_iris()
rand.seed()

for i in range(len(iris)):
    swap = iris[i]
    swapIndex = rand.randint(0,len(iris)-1)
    iris[i] = iris[swapIndex]
    iris[swapIndex] = swap

trainArray = iris[:105]
predictArray = iris[105:]

HardCoded.train(trainArray)
predictedValues = HardCoded.predict(predictArray)

count = 0
for index in range(len(predictArray)):
    if predictArray[index] == predictedValues[index]:
        count = count+1

print("The percent correct is:" + (count / len(predictArray)))