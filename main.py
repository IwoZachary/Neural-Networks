

import random as rnd
from perceptron import  Perceptron
from adeline import  Adeline

data = [[1, 1], [1, 0], [0, 1], [0, 0]]
cls = [1, -1, -1, -1]

for i in range(0, 10):
    data.append([0-rnd.uniform(0, 0.3).__round__(3), 0-rnd.uniform(0, 0.3).__round__(3)])
    cls.append(-1)
for i in range(0, 10):
    data.append([0-rnd.uniform(0, 0.3).__round__(3), 1-rnd.uniform(0, 0.3).__round__(3)])
    cls.append(-1)
for i in range(0, 20):
    data.append([1-rnd.uniform(0, 0.3).__round__(3), 1-rnd.uniform(0, 0.3).__round__(3)])
    cls.append(1)

for i in range(0, 20):
    data.append([1-rnd.uniform(0, 0.3).__round__(3), 0-rnd.uniform(0, 0.3).__round__(3)])
    cls.append(-1)

'''
perc = Perceptron(0.01, data, cls)
perc.fit()
print(perc.predict([0,0,1]))
print(perc.predict([0,1,1]))
print(perc.predict([1,0,1]))
print(perc.predict([1,1,1]))
'''

adeline = Adeline(0.01, data, cls)
adeline.fit()
print(adeline.predict([0, 0, 1]))
print(adeline.predict([0, 1, 1]))
print(adeline.predict([1, 0, 1]))
print(adeline.predict([1, 1, 1]))
