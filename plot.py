import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join

filesDir = "/home/jab/EPFL/ML/svdformer_/multiClassOutput/out"

onlyfiles = [filesDir+"/"+f for f in listdir(filesDir) if isfile(join(filesDir, f)) and f.split(".")[-1] == "npy" and len(f.split("_")) == 7]

classMap = {}

for file in onlyfiles:
  className = file.split(".")[0].split("_")[-1]
  if className not in classMap:
    classMap[className] = []
  pointCloud = np.load(file)
  representationPoint = np.array([0,0], dtype=np.float64)
  numPts = 0
  for pt in pointCloud:
    numPts += 1
    representationPoint += pt[:-1]
  classMap[className].append(representationPoint/numPts)

classMap["uniformRandom"] = []
for i in range(2048):
  x = np.random.randint(28)
  y = np.random.randint(28)
  classMap["uniformRandom"].append(np.array([x,y]))

colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f']
i = 0
l = []
ln = []
for className, points in classMap.items():
  points = np.array(points)
  l.append(plt.scatter(points[:,0],points[:,1], c=colors[i]))
  ln.append(className)
  i += 1
plt.legend(tuple(l), tuple(ln))
plt.savefig("class-distribution.pdf")