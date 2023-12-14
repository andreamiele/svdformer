import numpy as np
from os import listdir
from os.path import isfile, join
onlyfiles = ["/home/dehame/svdformer_/quickdraw_dataset/"+f for f in listdir("/home/dehame/svdformer_/quickdraw_dataset") if isfile(join("/home/dehame/svdformer_/quickdraw_dataset/", f)) and f.split(".")[-1] == "npy" and len(f.split("-")) == 1]

nbExamplesPerClass = 400

test = open("quickdraw_dataset/test.txt", "w")
train = open("quickdraw_dataset/train.txt", "w")
examples = {}
minNbPts = -1
for file in onlyfiles:
  fileNPY = np.load(file)
  print(file, fileNPY.shape)

  examplesFile = []
  for i, example in enumerate(fileNPY):
    if i == nbExamplesPerClass: break
    ex = example.reshape(28, 28)
    x, y = np.where(ex > 0)

    # Replicate the 2D points along the z-axis
    z_levels = np.arange(28)  # 28 levels along z-axis
    x_3d = np.tile(x, 28)  # Repeat x-coordinates for each z-level
    y_3d = np.tile(y, 28)  # Repeat y-coordinates for each z-level
    z_3d = np.repeat(z_levels, len(x))  # Repeat each z-level len(x) times

    ex_3d = np.vstack((x_3d, y_3d, z_3d)).T
    examplesFile.append(ex_3d)
    if minNbPts == -1 or ex_3d.shape[0] < minNbPts:
      minNbPts = ex_3d.shape[0]
  examples[file] = examplesFile
    
print(minNbPts)

for file in onlyfiles:
  for i, example in enumerate(examples[file]):
    l = example.shape[0]
    for _ in range(l - minNbPts):
      ind = np.random.randint(l)
      l -= 1
      example = np.delete(example, ind)
    name = file.split('.')[0] + f"-{i}.npy"
    np.save("quickdraw_dataset/" + name, example)
    if i < nbExamplesPerClass*3/4:
      train.write(name+"\n")
    else:
      test.write(name+"\n")
test.close()
train.close()
