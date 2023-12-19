import numpy as np
from os import listdir
from os.path import isfile, join

dataset_path = "quickdraw_dataset"

onlyfiles = [dataset_path+"/"+f for f in listdir(dataset_path) if isfile(join(dataset_path+"/", f)) and f.split(".")[-1] == "npy" and len(f.split("-")) == 1]
listfiles = [dataset_path+"/"+f for f in listdir(dataset_path) if isfile(join(dataset_path+"/", f)) and f.split(".")[-1] == "npy" and len(f.split("-")) == 1]
nbExamplesPerClass =100

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
    ex = example.reshape(256, 256)
    x, y = np.where(ex > 0)

    # Replicate the 2D points along the z-axis
    z_levels = np.arange(100)  # 28 levels along z-axis
    x_3d = np.tile(x, 100)  # Repeat x-coordinates for each z-level
    y_3d = np.tile(y, 100)  # Repeat y-coordinates for each z-level
    z_3d = np.repeat(z_levels, len(x))  # Repeat each z-level len(x) times

    ex_3d = np.vstack((x_3d, y_3d, z_3d)).T
    examplesFile.append(ex_3d)
    if minNbPts == -1 or ex_3d.shape[0] < minNbPts:
      minNbPts = ex_3d.shape[0]
  examples[file] = examplesFile
    
print(minNbPts)

for file in listfiles:
  for i, example in enumerate(examples[file]):
    if example.shape[0] > minNbPts:
        indices_to_keep = np.random.choice(example.shape[0], minNbPts, replace=False)
        example = example[indices_to_keep, :]

    a,b=file.split("/")
    c,d= b.split(".")
    n = f"{c}-{i}.npy"
    name = f"quickdraw_dataset/{c}-{i}.npy"
    np.save(name, example)
    if i < nbExamplesPerClass*1/4:
      train.write(n+"\n")
    else:
      test.write(n+"\n")
test.close()
train.close()
