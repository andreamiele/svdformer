import numpy as np
airplane = np.load("quickdraw_dataset/airplane.npy")
test = open("quickdraw_dataset/test.txt", "w")
train = open("quickdraw_dataset/train.txt", "w")

examples = []
minNbPts = -1


for i, example in enumerate(airplane):
  if i == 1000: break
  ex = example.reshape(28,28)
  x,y = np.where(ex>0)
  ex = np.vstack((x,y)).T.flatten()
  examples.append(ex)
  if minNbPts == -1 or ex.shape[0] < minNbPts:
    minNbPts = ex.shape[0]
print(minNbPts)
for i, example in enumerate(examples):
  l = example.shape[0]
  for _ in range(l - minNbPts):
    ind = np.random.randint(l)
    l -= 1
    example = np.delete(example, ind)
  np.save(f"quickdraw_dataset/airplane-{i}.npy", example)
  if i < 750:
    train.write(f"airplane-{i}.npy\n")
  else:
    test.write(f"airplane-{i}.npy\n")
test.close()
train.close()