import numpy as np
airplane = np.load("quickdraw_dataset/airplane.npy")
test = open("quickdraw_dataset/test.txt", "w")
train = open("quickdraw_dataset/train.txt", "w")
for i, example in enumerate(airplane):
  if i == 20: break
  np.save(f"airplane-{i}.npy", example)
  if i < 15:
    train.write(f"airplane-{i}.npy\n")
  else:
    test.write(f"airplane-{i}.npy\n")
test.close()
train.close()