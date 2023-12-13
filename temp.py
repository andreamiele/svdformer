import numpy as np
airplane = np.load("quickdraw_dataset/airplane.npy")
test = open("quickdraw_dataset/test.txt", "w")
train = open("quickdraw_dataset/train.txt", "w")

examples = []
minNbPts = -1

# Process each 2D drawing
for i, example in enumerate(airplane):
    if i == 1000: break
    ex = example.reshape(28, 28)
    x, y = np.where(ex > 0)

    # Replicate the 2D points along the z-axis
    z_levels = np.arange(28)  # 28 levels along z-axis
    x_3d = np.tile(x, 28)  # Repeat x-coordinates for each z-level
    y_3d = np.tile(y, 28)  # Repeat y-coordinates for each z-level
    z_3d = np.repeat(z_levels, len(x))  # Repeat each z-level len(x) times

    ex_3d = np.vstack((x_3d, y_3d, z_3d)).T
    examples.append(ex_3d)
    if minNbPts == -1 or ex_3d.shape[0] < minNbPts:
        minNbPts = ex_3d.shape[0]

print(minNbPts)

train = open("train.txt", "w")
test = open("test.txt", "w")

# Process and save each 3D example
for i, example in enumerate(examples):
    # Ensure all examples have the same number of points
    if example.shape[0] > minNbPts:
        example = example[:minNbPts, :]
    
    np.save(f"airplane-{i}.npy", example)
    if i < 250:
        train.write(f"airplane-{i}.npy\n")
    else:
        test.write(f"airplane-{i}.npy\n")

test.close()
train.close()