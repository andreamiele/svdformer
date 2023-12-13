import numpy as np
airplane = np.load("quickdraw_dataset/airplane.npy")
fork = np.load("quickdraw_dataset/fork.npy")
bench = np.load("quickdraw_dataset/bench.npy")
test = open("quickdraw_dataset/test.txt", "w")
train = open("quickdraw_dataset/train.txt", "w")

examples = []
minNbPts = -1

# Process each 2D drawing
for i, example in enumerate(airplane):
    if i == 400: break
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

examples2 = []
minNbPts2 = -1

# Process each 2D drawing
for i, example in enumerate(fork):
    if i == 400: break
    ex = example.reshape(28, 28)
    x, y = np.where(ex > 0)

    # Replicate the 2D points along the z-axis
    z_levels = np.arange(28)  # 28 levels along z-axis
    x_3d = np.tile(x, 28)  # Repeat x-coordinates for each z-level
    y_3d = np.tile(y, 28)  # Repeat y-coordinates for each z-level
    z_3d = np.repeat(z_levels, len(x))  # Repeat each z-level len(x) times

    ex_3d = np.vstack((x_3d, y_3d, z_3d)).T
    examples2.append(ex_3d)
    if minNbPts2 == -1 or ex_3d.shape[0] < minNbPts2:
        minNbPts2 = ex_3d.shape[0]

print(minNbPts2)

examples3 = []
minNbPts3 = -1

# Process each 2D drawing
for i, example in enumerate(bench):
    if i == 400: break
    ex = example.reshape(28, 28)
    x, y = np.where(ex > 0)

    # Replicate the 2D points along the z-axis
    z_levels = np.arange(28)  # 28 levels along z-axis
    x_3d = np.tile(x, 28)  # Repeat x-coordinates for each z-level
    y_3d = np.tile(y, 28)  # Repeat y-coordinates for each z-level
    z_3d = np.repeat(z_levels, len(x))  # Repeat each z-level len(x) times

    ex_3d = np.vstack((x_3d, y_3d, z_3d)).T
    examples3.append(ex_3d)
    if minNbPts3 == -1 or ex_3d.shape[0] < minNbPts3:
        minNbPts3 = ex_3d.shape[0]

print(minNbPts3)

train = open("quickdraw_dataset/train.txt", "w")
test = open("quickdraw_dataset/test.txt", "w")

mini = np.min(np.min(minNbPts3,minNbPts2),minNbPts)
# Process and save each 3D example
for i, example in enumerate(examples):
    # Ensure all examples have the same number of points
    if example.shape[0] > mini:
        example = example[:mini, :]

    np.save(f"quickdraw_dataset/airplane-{i}.npy", example)
    if i < 100:
        train.write(f"airplane-{i}.npy\n")
    else:
        test.write(f"airplane-{i}.npy\n")


# Process and save each 3D example
for i, example in enumerate(examples2):
    # Ensure all examples have the same number of points
    if example.shape[0] > mini:
        example = example[:mini, :]

    np.save(f"fork-{i}.npy", example)
    if i < 100:
        train.write(f"fork-{i}.npy\n")
    else:
        test.write(f"fork-{i}.npy\n")

# Process and save each 3D example
for i, example in enumerate(examples3):
    # Ensure all examples have the same number of points
    if example.shape[0] > mini:
        example = example[:mini, :]
    np.save(f"bench-{i}.npy", example)
    if i < 100:
        train.write(f"bench-{i}.npy\n")
    else:
        test.write(f"bench-{i}.npy\n")
print(mini)
test.close()
train.close()
