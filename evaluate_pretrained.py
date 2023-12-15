import torch
from models.SVDFormer import Model
from config_qd import cfg
from os import listdir
from os.path import isfile, join
from utils.loss_utils import *
import utils.helpers
from models.model_utils import PCViews

model_path = ""
files_path = ""


def convert_to_3d_point_cloud(drawing):
    """
    Converts a 2D drawing to a 3D point cloud tensor of shape 28x28x28.

    :param drawing: 2D drawing tensor of shape 28x28.
    :return: 3D point cloud tensor of shape 28x28x28.
    """
    # Convert to torch tensor
    point_cloud = drawing.float()

    return point_cloud


def convert_to_3d_point_cloud_data(drawings):
    """
    Converts a batch of 2D drawings to 3D point clouds, each of shape 28x28x28.

    :param drawings: Batch of 2D drawing tensors.
    :return: Batch of 3D point cloud tensors.
    """
    res = [convert_to_3d_point_cloud(drawing) for drawing in drawings]

    # Stack all tensors in the list to create a batch tensor
    return torch.stack(res)

crop_ratio = {
        'easy': 1 / 4,
        'median': 1 / 2,
        'hard': 3 / 4
    }


onlyfiles = [files_path+"/"+f for f in listdir(files_path) if isfile(join(files_path+"/", f)) and f.split(".")[-1] == "npy" and len(f.split("-")) > 1]

model = Model(cfg)
model.load_state_dict(torch.load(model_path))
model.eval()
render = PCViews(TRANS=-cfg.NETWORK.view_distance, RESOLUTION=224)
mode = cfg.CONST.mode
choice = [torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]), torch.Tensor([-1, 1, 1]),
              torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]),
              torch.Tensor([-1, -1, -1])]

for file in onlyfiles:
  example = torch.load(file)
  
  gt = convert_to_3d_point_cloud_data(example).cuda()
  _, npoints = gt.size()
  num_crop = int(npoints * crop_ratio[mode])
  for partial_id, item in enumerate(choice):
    partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, num_crop, fixed_points=item)
    partial = fps_subsample(partial, 2048)
    partial_depth = torch.unsqueeze(render.get_img(partial), 1)
    prediction = model(partial.contiguous(),partial_depth)
    torch.save(prediction, file.replace(".npy", f"-output-{partial_id}.npy"))