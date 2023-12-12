import logging
import torch
import utils.data_loaders
import utils.helpers
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from models.model_utils import PCViews
from models.SVDFormer import Model


import torch
import logging
from tqdm import tqdm
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from models.model_utils import PCViews
from models.SVDFormer import Model
import numpy as np

def convert_to_3d_point_cloud(drawing):
    """
    Converts a 2D drawing to a 3D point cloud tensor of shape 28x28x28.

    :param drawing: 2D drawing tensor of shape 28x28.
    :return: 3D point cloud tensor of shape 28x28x28.
    """
    # Add to the numpy list of 2D points a column of 0 to map to the hyperplane z=0 in R³
    tmp = np.zeros((int(drawing.shape[0]/2), 3))
    tmp[:,:-1] = drawing.reshape((int(drawing.shape[0]/2),2))

    # Convert to torch tensor
    point_cloud = torch.from_numpy(tmp).float()

    return point_cloud
def convert_to_3d_point_cloud_data(drawings):
    """
    Converts a batch of 2D drawings to 3D point clouds, each of shape 28x28x28.
    :param drawings: Batch of 2D drawing tensors.
    :return: Batch of 3D point cloud tensors.
    """
    res = [convert_to_3d_point_cloud(drawing) for drawing in drawings]
    return torch.stack(res)

def convert_grid_to_point_list(point_cloud):
    """
    Converts a 3D grid point cloud to a list of points.
    :param point_cloud: 3D tensor of shape 28x28x28 representing a point cloud.
    :return: 2D tensor of shape (N, 3) representing the point cloud as a list of points.
    """
    non_zero_indices = torch.nonzero(point_cloud)
    return non_zero_indices.float()

def convert_batch_to_xyz_format(batch_point_clouds):
    """
    Converts a batch of 3D grid point clouds to the xyz format.
    :param batch_point_clouds: Batch of 3D grid point clouds.
    :return: Tensor of shape (B, N, 3), where B is the batch size, N is the number of points, and 3 represents the x, y, z coordinates.
    """
    batch_size = batch_point_clouds.size(0)
    point_list_batch = [convert_grid_to_point_list(batch_point_clouds[b]) for b in range(batch_size)]
    max_points = max([points.size(0) for points in point_list_batch])
    padded_point_list_batch = [torch.nn.functional.pad(points, (0, 0, 0, max_points - points.size(0))) for points in point_list_batch]
    return torch.stack(padded_point_list_batch)

def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None):
    torch.backends.cudnn.benchmark = True

    if test_data_loader is None:
        # Set up data loader
        test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING["QuickDraw"](cfg).get_dataset("test")
        test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader,
                                                  batch_size=2,
                                                  num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  pin_memory=True,
                                                  shuffle=False)
    '''
    # Setup networks and initialize networks
    if model is None:
        model = Model(cfg)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
    '''
    # Switch models to evaluation mode
    model.eval()

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['CD', 'DCD', 'F1'])
    test_metrics = AverageMeter(['CD', 'DCD', 'F1'])
    category_metrics = dict()
    render = PCViews(TRANS=-cfg.NETWORK.view_distance, RESOLUTION=224)

    # Eval settings
    crop_ratio = {
        'easy': 1 / 4,
        'median': 1 / 2,
        'hard': 3 / 4
    }
    choice = [torch.Tensor([1, 1, 1]), torch.Tensor([1, 1, -1]), torch.Tensor([1, -1, 1]), torch.Tensor([-1, 1, 1]),
              torch.Tensor([-1, -1, 1]), torch.Tensor([-1, 1, -1]), torch.Tensor([1, -1, -1]),
              torch.Tensor([-1, -1, -1])]

    mode = cfg.CONST.mode

    print('Start evaluating (mode: {:s}) ...'.format(mode))

    # Testing loop
    with tqdm(test_data_loader) as t:
            for batch_idx, data in enumerate(test_data_loader):
           
                gt = convert_to_3d_point_cloud_data(data).cuda()
                print(gt.size())
                batchsize,npoints,_ = gt.size()
                num_crop = int(npoints * crop_ratio[mode])
                for partial_id, item in enumerate(choice):
                    partial, _ = utils.helpers.seprate_point_cloud(gt, npoints, num_crop, fixed_points=item)
                    partial = fps_subsample(partial, 2048)
                    partial_depth = torch.unsqueeze(render.get_img(partial), 1)
                    pcds_pred = model(partial.contiguous(),partial_depth)
                    cdl1, cdl2, f1 = calc_cd(pcds_pred[-1], gt, calc_f1=True)
                    dcd, _, _ = calc_dcd(pcds_pred[-1], gt)

                    cd = cdl2.mean().item() * 1e3
                    dcd = dcd.mean().item()
                    f1 = f1.mean().item()

                    _metrics = [cd, dcd, f1]
                    test_losses.update([cd, dcd, f1])

                    test_metrics.update(_metrics)

                    t.set_description('Test[%d/%d]  Losses = %s Metrics = %s' %(batch_idx, n_samples,  ['%.4f' % l for l in test_losses.avg()
                                                                                ], ['%.4f' % m for m in _metrics]))

    # Print testing results
    print('============================ TEST RESULTS ============================')
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    print('Epoch ', epoch_idx, end='\t')
    for value in test_losses.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/cd', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/dcd', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/f1', test_losses.avg(2), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(0)
