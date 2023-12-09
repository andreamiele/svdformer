import torch
import utils.data_loaders
from models.SVDFormer import Model
import sys
sys.path.append('/content/svdformer_/utils')
import dataloader_quickdraw as qd
import torchvision.transforms as transforms
# Define folder path and transformations
import logging
import os
import torch
import utils.data_loaders
import utils.helpers
import argparse
from tensorboardX import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from time import time
from utils.average_meter import AverageMeter
from torch.optim.lr_scheduler import StepLR
from utils.schedular import GradualWarmupScheduler
from utils.loss_utils import *
from utils.helpers import seprate_point_cloud
from models.model_utils import PCViews
from models.SVDFormer import Model
from core.eval_qd import test_net

def convert_to_3d_point_cloud(drawing):
    """
    Converts a 2D drawing to a 3D point cloud tensor of shape 28x28x28.

    :param drawing: 2D drawing tensor of shape 28x28.
    :return: 3D point cloud tensor of shape 28x28x28.
    """
    # Add to the numpy list of 2D points a column of 0 to map to the hyperplane z=0 in R³
    tmp = np.zeros((drawing.shape[0]/2, 3))
    tmp[:,:-1] = drawing.reshape((drawing.shape[0]/2,2))

    # Convert to torch tensor
    point_cloud = torch.from_numpy(tmp)

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

    # In case you need to pad the tensors to have the same number of points in each cloud
    max_points = max([points.size(0) for points in point_list_batch])
    padded_point_list_batch = [torch.nn.functional.pad(points, (0, 0, 0, max_points - points.size(0))) for points in point_list_batch]

    return torch.stack(padded_point_list_batch)


def train_net(cfg):
    torch.backends.cudnn.benchmark = True

    train_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING["QuickDraw"](cfg).get_dataset("train")
    test_dataset_loader = utils.data_loaders.DATASET_LOADER_MAPPING["QuickDraw"](cfg).get_dataset("test")

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset_loader,
                                                    #batch_size=cfg.TRAIN.BATCH_SIZE,
                                                    #num_workers=cfg.CONST.NUM_WORKERS,
                                                    batch_size=4,
                                                    num_workers=2,
                                                    #collate_fn=utils.data_loaders.collate_fn_55,
                                                    pin_memory=True,
                                                    shuffle=True,
                                                    drop_last=False)
    val_data_loader = torch.utils.data.DataLoader(dataset=test_dataset_loader,
                                                  batch_size=2,
                                                  #num_workers=cfg.CONST.NUM_WORKERS//2,
                                                  num_workers=2,
                                                  #collate_fn=utils.data_loaders.collate_fn_55,
                                                  pin_memory=True,
                                                  shuffle=False)

    # Set up folders for logs and checkpoints
    #output_dir = os.path.join(cfg.DIR.OUT_PATH, '%s', datetime.now().isoformat())
    output_dir = os.path.join("quickdraw-out", '%s', datetime.now().isoformat())
    #cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    #cfg.DIR.LOGS = output_dir % 'logs'
    DIR_CHECKPOINTS = output_dir % 'checkpoints'
    DIR_LOGS = output_dir % 'logs'
    if not os.path.exists(DIR_CHECKPOINTS):
        os.makedirs(DIR_CHECKPOINTS)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(DIR_LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(DIR_LOGS, 'test'))

    model = Model(cfg)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # Create the optimizers
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                 #lr=cfg.TRAIN.LEARNING_RATE,
                                 lr=0.0001,
                                 weight_decay=0.0005)

    # lr scheduler
    #scheduler_steplr = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    scheduler_steplr = StepLR(optimizer, step_size=2, gamma=0.98)

    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=300,#cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)


    init_epoch = 0
    best_metrics = float('inf')
    steps = 0
    BestEpoch = 0

    #render = PCViews(TRANS=-cfg.NETWORK.view_distance, RESOLUTION=224)
    render = PCViews(TRANS=-1.5, RESOLUTION=224)
    """
    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        steps = cfg.TRAIN.WARMUP_STEPS+1
        lr_scheduler = StepLR(optimizer, step_size=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        optimizer.param_groups[0]['lr'] = cfg.TRAIN.LEARNING_RATE

        logging.info('Recover complete.')
    """
    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, 300+1):#cfg.TRAIN.N_EPOCHS + 1):
        epoch_start_time = time()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        model.train()

        total_cd_pc = 0
        total_cd_p1 = 0
        total_cd_p2 = 0

        batch_end_time = time()
        n_batches = len(train_data_loader)
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        with tqdm(train_data_loader) as t:
            for batch_idx, data in enumerate(t):
                data_time.update(time() - batch_end_time)
                #for k, v in data.items():
                #    data[k] = utils.helpers.var_or_cuda(v)
                # partial = data['partial_cloud']
                #gt = data['gtcloud']
                gt = convert_batch_to_xyz_format(convert_to_3d_point_cloud_data(data)).cuda()
                batchsize,npoints,_ = gt.size()
                if batchsize%2 != 0:
                    gt = torch.cat([gt,gt],0)
                partial, _ = seprate_point_cloud(gt, npoints, [int(npoints * 1/4) , int(npoints * 3/4)], fixed_points = None)
                partial_depth = torch.unsqueeze(render.get_img(partial),1)
                pcds_pred = model(partial,partial_depth)

                loss_total, losses = get_loss_PM(pcds_pred, partial, gt, sqrt=False)

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                cd_pc_item = losses[0].item() * 1e3
                total_cd_pc += cd_pc_item
                cd_p1_item = losses[1].item() * 1e3
                total_cd_p1 += cd_p1_item
                cd_p2_item = losses[2].item() * 1e3
                total_cd_p2 += cd_p2_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd_pc', cd_pc_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p1', cd_p1_item, n_itr)
                train_writer.add_scalar('Loss/Batch/cd_p2', cd_p2_item, n_itr)
                batch_time.update(time() - batch_end_time)
                batch_end_time = time()
                #t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                t.set_description('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, 300, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_pc_item, cd_p1_item, cd_p2_item]])

                #if steps <= cfg.TRAIN.WARMUP_STEPS:
                if steps <= 300:
                    lr_scheduler.step()
                    steps += 1

        avg_cdc = total_cd_pc / n_batches
        avg_cd1 = total_cd_p1 / n_batches
        avg_cd2 = total_cd_p2 / n_batches

        lr_scheduler.step()
        epoch_end_time = time()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p1', avg_cd1, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/cd_p2', avg_cd2, epoch_idx)
        logging.info(
            '[Epoch %d/%d] EpochTime = %.3f (s) Losses = %s' %
            #(epoch_idx, cfg.TRAIN.N_EPOCHS, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2]]))
            (epoch_idx, 300, epoch_end_time - epoch_start_time, ['%.4f' % l for l in [avg_cdc, avg_cd1, avg_cd2]]))

        
        if epoch_idx >= 150:
            if 150 <= epoch_idx <= 240 and epoch_idx % 20 == 0:
            # Validate and Checkpoint
            cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                BestEpoch = epoch_idx
                file_name = 'ckpt-best.pth'
            else:
                file_name = f'ckpt-epoch-{epoch_idx:03d}.pth'
            output_path = os.path.join(DIR_CHECKPOINTS, file_name)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

        elif 240 < epoch_idx <= 280 and epoch_idx % 10 == 0:
            # Similar operation as above
            cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                BestEpoch = epoch_idx
                file_name = 'ckpt-best.pth'
            else:
                file_name = f'ckpt-epoch-{epoch_idx:03d}.pth'
            output_path = os.path.join(DIR_CHECKPOINTS, file_name)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

        elif 280 < epoch_idx <= 300:
            # Again, similar operation as above
            cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model)
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                BestEpoch = epoch_idx
                file_name = 'ckpt-best.pth'
            else:
                file_name = f'ckpt-epoch-{epoch_idx:03d}.pth'
            output_path = os.path.join(DIR_CHECKPOINTS, file_name)
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, output_path)
            logging.info('Saved checkpoint to %s ...' % output_path)

    # Other training operations here

# Log the best performance after all epochs
        logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch, best_metrics))
    train_writer.close()
    val_writer.close()
