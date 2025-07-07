from dataclasses import dataclass

import os

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from nodeik.utils import build_model

import warp as wp

from nodeik.robots.robot import Robot
from nodeik.training.datasets import KinematicsDataset
from nodeik.training.learner import Learner
from urdfpy import URDF  # 确保有这个 import

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint  # NEW

@dataclass
class Args:
    layer_type = 'concatsquash'
    dims = '1024-1024-1024-1024'
    num_blocks = 1 
    time_length = 0.5
    train_T = False
    divergence_fn = 'approximate'
    nonlinearity = 'tanh'
    solver = 'dopri5'
    atol = 1e-5
    rtol = 1e-5
    use_gpu = True
    gpu = 0
    rademacher = False
    adjoint = True
    max_epoch = 1000000000

args = Args()

if args.use_gpu:
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
else:
    device = 'cpu'

wp.init()

def get_robot(filepath):
    r = Robot(robot_path=filepath, ee_link_name='link_eef')

    batch_size = 512
    batch_in_epoch = 500
    val_size = 512
    dataset = KinematicsDataset(r, len_batch=batch_size*batch_in_epoch)
    val_dataset = KinematicsDataset(r, len_batch=val_size)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=val_size)
    
    return r, dataloader, val_dataloader

def run():
    # URDF file path
    filepath = os.path.join(os.path.dirname(__file__), 'assets', 'robots', 'xarm7', 'xarm7_robot.urdf')

    # Get robot object
    r, dataloader, val_dataloader = get_robot(filepath)

    # Build a CNF model
    model = build_model(args, r.active_joint_dim, condition_dims=7).to(device)
    params = sum(p.numel() for p in model.parameters())
    print('parameters', params)

    # Create a learner
    learn = Learner(model, robot=r, std=1.0, num_samples=250, state_dim=r.active_joint_dim, condition_dim=7)
    learn.model_wrapper.device = device
    print('device', device)

    # Initialize wandb
    wandb_logger = WandbLogger(
        project="nodeik-xarm7",
        name="train-xarm7-run",
        log_model=False  # 如果你不想自动保存模型
    )
    wandb_logger.experiment.config.update(vars(args))  # 保存超参数



    # 1️⃣ 定义 checkpoint 回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(os.path.dirname(__file__), "checkpoints"),  # 保存目录
        filename="xarm7-{epoch:04d}-{val_loss:.4f}",                     # 文件名格式
        monitor="val_loss",                                              # 监控指标
        mode="min",                                                      # 越小越好
        save_top_k=3,                                                    # 只保留 val_loss 最小的前三个
        save_last=True                                                   # 额外保存最后一次
    )


    # 2️⃣ 把 callback 交给 Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epoch,
        accelerator="gpu" if device != "cpu" else None,
        gpus=[args.gpu] if device != "cpu" else None,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        default_root_dir=os.path.join(os.path.dirname(__file__), "checkpoints"),
        logger=wandb_logger,
        callbacks=[checkpoint_callback]          # ← 关键行
    )

    trainer.fit(learn, dataloader, val_dataloader)


    # （可选）手动保存最终权重
    # torch.save(learn.state_dict(), "final_model.pth")


    wandb.finish()


if __name__ == '__main__':

    run()
    