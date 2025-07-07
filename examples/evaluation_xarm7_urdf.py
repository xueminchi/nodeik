from dataclasses import dataclass

import os
import copy
import numpy as np
import torch
import time
from torch.utils.data import Dataset, DataLoader
# add path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nodeik.utils import build_model

import warp as wp

from nodeik.robots.robot import Robot
from nodeik.training import KinematicsDataset, Learner, ModelWrapper

from pyquaternion import Quaternion
# from pybullet_panda_sim import PandaSim

@dataclass
class args:
    layer_type = 'concatsquash'
    dims = '1024-1024-1024-1024'
    num_blocks = 1   # CNF block 数
    time_length = 0.5 # CNF 时间长度
    train_T = False 
    divergence_fn = 'approximate'  # 发散度计算方法
    nonlinearity = 'tanh'  # 激活函数
    solver = 'dopri5'  # ODE 求解器
    atol = 1e-5 # 1e-3 for fast inference, 1e-5 for accurate inference
    rtol = 1e-5 # 1e-3 for fast inference, 1e-5 for accurate inference
    gpu = 0 # 使用第几号 GPU
    rademacher = False # 是否使用 Rademacher 随机向量
    num_samples = 10 # 每个参考姿态采样数
    num_references = 1 # 参考姿态数量
    seed = 1 # 全局随机种子
    model_checkpoint = os.path.join(os.path.dirname(__file__), '..', 'model','xarm7_epoch=0049-val_loss=-0.7936.ckpt') # 训练好模型的权重文件路径
    
np.random.seed(args.seed) # 设定 NumPy 随机种子
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')  # 选择 GPU 或 CPU

wp.init()

def run():
    filepath = os.path.join(os.path.dirname(__file__), 'assets', 'robots', 'xarm7', 'xarm7_robot.urdf')

    r = Robot(robot_path=filepath, ee_link_name='link_eef')
    task_dim = 7 # 1 end-effector SE(3)
    model = build_model(args, r.active_joint_dim, condition_dims=task_dim).to(device)
    learn = Learner.load_from_checkpoint(args.model_checkpoint, model=model, robot=r, std=1.0, num_samples=250, state_dim=r.active_joint_dim, condition_dim=task_dim)
    learn.model_wrapper.device = device
    nodeik = learn.model_wrapper
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ④ 采样参考末端姿态集合 pose_sets
    pose_sets = []                                        # 存储末端目标 SE(3)
    for _ in range(args.num_references):                  # 遍历参考数量
        qx = r.get_pair()                                 # get_pair() → (q, x) 混合向量
        x = qx[r.active_joint_dim:]                       # 取末端位姿 x（长度 7）
        for _ in range(args.num_samples):                 # 为同一姿态复制 num_samples 次
            pose_sets.append(x)

    c = np.array(pose_sets, dtype=np.float32)
    zero = torch.zeros(c.shape[0], 1).to(device)
    c = torch.from_numpy(c).to(device)
    x = torch.normal(mean=0.0, std=1.0, size=(c.shape[0],task_dim)).to(device)
    print(x.shape, c.shape, zero.shape)

    # extract the first ele of pose_sets
    pose_sets_0 = pose_sets[0]
    # repete for 5 times
    pose_sets = np.tile(pose_sets_0, (args.num_references, 1)).reshape(-1, task_dim)

    # ⑦ 推理：逆运动学 & 正运动学
    nodeik.eval()                                         # 设为评估模式（关闭 Dropout 等）
    ik_q, _ = nodeik.inverse_kinematics(pose_sets)        # IK 推理得到关节角 ik_q
    fk_sets = nodeik.forward_kinematics(ik_q)             # FK 计算得到末端位姿 fk_sets


    # save ik_q
    # np.save(os.path.join(os.path.dirname(__file__), 'assets', 'ik_q_xarm.npy'), ik_q)

    p_err = []
    q_err = []
    for a, b in zip(pose_sets, fk_sets):
        pos_norm = np.linalg.norm(a[:3] - b[:3])
        q1 = Quaternion(array=a[3:])
        q2 = Quaternion(array=b[3:])
        quat_norm = Quaternion.distance(q1,q2)
        p_err.append(pos_norm)
        q_err.append(quat_norm)

    print('mean position    error:', np.array(p_err).mean())
    print('mean orientation error:', np.array(q_err).mean())
    
if __name__ == '__main__':

    run()
    