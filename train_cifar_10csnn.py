import torch
import torch.nn as nn
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import numpy as np

# 补丁
np.int = int


# === 定义简单的 CSNN 架构 ===
class CSNN(nn.Module):
    def __init__(self, T: int, channels: int = 128, use_cupy=False):
        super().__init__()
        self.T = T
        # 结构：Conv-BN-LIF-MaxPool 重复几次
        self.conv_fc = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),  # 128 -> 64

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),  # 64 -> 32

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),  # 32 -> 16

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.MaxPool2d(2, 2),  # 16 -> 8

            layer.Flatten(),
            layer.Linear(channels * 8 * 8, 10),  # CIFAR10 是 10 类
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
        )

        functional.set_step_mode(self, 'm')
        if use_cupy:
            functional.set_backend(self, 'cupy', instance=neuron.LIFNode)

    def forward(self, x):
        return self.conv_fc(x)


def main():
    parser = argparse.ArgumentParser(description='Classify CIFAR10-DVS with CSNN')
    parser.add_argument('-T', default=10, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=80, type=int, metavar='N', help='number of total epochs')
    parser.add_argument('-data-dir', type=str, required=True, help='root dir of CIFAR10DVS')
    parser.add_argument('-out-dir', type=str, default='./logs_cifar_csnn', help='root dir for saving logs')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    args = parser.parse_args()

    # 1. 初始化模型
    net = CSNN(T=args.T, channels=128, use_cupy=args.cupy)
    net.to(args.device)

    # 2. 数据加载
    full_dataset = CIFAR10DVS(root=args.data_dir, data_type='frame', frames_number=args.T, split_by='number')
    # 9:1 划分
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = random_split(full_dataset, [train_size, test_size],
                                       generator=torch.Generator().manual_seed(42))

    train_data_loader = DataLoader(train_set, batch_size=args.b, shuffle=True, drop_last=True, num_workers=4,
                                   pin_memory=True)
    test_data_loader = DataLoader(test_set, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scaler = amp.GradScaler() if args.amp else None

    # 3. 训练循环
    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [T, N, C, H, W]

            # 简单的 Data Augmentation: 随机水平翻转
            if np.random.rand() < 0.5:
                frame = torch.flip(frame, [-1])

            label = label.to(args.device)
            label_onehot = torch.nn.functional.one_hot(label, 10).float()

            if scaler:
                with amp.autocast():
                    out_fr = net(frame).mean(0)
                    loss = torch.nn.functional.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(frame).mean(0)
                loss = torch.nn.functional.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)

        train_loss /= train_samples
        train_acc /= train_samples

        # 测试
        net.eval()
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)
                label = label.to(args.device)
                out_fr = net(frame).mean(0)
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
                test_samples += label.numel()

        test_acc /= test_samples
        print(f'Epoch [{epoch}/{args.epochs}] Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f}')


if __name__ == '__main__':
    main()