import torch
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron, layer
# 引入 Spiking ResNet
from spikingjelly.activation_based.model import spiking_resnet
# 引入 CIFAR10DVS 数据集
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import datetime
import numpy as np


# ========== 加上这一行代码 ==========
np.int = int
# ======================================


def main():
    # 示例运行命令：
    # python train_cifar10dvs.py -data-dir /你的/数据集/路径/CIFAR10DVS -b 16 -T 10 -amp -cupy

    parser = argparse.ArgumentParser(description='Classify CIFAR-10-DVS')
    parser.add_argument('-T', default=10, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-data-dir', type=str, required=True, help='root dir of CIFAR10DVS dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')

    args = parser.parse_args()
    print(args)

    # 1. 定义模型：使用 Spiking ResNet18
    # CIFAR10-DVS 是双通道输入(On/Off)
    net = spiking_resnet.spiking_resnet18(pretrained=False, spiking_neuron=neuron.LIFNode,
                                          surrogate_function=surrogate.ATan(), detach_reset=True)
    # 修改第一层卷积以接受2通道输入 (默认为3)
    net.conv1 = layer.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # 修改全连接层输出为10类
    net.fc = layer.Linear(net.fc.in_features, 10)

    functional.set_step_mode(net, 'm')  # 开启多步模式

    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)

    net.to(args.device)

    # 2. 加载数据
    # data_type='frame' 会自动将事件流积分成 T 张图片
    # 第一次运行会比较慢，因为要生成缓存文件
    full_dataset = CIFAR10DVS(root=args.data_dir, data_type='frame', frames_number=args.T, split_by='number')

    # 手动划分 训练集(90%) 和 测试集(10%)
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = random_split(full_dataset, [train_size, test_size],
                                       generator=torch.Generator().manual_seed(42))

    train_data_loader = DataLoader(dataset=train_set, batch_size=args.b, shuffle=True, drop_last=True,
                                   num_workers=args.j, pin_memory=True)
    test_data_loader = DataLoader(dataset=test_set, batch_size=args.b, shuffle=False, drop_last=False,
                                  num_workers=args.j, pin_memory=True)

    # 3. 优化器与混合精度
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    # 增加 weight_decay=1e-4
    # optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    out_dir = os.path.join(args.out_dir, f'CIFAR10DVS_T{args.T}_ResNet18')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    writer = SummaryWriter(out_dir)

    # 4. 训练循环
    for epoch in range(args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        for frame, label in train_data_loader:
            optimizer.zero_grad()
            # === 新增：随机水平翻转 ===
            # if np.random.rand() < 0.5:
            #     frame = torch.flip(frame, [-1])  # 在最后一个维度(宽度W)上翻转
            # ========================
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(args.device)
            # CIFAR10 是 10 类
            label_onehot = F.one_hot(label, 10).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            # 重置脉冲神经元状态
            functional.reset_net(net)

        train_loss /= train_samples
        train_acc /= train_samples
        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

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

        test_acc /= (test_samples + len(test_set))  # fix denominator

        print(
            f'Epoch [{epoch}/{args.epochs}] Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f} Time: {time.time() - start_time:.1f}s')


if __name__ == '__main__':
    main()