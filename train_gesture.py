import torch
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from spikingjelly.activation_based.model import spiking_resnet
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import numpy as np

# 补丁：防止 numpy 版本报错
np.int = int


def main():
    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    # 默认 T=16 (你的最佳配置)
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N', help='number of total epochs')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-data-dir', type=str, required=True, help='root dir of DVS Gesture dataset')
    # 修改默认日志目录名为 logs_gesture，避免混淆
    parser.add_argument('-out-dir', type=str, default='./logs_gesture', help='root dir for saving logs')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')

    args = parser.parse_args()
    print(args)

    # 1. 定义模型：Gesture 是 11 类
    net = spiking_resnet.spiking_resnet18(pretrained=False, spiking_neuron=neuron.LIFNode,
                                          surrogate_function=surrogate.ATan(), detach_reset=True)
    net.conv1 = layer.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    net.fc = layer.Linear(net.fc.in_features, 11)

    functional.set_step_mode(net, 'm')

    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.LIFNode)

    net.to(args.device)

    # 2. 加载数据 (标准方式：分别加载 Train 和 Test)
    # 这样更严谨，且利用缓存速度快
    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T,
                              split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T,
                             split_by='number')

    train_data_loader = DataLoader(dataset=train_set, batch_size=args.b, shuffle=True, drop_last=True,
                                   num_workers=args.j, pin_memory=True)
    test_data_loader = DataLoader(dataset=test_set, batch_size=args.b, shuffle=False, drop_last=False,
                                  num_workers=args.j, pin_memory=True)

    # 3. 优化器
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # 4. 修正日志路径命名
    out_dir = os.path.join(args.out_dir, f'Gesture_T{args.T}_ResNet18')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    writer = SummaryWriter(out_dir)

    max_test_acc = 0

    # 5. 训练循环
    for epoch in range(args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 11).float()

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
                test_samples += label.numel()  # 修正计数逻辑
                functional.reset_net(net)

        test_acc /= test_samples

        # 6. 保存模型 (最重要的一步！)
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            checkpoint = {
                'net': net.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_acc
            }
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        print(
            f'Epoch [{epoch}/{args.epochs}] Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f} Max Acc: {max_test_acc:.4f} Time: {time.time() - start_time:.1f}s')


if __name__ == '__main__':
    main()