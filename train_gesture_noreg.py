import torch
import torch.nn as nn
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
import matplotlib.pyplot as plt

# 补丁
np.int = int

def main():
    parser = argparse.ArgumentParser(description='Ablation Study: PLIF Only (No Regularization)')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    # 既然是对比实验，保持相同的 Epoch 数
    parser.add_argument('-epochs', default=64, type=int, metavar='N', help='number of total epochs')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-data-dir', type=str, required=True, help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs_gesture', help='root dir for saving logs')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')

    args = parser.parse_args()
    print("Running Ablation Study: PLIF + ATan ONLY (No Dropout, No Label Smoothing)")

    # === [保留] 创新点 1 & 2: PLIF 和 ATan ===
    net = spiking_resnet.spiking_resnet18(
        pretrained=False,
        spiking_neuron=neuron.ParametricLIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        init_tau=2.0
    )

    # 修改第一层卷积
    net.conv1 = layer.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # === [修改] 移除了 Dropout ===
    # 这里的 fc 变回普通的线性层，没有 Dropout
    net.fc = nn.Sequential(
        layer.Flatten(),
        # layer.Dropout(p=0.5), <--- 已注释掉
        layer.Linear(net.fc.in_features, 11)
    )

    functional.set_step_mode(net, 'm')

    if args.cupy:
        functional.set_backend(net, 'cupy', instance=neuron.ParametricLIFNode)

    net.to(args.device)

    # 数据加载
    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T, split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T, split_by='number')

    train_data_loader = DataLoader(dataset=train_set, batch_size=args.b, shuffle=True, drop_last=True, num_workers=args.j, pin_memory=True)
    test_data_loader = DataLoader(dataset=test_set, batch_size=args.b, shuffle=False, drop_last=False, num_workers=args.j, pin_memory=True)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # === [修改] 移除了 Label Smoothing ===
    # 变回普通的 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()

    # 修改日志路径，避免覆盖
    out_dir = os.path.join(args.out_dir, f'Gesture_T{args.T}_ResNet18_PLIF_NoReg')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    writer = SummaryWriter(out_dir)
    max_test_acc = 0
    history = {'train_acc': [], 'test_acc': [], 'epochs': []}

    # 训练循环
    for epoch in range(args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0

        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)
            label = label.to(args.device)

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(frame).mean(0)
                    loss = criterion(out_fr, label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(frame).mean(0)
                loss = criterion(out_fr, label)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()
            functional.reset_net(net)

        train_loss /= train_samples
        train_acc /= train_samples
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
                test_samples += label.numel()
                functional.reset_net(net)

        test_acc /= test_samples

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epochs'].append(epoch + 1)

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            checkpoint = {
                'net': net.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_acc
            }
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        print(f'Epoch [{epoch+1}/{args.epochs}] Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f} Max Acc: {max_test_acc:.4f} Time: {time.time() - start_time:.1f}s')

    # 自动画图
    plt.figure(figsize=(10, 6))
    plt.plot(history['epochs'], history['train_acc'], label='Train Accuracy', marker='.')
    plt.plot(history['epochs'], history['test_acc'], label='Test Accuracy', marker='.')
    plt.title(f'Ablation: PLIF without Reg and Variable learning rate')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'accuracy_curve.png'))

if __name__ == '__main__':
    main()