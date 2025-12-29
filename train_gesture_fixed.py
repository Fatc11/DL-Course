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
import matplotlib.pyplot as plt  # [新增] 用于画图

# 补丁：防止 numpy 版本报错
np.int = int


def main():
    parser = argparse.ArgumentParser(description='Classify DVS Gesture with Improved SNN')
    # T=16 是 DVS Gesture 的经典设置，因为它包含时间信息较丰富
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    # DVS Gesture 数据少，Batch Size 16 是合适的，太大容易陷入局部最优
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N', help='number of total epochs')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-data-dir', type=str, required=True, help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs_gesture', help='root dir for saving logs')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    # 初始学习率可以稍大一点，配合 Cosine 衰减
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')

    args = parser.parse_args()
    print(args)

    # === [创新点 1 & 2] 使用 PLIF 和 ATan 替代梯度 ===
    # spiking_resnet 允许通过 kwargs 传递参数给 neuron
    net = spiking_resnet.spiking_resnet18(
        pretrained=False,
        spiking_neuron=neuron.ParametricLIFNode,  # [修改] 换成 PLIF
        surrogate_function=surrogate.ATan(),  # [修改] 确保使用 ATan
        detach_reset=True,
        init_tau=2.0  # PLIF 的初始 tau
    )

    # 修改第一层卷积以适应 DVS 的 2 通道输入
    net.conv1 = layer.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # === [创新点 3-B] Dropout 正则化 ===
    # DVS Gesture 数据极少，ResNet18 很容易过拟合，所以这里 Dropout 设为 0.5 比较稳妥
    # 将原来的单层 FC 替换为 Dropout + FC
    net.fc = nn.Sequential(
        layer.Flatten(),
        layer.Dropout(p=0.25),
        layer.Linear(net.fc.in_features, 11)
    )

    functional.set_step_mode(net, 'm')

    if args.cupy:
        # 注意：这里要指定 PLIF
        functional.set_backend(net, 'cupy', instance=neuron.ParametricLIFNode)

    net.to(args.device)

    # 2. 加载数据
    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T,
                              split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T,
                             split_by='number')

    train_data_loader = DataLoader(dataset=train_set, batch_size=args.b, shuffle=True, drop_last=True,
                                   num_workers=args.j, pin_memory=True)
    test_data_loader = DataLoader(dataset=test_set, batch_size=args.b, shuffle=False, drop_last=False,
                                  num_workers=args.j, pin_memory=True)

    # 3. 优化器与损失函数
    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # [新增] 余弦退火调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # === [创新点 3-A] Label Smoothing ===
    # 注意：使用 CrossEntropyLoss 自动处理 Logits，不需要手动 OneHot
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 4. 日志路径
    out_dir = os.path.join(args.out_dir, f'Gesture_T{args.T}_ResNet18_PLIF')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    writer = SummaryWriter(out_dir)

    max_test_acc = 0
    history = {'train_acc': [], 'test_acc': [], 'epochs': []}

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
            frame = frame.transpose(0, 1)  # [T, N, C, H, W]
            label = label.to(args.device)

            # [数据增强建议] DVS Gesture 也可以加一点简单的翻转，虽然不是必须
            # if np.random.rand() < 0.5:
            #     frame = torch.flip(frame, [-1])

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(frame).mean(0)
                    loss = criterion(out_fr, label)  # CrossEntropy 直接吃 label index
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

        # 更新学习率
        lr_scheduler.step()

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)

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

        # 记录数据
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['epochs'].append(epoch + 1)

        # 6. 保存模型
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            checkpoint = {
                'net': net.state_dict(),
                'epoch': epoch,
                'max_test_acc': max_test_acc
            }
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        print(
            f'Epoch [{epoch + 1}/{args.epochs}] LR: {optimizer.param_groups[0]["lr"]:.6f} Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f} Max Acc: {max_test_acc:.4f} Time: {time.time() - start_time:.1f}s')

    # 7. 自动画图
    plt.figure(figsize=(10, 6))
    plt.plot(history['epochs'], history['train_acc'], label='Train Accuracy', marker='.')
    plt.plot(history['epochs'], history['test_acc'], label='Test Accuracy', marker='.')
    plt.title(f'DVS128-Gesture Accuracy (PLIF + Reg + Variable learning rate)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'accuracy_curve.png'))
    print(f"Curve saved to {os.path.join(out_dir, 'accuracy_curve.png')}")


if __name__ == '__main__':
    main()