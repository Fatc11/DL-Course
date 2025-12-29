import torch
import torch.nn as nn
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torch.utils.data import DataLoader, random_split
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import time  # 必须导入

# 补丁
np.int = int


class ImprovedCSNN(nn.Module):
    def __init__(self, T: int, channels: int = 128, use_cupy=False):
        super().__init__()
        self.T = T
        # [Paper Method] PLIF + Atan
        sg_function = surrogate.ATan()

        # 定义网络结构
        self.conv_fc = nn.Sequential(
            layer.Conv2d(2, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=sg_function, detach_reset=True),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=sg_function, detach_reset=True),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=sg_function, detach_reset=True),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=sg_function, detach_reset=True),
            layer.MaxPool2d(2, 2),

            layer.Flatten(),
            layer.Dropout(p=0.25),  # Dropout
            layer.Linear(channels * 8 * 8, 10),
            neuron.ParametricLIFNode(init_tau=2.0, surrogate_function=sg_function, detach_reset=True)
        )

        functional.set_step_mode(self, 'm')
        if use_cupy:
            functional.set_backend(self, 'cupy', instance=neuron.ParametricLIFNode)

    def forward(self, x):
        return self.conv_fc(x)


def main():
    parser = argparse.ArgumentParser(description='Classify CIFAR10-DVS with Improved CSNN')
    parser.add_argument('-T', default=10, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-epochs', default=50, type=int, metavar='N', help='number of total epochs')
    parser.add_argument('-data-dir', type=str, required=True, help='root dir of CIFAR10DVS')
    parser.add_argument('-out-dir', type=str, default='./logs_cifar_csnn', help='root dir for saving logs')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-lr', default=1e-3, type=float, help='learning rate')
    # [Added] resume 参数
    parser.add_argument('-resume', type=str, help='path to the checkpoint (pth file) to resume or evaluate')

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 初始化网络
    net = ImprovedCSNN(T=args.T, channels=128, use_cupy=args.cupy)
    net.to(args.device)

    # 数据准备
    full_dataset = CIFAR10DVS(root=args.data_dir, data_type='frame', frames_number=args.T, split_by='number')
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_set, test_set = random_split(full_dataset, [train_size, test_size],
                                       generator=torch.Generator().manual_seed(42))

    train_data_loader = DataLoader(train_set, batch_size=args.b, shuffle=True, drop_last=True, num_workers=4,
                                   pin_memory=True)
    test_data_loader = DataLoader(test_set, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    # === [新增] 余弦退火学习率调度器 ===
    # T_max 对应你的总 epochs 数
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    scaler = amp.GradScaler() if args.amp else None

    # [Paper Method] Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    start_epoch = 0

    # [Logic] 如果指定了 resume，则加载模型
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        net.load_state_dict(checkpoint['net'])
        # 如果是想继续训练，可以加载 optimizer，这里简化处理，假设 resume 主要是为了测试
        # optimizer.load_state_dict(checkpoint['optimizer'])
        # start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['acc']
        print(f"Loaded model with acc {best_acc} from epoch {checkpoint['epoch']}")

        # 如果你想 resume 后直接跳过训练去画图，可以把下面这行解开：
        # start_epoch = args.epochs

    history = {'train_acc': [], 'test_acc': [], 'epochs': []}

    # === 训练阶段 ===
    # 如果 start_epoch 已经达到设定值（比如通过 resume 跳过），则不进入循环
    if start_epoch < args.epochs:
        print(f"Start training on {args.device} with T={args.T}...")
        start_time = time.time()

        for epoch in range(start_epoch, args.epochs):
            net.train()
            train_loss = 0
            train_acc = 0
            train_samples = 0

            for frame, label in train_data_loader:
                optimizer.zero_grad()
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [T, N, C, H, W]

                if np.random.rand() < 0.5:
                    frame = torch.flip(frame, [-1])

                label = label.to(args.device)

                if scaler:
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

            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['epochs'].append(epoch + 1)

            print(f'Epoch [{epoch + 1}/{args.epochs}] Train Acc: {train_acc:.4f} Test Acc: {test_acc:.4f}')

            # === [新增] 更新学习率 ===
            scheduler.step()

            # [Logic] 保存最佳模型到硬盘
            if test_acc > best_acc:
                best_acc = test_acc
                save_dict = {
                    'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'acc': best_acc,
                    'epoch': epoch
                }
                save_path = os.path.join(args.out_dir, 'best_model.pth')
                torch.save(save_dict, save_path)
                print(f"Saved best model to {save_path}")

        total_time = time.time() - start_time
        print(f"Training finished in {total_time:.2f}s. Best Test Acc: {best_acc:.4f}")

        # 只有真正跑了训练才画这个图
        plt.figure(figsize=(10, 6))
        plt.plot(history['epochs'], history['train_acc'], label='Train Accuracy', marker='.')
        plt.plot(history['epochs'], history['test_acc'], label='Test Accuracy', marker='.')
        plt.title('Training and Testing Accuracy vs. Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.out_dir, 'accuracy_curve.png'))
        plt.close()

    # === 鲁棒性分析 (T变化) ===
    # 无论是否刚刚训练过，这里都尝试加载最好的模型来跑测试
    best_model_path = os.path.join(args.out_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print("Loading best model for robustness analysis...")
        checkpoint = torch.load(best_model_path, map_location=args.device)
        net.load_state_dict(checkpoint['net'])
    else:
        print("Warning: No best model found on disk. Using current model weights.")

    print("Starting Time-Step Robustness Analysis...")
    test_t_steps = [2, 4, 8, 10, 12]  # 建议的测试点
    t_acc_list = []

    net.eval()

    for t in test_t_steps:
        # 重新构建数据集（必须步骤，因为数据积分依赖 T）
        # 注意：这里会重新读取数据，可能稍微花点时间，但为了严谨是必须的
        temp_dataset = CIFAR10DVS(root=args.data_dir, data_type='frame', frames_number=t, split_by='number')

        # 保持划分一致性（Seed必须和前面一样）
        train_size_temp = int(0.9 * len(temp_dataset))
        test_size_temp = len(temp_dataset) - train_size_temp
        _, temp_test_set = random_split(temp_dataset, [train_size_temp, test_size_temp],
                                        generator=torch.Generator().manual_seed(42))

        temp_loader = DataLoader(temp_test_set, batch_size=args.b, shuffle=False, num_workers=4)

        current_t_acc = 0
        samples = 0

        # 告知网络当前的 T (虽然 PLIF 内部主要看输入，但逻辑上保持一致)
        net.T = t

        with torch.no_grad():
            for frame, label in temp_loader:
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [T, N, C, H, W]
                label = label.to(args.device)

                out_fr = net(frame).mean(0)
                current_t_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
                samples += label.numel()

        acc = current_t_acc / samples
        t_acc_list.append(acc)
        print(f"T={t}, Accuracy={acc:.4f}")

    # 绘制 T 变化图
    plt.figure(figsize=(8, 5))
    plt.plot(test_t_steps, t_acc_list, marker='o', linestyle='-', color='purple')
    plt.title('Accuracy vs. Simulation Time Steps (T)')
    plt.xlabel('Time Steps (T)')
    plt.ylabel('Test Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(args.out_dir, 't_robustness_curve.png'))
    plt.close()


if __name__ == '__main__':
    main()