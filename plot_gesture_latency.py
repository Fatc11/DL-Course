import torch
import torch.nn as nn  # 需要导入 nn
from spikingjelly.activation_based import functional, surrogate, neuron, layer
from spikingjelly.activation_based.model import spiking_resnet
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

np.int = int


def main():
    parser = argparse.ArgumentParser(description='Plot Gesture Latency Curve')
    parser.add_argument('-T', default=16, type=int)
    parser.add_argument('-device', default='cuda:0')
    parser.add_argument('-b', default=16, type=int)
    parser.add_argument('-data-dir', type=str, required=True, help='/path/to/DVSGesture')
    parser.add_argument('-resume', type=str, required=True, help='/path/to/checkpoint_max.pth')
    parser.add_argument('-cupy', action='store_true')
    args = parser.parse_args()

    # ==========================================
    # 1. 重建模型 (必须与训练代码完全一致！)
    # ==========================================

    # [修正 1] 使用 ParametricLIFNode 而不是 LIFNode
    net = spiking_resnet.spiking_resnet18(
        pretrained=False,
        spiking_neuron=neuron.ParametricLIFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True,
        init_tau=2.0
    )

    # [标准操作] 修改第一层卷积以适应 DVS 的 2 通道
    net.conv1 = layer.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # [修正 2] 重建 FC 层结构 (Sequential: Flatten -> Dropout -> Linear)
    # 即使测试时不需要 Dropout，为了加载权重 key (fc.2.weight)，结构必须存在
    net.fc = nn.Sequential(
        layer.Flatten(),
        layer.Dropout(p=0.5),
        layer.Linear(net.fc.in_features, 11)
    )

    functional.set_step_mode(net, 'm')

    if args.cupy:
        # [修正 3] 指定 PLIF 后端
        functional.set_backend(net, 'cupy', instance=neuron.ParametricLIFNode)

    net.to(args.device)

    # 2. 加载权重
    if not os.path.exists(args.resume):
        print(f"Error: Model file not found at {args.resume}")
        return

    print(f"Loading model from {args.resume}...")
    checkpoint = torch.load(args.resume, map_location=args.device)

    if 'net' in checkpoint:
        net.load_state_dict(checkpoint['net'])
    else:
        net.load_state_dict(checkpoint)

    # [关键] 开启评估模式，这会让 Dropout 失效，保证测试结果稳定
    net.eval()
    print("Model loaded successfully!")

    # 3. 加载测试集
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T,
                             split_by='number')
    test_loader = DataLoader(test_set, batch_size=args.b, shuffle=False, drop_last=False, num_workers=4)

    # 4. 评估每个时间步的精度
    print("Evaluating Anytime Inference Accuracy...")

    correct_at_step = np.zeros(args.T)
    total_samples = 0

    with torch.no_grad():
        for frame, label in test_loader:
            frame = frame.to(args.device).transpose(0, 1)  # [T, N, C, H, W]
            label = label.to(args.device)

            # 获取输出序列 [T, N, 11]
            # 注意：Spiking ResNet 默认返回的是 [T, N, 11] (如果 step_mode='m')
            # 我们需要累积电压或发放率
            out_seq = net(frame)

            # 累积逻辑：模拟随时间推移，置信度越来越高
            accumulated_out = 0.0
            for t in range(args.T):
                accumulated_out += out_seq[t]
                pred = accumulated_out.argmax(1)
                correct_at_step[t] += (pred == label).float().sum().item()

            total_samples += label.numel()
            functional.reset_net(net)

    # 计算精度
    acc_curve = correct_at_step / total_samples

    # 5. 打印并画图
    print("\n=== Accuracy vs Time Step ===")
    for t, acc in enumerate(acc_curve):
        print(f"T={t + 1}: {acc:.2%}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.T + 1), acc_curve * 100, 'b-o', linewidth=2, label='Ours (PLIF + Reg)')

    # 标注最高点
    max_acc = acc_curve[-1] * 100
    plt.axhline(y=max_acc, color='r', linestyle='--', alpha=0.5, label=f'Final Acc: {max_acc:.2f}%')

    plt.title('Latency-Accuracy Trade-off', fontsize=14)
    plt.xlabel('Time Step (T)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xticks(range(1, args.T + 1))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    save_path = 'latency_chart_final_11.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nChart saved to {save_path}")


if __name__ == '__main__':
    main()