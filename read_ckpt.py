import torch

# 你的 pth 文件路径
# 记得根据实际情况修改文件名，比如 CIFAR 的是 best_model.pth，Gesture 的是 checkpoint_max.pth
ckpt_path = './logs_cifar_csnn/best_model.pth'

if __name__ == '__main__':
    try:
        # map_location='cpu' 保证即使你在没有 GPU 的电脑上也能读
        checkpoint = torch.load(ckpt_path, map_location='cpu')

        print(f"Loading {ckpt_path} ...")

        # 打印所有键值，防止我记错 key 的名字
        print("Keys found:", checkpoint.keys())

        # 尝试获取准确率
        if 'max_test_acc' in checkpoint:
            print(f"🏆 Max Test Accuracy: {checkpoint['max_test_acc'] * 100:.4f}%")
        elif 'acc' in checkpoint:
            print(f"🏆 Max Test Accuracy: {checkpoint['acc'] * 100:.4f}%")

        # 顺便看看是第几个 epoch 跑出来的
        if 'epoch' in checkpoint:
            print(f"📅 Achieved at Epoch: {checkpoint['epoch'] + 1}")

    except FileNotFoundError:
        print("❌ 错误：找不到文件，请检查路径是否正确。")
    except Exception as e:
        print(f"❌ 读取出错: {e}")