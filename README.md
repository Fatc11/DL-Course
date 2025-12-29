# DL-Course
该仓库用于提交对应的DL课程论文源码

# 文件说明
1.两个plot文件用于绘图，可以不用关注

2.read_ckpt是中途的获取权重脚本

3.train_cifar10dvs为起始的文件结构，由于Resnet在cifar10dvs上的效果过差，论文所示结果是使用浅层CSNN为网络架构对CIFAR-10DVS作评估

4.train_cifar_10csnn为cifar10dvs上的baseline模型（fixed为修改后的模型）

5.train_gesture为dvs-128gesture上的模型，fixed为完整修改后的模型

6.train_gesture_noreg为论文所示实验2，没有加入正则化，论文中另一个可变学习率的添加也是在这个文件上进行修改

# 运行说明(举例)

1.python train_cifar10_csnn.py -data-dir /path to your datasets/CIFAR-10DVS -T 10 -b 32 -amp -cupy

2.python train_gesture.py -data-dir /path to your datasets/DVS-128Gesture -T 16 -b 16 -amp -cupy
