import matplotlib.pyplot as plt
import numpy as np


def draw_final_ablation_chart():
    # === 1. 最终数据 ===
    methods = ['Baseline', 'Arch. Innovation\n(PLIF + ATan)', '+ Scheduler', '+ Regularization\n(Full Method)']
    accuracies = [88.54, 91.32, 92.36, 93.75]  # 更新为 93.75

    # 颜色设计：强调第二根柱子（架构创新）
    # 灰 -> 亮红(核心) -> 浅蓝 -> 深蓝(最终)
    colors = ['#bdc3c7', '#e74c3c', '#3498db', '#2c3e50']

    # === 2. 创建画布 ===
    plt.figure(figsize=(10, 6.5))
    bars = plt.bar(methods, accuracies, color=colors, width=0.55, edgecolor='black', linewidth=1.2, alpha=0.9)

    # === 3. 细节美化 ===
    plt.ylim(85, 95.5)  # 调整范围适应 93.75
    plt.ylabel('Test Accuracy (%)', fontsize=14, labelpad=10)
    plt.title('Ablation Study: Architecture is the Key Driver', fontsize=16, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    plt.xticks(fontsize=11, fontweight='medium')
    plt.yticks(fontsize=12)

    # === 4. 标注数值与增幅 ===
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()

        # 柱顶数值
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.15,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

        # 标注增幅
        if i > 0:
            previous_acc = accuracies[i - 1]
            gain = acc - previous_acc

            # 专门为"架构创新"这一步做特殊的高亮标注
            if i == 1:
                plt.text(bar.get_x() + bar.get_width() / 2.0, height - 3.5,
                         f'MAX GAIN\n+{gain:.2f}%', ha='center', va='center',
                         color='white', fontsize=12, fontweight='bold')
            else:
                plt.text(bar.get_x() + bar.get_width() / 2.0, height - 2.5,
                         f'+{gain:.2f}%', ha='center', va='center',
                         color='white', fontsize=11, fontweight='bold')

    # === 5. 添加强调注释 (Highlight Architecture) ===
    plt.text(1, 91.32 + 1.2, 'Core\nInnovation', ha='center', va='bottom', color='#3498db', fontweight='bold')
    plt.plot([1, 1], [91.32 + 0.8, 91.32 + 1.1], color='#3498db', linewidth=1.5)

    plt.tight_layout()
    plt.savefig('final_thesis_chart.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    draw_final_ablation_chart()