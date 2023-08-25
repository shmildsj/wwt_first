import matplotlib.pyplot as plt
import seaborn as sns

# 示例数据
datasets = ['MDAD', 'DrugVirus', 'aBiofilm']
models = ['RI', 'FI']
auc_values = [[98.86, 94.07, 99.03], [96.00, 86.27, 98.08]]

# 绘制柱状图
plt.figure(figsize=(12, 6))
bar_width = 0.35
bar_positions = [0, 1, 2]

for i in range(len(models)):
    plt.bar([pos + i * bar_width for pos in bar_positions], auc_values[i], bar_width,  label=models[i])

# 在柱形图上标明数值
for i in range(len(models)):
    for j in range(len(datasets)):
        plt.text(j + i * bar_width, auc_values[i][j] + 0.5, f'{auc_values[i][j]:.2f}', ha='center')

# 添加空白柱形图，使得数据集名称位于两个柱形图组的中间位置
plt.bar([pos + bar_width / 2 for pos in bar_positions], [0, 0, 0], bar_width, alpha=0)

plt.xlabel('Dataset')
plt.ylabel('AUC (%)')
plt.title('AUC Performance Comparison by Dataset and Model')
plt.xticks([pos + bar_width / 2 for pos in bar_positions], datasets)
plt.legend()
plt.tight_layout()
plt.show()
