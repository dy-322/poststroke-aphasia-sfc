# ----------云雨图----------
# Python版本要求：3.6
# 需要安装的库：numpy pandas matplotlib openpyxl
# 镜像安装命令：
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy pandas matplotlib openpyxl
# 导入必要的库
import matplotlib
# matplotlib.use('Qt5Agg')  # 或 TkAgg


import numpy as np  # 数值计算库
import pandas as pd  # 数据处理库
import matplotlib.pyplot as plt  # 绘图库
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch
# ==================== 1. 数据准备 ====================
# 定义CSV文件路径（注意：r表示原始字符串，避免转义字符问题）
file_path = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\source\results\significance_analysis\subject_network_cohens_d.csv"


# 使用pandas读取CSV文件
df = pd.read_csv(file_path)
# 数据集的特征变量（列名）
categories = ["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis"]


def draw_half_violin(ax, data, position, color, width, alpha=0.45):
    """参考 good_look.py 的实现，绘制水平半小提琴图"""
    if len(data) == 0:
        return
    violin_parts = ax.violinplot(
        data,
        positions=[position],
        vert=False,
        showmeans=False,
        showextrema=False,
        widths=width
    )
    body = violin_parts['bodies'][0]
    path = body.get_paths()[0]
    vertices = path.vertices
    # 只保留中心线之上的区域，实现半小提琴效果
    vertices[:, 1] = np.where(vertices[:, 1] > position, vertices[:, 1], position)
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(alpha)
    body.set_linewidth(1.5)
# 获取所有的aphasia_type类型
aphasia_types = sorted(df['aphasia_type'].unique())
n_groups = len(aphasia_types)  # 每个类别下的组数
print(f"发现 {n_groups} 种 aphasia_type: {aphasia_types}")
# 计算所有数据的最小值和最大值，用于设置x轴范围与背景颜色
x_min = df[categories].min().min()
x_max = df[categories].max().max()
padding = (x_max - x_min) * 0.05
# ==================== 2. 可视化设置 ====================
# 创建图形和坐标轴，设置图形大小（交换长宽）
fig, ax = plt.subplots(figsize=(6, 10))
# 设置画布背景颜色（x<0区域与x>0区域不同颜色）
left_limit = x_min - padding
right_limit = x_max + padding
# ax.axvspan(left_limit, 0, color="#f0f4ff", alpha=0.7, zorder=0)
# 突出显示 x=0 的分界线
ax.axvline(0, color='black', linewidth=1, zorder=1)
# 定义颜色方案（按失语症类别分配颜色，颜色个数与aphasia_type个数相同）
# 为每个aphasia_type定义颜色
aphasia_colors = ["#d36a87", "#ea9979", "#83b6b5", "#bcdfa7", "#a596ee"]  # 5种颜色对应5种失语症类型
# 如果失语症类型超过5种，可以添加更多颜色
if n_groups > len(aphasia_colors):
    # 扩展颜色列表
    additional_colors = ['#9bb55e', '#fcd744', '#dc8e4e', '#ba7ab1', '#ea617b']
    aphasia_colors.extend(additional_colors[:n_groups - len(aphasia_colors)])
# 创建颜色字典，将每个aphasia_type映射到对应颜色
color_dict = {aphasia_type: aphasia_colors[j] for j, aphasia_type in enumerate(aphasia_types)}
print(f"颜色映射: {color_dict}")
# 设置类别在y轴上的位置（0,1,2,3,4,5,6）
positions = np.arange(len(categories))
violin_width = 0.15  # 半小提琴图宽度（更窄）
group_spacing = 0.1  # 每组之间的间距（沿y轴）
# ==================== 3. 绘制图形 ====================
# 遍历每个类别进行绘图（沿y轴布局）
for i, category in enumerate(categories):
    # 计算每组的位置偏移（使多组沿y轴并排显示）
    # 总高度 = (n_groups - 1) * group_spacing
    # 起始位置 = positions[i] - 总高度 / 2
    total_height = (n_groups - 1) * group_spacing
    start_pos = positions[i] - total_height / 2

    # 遍历每个aphasia_type组
    for j, aphasia_type in enumerate(aphasia_types):
        # 获取当前类别和当前aphasia_type的数据值
        data_subset = df[df['aphasia_type'] == aphasia_type]
        data_points = data_subset[category].values

        if len(data_points) == 0:
            continue  # 如果该组没有数据，跳过

        # 计算当前组在y轴上的位置
        group_pos = start_pos + j * group_spacing

        # ----------------- 3.1 绘制半小提琴图（无箱体与散点） -----------------
        current_color = color_dict[aphasia_type]
        draw_half_violin(ax, data_points, group_pos, current_color, violin_width, alpha=0.45)

# 构造图例，展示失语症类型与颜色映射
legend_elements = [
    Patch(facecolor=color_dict[a_type], edgecolor=color_dict[a_type], label=a_type, alpha=0.6)
    for a_type in aphasia_types
]
ax.legend(
    handles=legend_elements,
    title="Aphasia type",
    frameon=False,
    loc="upper right",
    fontsize=12,
    title_fontsize=12
)
# ==================== 4. 图表美化 ====================
# 设置y轴刻度位置和标签（显示网络类别）
ax.set_yticks(positions)
ax.set_yticklabels(categories, fontsize=14)
# 设置x轴范围、刻度标签字体大小
ax.set_xlim(left_limit, right_limit)
ax.tick_params(axis='x', labelsize=14)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel("Cohen's d value", fontsize=15)

# 添加x轴网格线（虚线，半透明）
ax.grid(axis='x', linestyle='--', alpha=0.7)
# # 显示边框
for spine in ['top', 'right', 'bottom', 'left']:
    ax.spines[spine].set_visible(True)
    ax.spines[spine].set_linewidth(2)  # 设置边框线宽
    ax.spines[spine].set_color('black')  # 设置边框颜色
# 添加图表标题（pad参数控制标题与图的间距）
# plt.title('', pad=20, fontsize=22)
# 调整子图布局（避免元素被裁剪）

plt.tight_layout()
# 显示图形
plt.savefig('Multi_group_network.png', dpi=600, bbox_inches='tight')
# plt.show()


