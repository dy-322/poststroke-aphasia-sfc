#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制相关性分析的散点图
根据correlation_analysis的结果绘制特定的散点图
参考scatter.py的风格
根据失语症类型使用不同颜色
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

# 设置matplotlib参数 - 参考scatter.py
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 定义失语症类型的颜色方案
APHASIA_COLORS = {
    'Broca': '#67001f',        # 深红色
    'Anomic': '#d6604d',       # 橙红色
    'Conduction': '#f4a582',   # 浅橙色
    'Global': '#fddbc7',       # 浅粉色
    'Non-aphasic Stroke': '#D3D3D3'   # 蓝色
}

# 失语症类型顺序（用于图例）- 仅包含失语症患者，排除 non-aphasic stroke
APHASIA_TYPES = ['Broca', 'Anomic', 'Conduction', 'Global']

# 需要排除的类别（non-aphasic stroke / normal stroke）
EXCLUDE_APHASIA_TYPES = ['Non-aphasic Stroke', 'Non-aphasic', 'None', 'non-aphasic stroke', 'normal stroke non-aphasia']

def plot_scatter_with_regression(csv_file, output_file, x_label, y_label, title, exclude_non_aphasic=True):
    """
    绘制带回归线的散点图 - 参考scatter.py风格
    根据失语症类型使用不同颜色
    
    Parameters:
    -----------
    csv_file : str
        CSV数据文件路径
    output_file : str
        输出PDF文件路径
    x_label : str
        X轴标签
    y_label : str
        Y轴标签
    title : str
        图表标题
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 获取列名（第一列为X，第二列为Y，第三列为wab_type）
    x_col = df.columns[0]
    y_col = df.columns[1]
    
    # 检查是否有wab_type列
    has_aphasia_type = 'wab_type' in df.columns
    
    if has_aphasia_type:
        # 将空值填充为'Non-aphasic Stroke'
        df['wab_type'] = df['wab_type'].fillna('Non-aphasic Stroke')
        # 排除 non-aphasic stroke 类别
        if exclude_non_aphasic:
            wab_str = df['wab_type'].astype(str).str.strip().str.lower()
            exclude_set = {s.strip().lower() for s in EXCLUDE_APHASIA_TYPES}
            exclude_mask = wab_str.isin(exclude_set)
            n_excluded = exclude_mask.sum()
            df = df[~exclude_mask].copy()
            if n_excluded > 0:
                print(f"  已排除 {n_excluded} 个 Non-aphasic Stroke 样本，剩余 n={len(df)}")
            if len(df) == 0:
                print(f"  警告: 排除 non-aphasic 后无数据，跳过")
                return
    
    # 提取数据
    X = df[x_col].values
    Y = df[y_col].values
    
    # 移除X或Y中的缺失值
    valid_mask = ~(np.isnan(X) | np.isnan(Y))
    X_valid = X[valid_mask]
    Y_valid = Y[valid_mask]
    
    if has_aphasia_type:
        aphasia_types = df['wab_type'].values[valid_mask]
    
    # 计算相关系数
    r, p = stats.pearsonr(X_valid, Y_valid)
    n = len(X_valid)
    
    # 创建图形 - 使用scatter.py的尺寸比例
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if has_aphasia_type:
        # 按失语症类型分组绘制散点
        for aphasia_type in APHASIA_TYPES:
            mask = aphasia_types == aphasia_type
            if np.sum(mask) > 0:
                ax.scatter(X_valid[mask], Y_valid[mask], 
                          c=APHASIA_COLORS[aphasia_type],
                          s=40,  # 小散点
                          alpha=0.8,  # 高透明度
                          edgecolors='none',  # 无边框
                          label=aphasia_type)
    else:
        # 如果没有失语症类型信息，使用单一颜色
        ax.scatter(X_valid, Y_valid, 
                  c='#b2182b',  # 使用深红色
                  s=40,  # 小散点
                  alpha=0.8,  # 高透明度
                  edgecolors='none')  # 无边框
    
    # 绘制回归线 - 使用黑色实线
    z = np.polyfit(X_valid, Y_valid, 1)
    p_fit = np.poly1d(z)
    x_line = np.linspace(np.min(X_valid), np.max(X_valid), 100)
    ax.plot(x_line, p_fit(x_line), "k-", linewidth=2, alpha=0.75, zorder=0)
    
    # 添加统计信息 - 放在左上角
    stats_text = f'r = {r:.3f}\np = {p:.4f}\nn = {n}'
    ax.text(0.05, 0.95, stats_text,
            transform=ax.transAxes, 
            fontsize=12, 
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1))
    
    # 添加图例 - 放在右下角
    if has_aphasia_type:
        legend = ax.legend(loc='lower right', fontsize=10, frameon=True, 
                          fancybox=True, shadow=False, framealpha=0.8)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1)
    
    # 设置标签 - 参考scatter.py的字体大小
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    
    # 设置边框样式 - 参考scatter.py
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置刻度样式
    ax.tick_params(axis='both', labelsize=12, width=1.5, length=3)
    
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"已保存: {os.path.basename(output_file)}")
    print(f"  r = {r:.4f}, p = {p:.4f}, n = {n}")
    
    if has_aphasia_type:
        # 打印每种失语症类型的数量
        print(f"  失语症类型分布:")
        for aphasia_type in APHASIA_TYPES:
            count = np.sum(aphasia_types == aphasia_type)
            if count > 0:
                print(f"    {aphasia_type}: {count}")

def main():
    """主函数"""
    print("="*80)
    print("绘制相关性分析散点图")
    print("="*80)
    
    # 数据目录
    data_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\correlation_analysis_results\scatter_data"
    output_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\correlation_analysis_results\custom_scatter_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义要绘制的散点图配置
    scatter_configs = [
        {
            'csv_file': '4_ai_clinical_DorsAttn-ai_vs_comprehension-subscore_data.csv',
            'output_file': 'scatter_DorsAttn_AI_vs_Comprehension.pdf',
            'x_label': 'DorsAttn AI (Asymmetry Index)',
            'y_label': 'Comprehension Subscore',
            'title': 'DorsAttn AI vs Comprehension'
        },
        {
            'csv_file': '4_ai_clinical_dorsal-ai_vs_comprehension-subscore_data.csv',
            'output_file': 'scatter_Dorsal_AI_vs_Comprehension.pdf',
            'x_label': 'Dorsal AI (Asymmetry Index)',
            'y_label': 'Comprehension Subscore',
            'title': 'Dorsal AI vs Comprehension'
        },
        {
            'csv_file': '4_ai_clinical_global-ai_vs_repetition-subscore_data.csv',
            'output_file': 'scatter_Global_AI_vs_Repetition.pdf',
            'x_label': 'Global AI (Asymmetry Index)',
            'y_label': 'Repetition Subscore',
            'title': 'Global AI vs Repetition'
        },
        {
            'csv_file': '3_lesion_sfc_af-disconnection_vs_dorsal-sfc_data.csv',
            'output_file': 'scatter_AF_Disconnection_vs_Dorsal_SFC.pdf',
            'x_label': 'AF Disconnection',
            'y_label': 'Dorsal SFC',
            'title': 'AF Disconnection vs Dorsal SFC'
        },
        {
            'csv_file': '3_lesion_sfc_af-disconnection_vs_ventral-sfc_data.csv',
            'output_file': 'scatter_AF_Disconnection_vs_Ventral_SFC.pdf',
            'x_label': 'AF Disconnection',
            'y_label': 'Ventral SFC',
            'title': 'AF Disconnection vs Ventral SFC'
        }
    ]
    
    # 绘制每个散点图
    print("\n开始绘制散点图...\n")
    for config in scatter_configs:
        csv_path = os.path.join(data_dir, config['csv_file'])
        output_path = os.path.join(output_dir, config['output_file'])
        
        if not os.path.exists(csv_path):
            print(f"警告: 文件不存在 - {config['csv_file']}")
            continue
        
        print(f"\n绘制: {config['title']}")
        plot_scatter_with_regression(
            csv_path,
            output_path,
            config['x_label'],
            config['y_label'],
            config['title']
        )
    
    print("\n" + "="*80)
    print(f"所有散点图已保存到: {output_dir}")
    print("="*80)

if __name__ == '__main__':
    main()

