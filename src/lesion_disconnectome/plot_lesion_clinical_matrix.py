#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesion vs Clinical Visualization Script
1. 散点图矩阵：以分数变量为行，lesion变量为列
2. 气泡图：展示lesion变量与临床分数的相关性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, linregress
from statsmodels.stats.multitest import fdrcorrection
import matplotlib
import os
import glob

# 设置matplotlib参数
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.unicode_minus'] = False

# 配色方案（参考scatter_array.py的Scheme_9）
COLOR_SCHEME = {
    'scatter_color': '#303f9f',  # 靛蓝
    'regr_line_color': '#ffc107',  # 琥珀金
    'kde_color': '#303f9f',  # 靛蓝
    'pos_corr_bg': '#fff9c4',  # 奶油黄
    'neg_corr_bg': '#c5cae9',  # 薰衣草蓝
}

def load_scatter_data(scatter_data_dir):
    """加载所有以1开头的CSV文件"""
    print("="*80)
    print("加载散点图数据...")
    print("="*80)
    
    # 获取所有以1开头的CSV文件
    csv_files = glob.glob(os.path.join(scatter_data_dir, '1_*.csv'))
    print(f"找到 {len(csv_files)} 个CSV文件")
    
    # 解析文件名，提取变量信息
    data_dict = {}
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # 解析文件名格式：1_lesion_clinical_<lesion_var>_vs_<clinical_var>_data.csv
        parts = filename.replace('.csv', '').replace('_data', '').split('_vs_')
        if len(parts) == 2:
            lesion_part = parts[0].replace('1_lesion_clinical_', '')
            clinical_part = parts[1]
            
            # 读取数据
            df = pd.read_csv(csv_file)
            if len(df.columns) == 2:
                lesion_var = df.columns[0]
                clinical_var = df.columns[1]
                
                key = (lesion_var, clinical_var)
                data_dict[key] = df
                print(f"  加载: {lesion_var} vs {clinical_var} ({len(df)} 样本)")
    
    return data_dict

def get_variable_display_name(var_name):
    """将变量名转换为显示名称"""
    display_names = {
        # 临床分数
        'repetition_subscore': 'Repetition',
        'comprehension_subscore': 'Comprehension',
        'naming_subscore': 'Naming',
        'wab_r_aq': 'WAB-R AQ',
        # Disconnection变量
        'af_disconnection': 'AF Disc.',
        'ilf_disconnection': 'ILF Disc.',
        'uf_disconnection': 'UF Disc.',
        'af_ilf_disconnection': 'AF+ILF Disc.',
        'af_uf_disconnection': 'AF+UF Disc.',
        'ilf_uf_disconnection': 'ILF+UF Disc.',
        'all_tracts_disconnection': 'All Tracts Disc.',
        # Network lesion变量
        'SomMot_lesion_pct': 'SomMot Lesion',
        'Vis_lesion_pct': 'Vis Lesion',
        'Cont_lesion_pct': 'Cont Lesion',
        'Default_lesion_pct': 'Default Lesion',
        'DorsAttn_lesion_pct': 'DorsAttn Lesion',
        'Limbic_lesion_pct': 'Limbic Lesion',
        'SalVentAttn_lesion_pct': 'SalVentAttn Lesion',
    }
    return display_names.get(var_name, var_name)

def plot_scatter_matrix(data_dict, output_dir):
    """绘制散点图矩阵：行为临床分数，列为lesion变量"""
    print("\n" + "="*80)
    print("绘制散点图矩阵...")
    print("="*80)
    
    # 定义临床分数（行）和lesion变量（列）
    clinical_vars = ['repetition_subscore', 'comprehension_subscore', 
                     'naming_subscore', 'wab_r_aq']
    
    # 为每个临床分数定义不同的配色方案
    clinical_colors = {
        'repetition_subscore': {
            'scatter': '#e74c3c',  # 红色
            'regr_line': '#c0392b',  # 深红色
            'pos_bg': '#fadbd8',  # 浅红色背景
            'neg_bg': '#d5dbdb'  # 灰色背景
        },
        'comprehension_subscore': {
            'scatter': '#3498db',  # 蓝色
            'regr_line': '#2980b9',  # 深蓝色
            'pos_bg': '#d6eaf8',  # 浅蓝色背景
            'neg_bg': '#d5dbdb'
        },
        'naming_subscore': {
            'scatter': '#2ecc71',  # 绿色
            'regr_line': '#27ae60',  # 深绿色
            'pos_bg': '#d5f4e6',  # 浅绿色背景
            'neg_bg': '#d5dbdb'
        },
        'wab_r_aq': {
            'scatter': '#9b59b6',  # 紫色
            'regr_line': '#8e44ad',  # 深紫色
            'pos_bg': '#ebdef0',  # 浅紫色背景
            'neg_bg': '#d5dbdb'
        }
    }
    
    # 从data_dict中提取所有lesion变量
    lesion_vars = []
    for (lesion_var, clinical_var) in data_dict.keys():
        if lesion_var not in lesion_vars:
            lesion_vars.append(lesion_var)
    
    # 按照特定顺序排列lesion变量
    lesion_order = [
        'af_disconnection', 'ilf_disconnection', 'uf_disconnection',
        'af_ilf_disconnection', 'af_uf_disconnection', 'ilf_uf_disconnection',
        'all_tracts_disconnection', 'SomMot_lesion_pct', 'Vis_lesion_pct'
    ]
    lesion_vars_sorted = [v for v in lesion_order if v in lesion_vars]
    
    print(f"临床分数变量 (行): {len(clinical_vars)}")
    print(f"Lesion变量 (列): {len(lesion_vars_sorted)}")
    
    # 创建子图网格
    n_rows = len(clinical_vars)
    n_cols = len(lesion_vars_sorted)
    
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, 
                            figsize=(n_cols * 2.5, n_rows * 2.5))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 遍历网格，填充每个子图
    for i, clinical_var in enumerate(clinical_vars):
        for j, lesion_var in enumerate(lesion_vars_sorted):
            ax = axes[i, j]
            
            # 获取数据
            key = (lesion_var, clinical_var)
            if key in data_dict:
                df = data_dict[key]
                X = df[lesion_var].values
                Y = df[clinical_var].values
                
                # 移除NaN值
                valid_mask = ~(np.isnan(X) | np.isnan(Y))
                X_valid = X[valid_mask]
                Y_valid = Y[valid_mask]
                
                if len(X_valid) >= 3:
                    # 计算相关系数
                    r, p = pearsonr(X_valid, Y_valid)
                    
                    # 获取当前临床分数的配色方案
                    colors = clinical_colors[clinical_var]
                    
                    # 设置背景色
                    if p < 0.05:
                        if r > 0:
                            ax.set_facecolor(colors['pos_bg'])
                        else:
                            ax.set_facecolor(colors['neg_bg'])
                    
                    # 绘制散点
                    ax.scatter(X_valid, Y_valid, 
                             s=20, alpha=0.7, 
                             color=colors['scatter'],
                             edgecolors='none')
                    
                    # 绘制回归线
                    slope, intercept, r_value, p_value_reg, std_err = linregress(X_valid, Y_valid)
                    x_line = np.linspace(np.min(X_valid), np.max(X_valid), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 
                           color=colors['regr_line'], 
                           linewidth=1.5, alpha=0.8)
                    
                    # 添加统计信息 - 显示r而不是R²
                    equation = f"y = {slope:.2f}x {'+' if intercept >= 0 else '-'} {abs(intercept):.2f}"
                    r_text = f"r = {r:.3f}"
                    
                    # 显著性标记
                    sig_marker = ''
                    if p < 0.001:
                        sig_marker = '***'
                    elif p < 0.01:
                        sig_marker = '**'
                    elif p < 0.05:
                        sig_marker = '*'
                    
                    text_str = f"{r_text}\n{equation}\np={p:.4f}{sig_marker}"
                    
                    ax.text(0.05, 0.95, text_str,
                           transform=ax.transAxes, fontsize=6,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='white', alpha=0.7))
            
            # 设置坐标轴标签
            if i == n_rows - 1:  # 最后一行显示x轴标签
                ax.set_xlabel(get_variable_display_name(lesion_var), 
                            fontsize=9, fontweight='bold')
            else:
                ax.set_xlabel('')
            
            if j == 0:  # 第一列显示y轴标签
                ax.set_ylabel(get_variable_display_name(clinical_var), 
                            fontsize=9, fontweight='bold')
            else:
                ax.set_ylabel('')
            
            # 设置刻度
            ax.tick_params(axis='both', which='major', labelsize=7, direction='in')
            
            # 添加网格
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            
            # 设置边框
            for spine in ['top', 'right', 'bottom', 'left']:
                ax.spines[spine].set_linewidth(1.0)
    
    # 添加总标题
    fig.suptitle('Lesion Variables vs Clinical Scores - Scatter Matrix', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # 保存图形
    output_file = os.path.join(output_dir, 'lesion_clinical_scatter_matrix.pdf')
    plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
    print(f"\n散点图矩阵已保存: {output_file}")
    
    # 同时保存PNG版本
    output_file_png = os.path.join(output_dir, 'lesion_clinical_scatter_matrix.png')
    plt.savefig(output_file_png, format='png', bbox_inches='tight', dpi=300)
    print(f"PNG版本已保存: {output_file_png}")
    
    plt.close()

def plot_bubble_chart(csv_file, output_dir):
    """绘制气泡图：lesion变量为列，临床分数为行"""
    print("\n" + "="*80)
    print("绘制气泡图...")
    print("="*80)
    
    # 读取数据
    df = pd.read_csv(csv_file)
    print(f"加载数据: {csv_file}")
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    
    # 定义临床分数（行）
    clinical_vars = ['repetition_subscore', 'comprehension_subscore', 
                     'naming_subscore', 'wab_r_aq']
    clinical_labels = [get_variable_display_name(v) for v in clinical_vars]
    
    # 提取lesion变量（列）- 排除临床分数和其他非lesion列
    exclude_cols = ['subject_id', 'X_Variable', 'Y_Variable', 'r', 'p_value', 
                    'n', 'X_name', 'Y_name', 'correlation_type'] + clinical_vars
    
    # 从CSV中提取X_Variable和Y_Variable来构建矩阵
    # 假设CSV格式是correlation_analysis的输出格式
    if 'X_Variable' in df.columns and 'Y_Variable' in df.columns:
        # 提取所有唯一的lesion变量
        lesion_vars_full = df['X_name'].unique().tolist()
        
        # 按照特定顺序排列
        lesion_order = [
            'af_disconnection', 'ilf_disconnection', 'uf_disconnection',
            'af_ilf_disconnection', 'af_uf_disconnection', 'ilf_uf_disconnection',
            'all_tracts_disconnection', 'SomMot_lesion_pct', 'Vis_lesion_pct',
            'Cont_lesion_pct', 'Default_lesion_pct', 'DorsAttn_lesion_pct',
            'Limbic_lesion_pct', 'SalVentAttn_lesion_pct'
        ]
        lesion_vars = [v for v in lesion_order if v in lesion_vars_full]
        lesion_labels = [get_variable_display_name(v) for v in lesion_vars]
        
        # 创建结果矩阵
        results = []
        for lesion_var in lesion_vars:
            for clinical_var in clinical_vars:
                # 查找对应的行
                row = df[(df['X_name'] == lesion_var) & (df['Y_name'] == clinical_var)]
                if len(row) > 0:
                    r = row['r'].values[0]
                    p = row['p_value'].values[0]
                    n = row['n'].values[0]
                    results.append({
                        'Lesion': lesion_var,
                        'Clinical': clinical_var,
                        'r': r,
                        'p_value': p,
                        'n': n
                    })
                else:
                    results.append({
                        'Lesion': lesion_var,
                        'Clinical': clinical_var,
                        'r': 0,
                        'p_value': 1.0,
                        'n': 0
                    })
        
        results_df = pd.DataFrame(results)
    else:
        print("错误：CSV文件格式不正确")
        return
    
    # FDR校正
    results_df['p_value'] = results_df['p_value'].fillna(1.0)
    rejected, q_values = fdrcorrection(results_df['p_value'])
    results_df['q_value'] = q_values
    
    # 计算实际数据中r的范围
    r_min = results_df['r'].min()
    r_max = results_df['r'].max()
    r_range = max(abs(r_min), abs(r_max))
    
    print(f"\n相关性分析结果:")
    print(f"  总共 {len(results_df)} 个相关性")
    print(f"  显著相关 (q < 0.05): {np.sum(results_df['q_value'] < 0.05)}")
    print(f"  相关系数范围: {r_min:.3f} 到 {r_max:.3f}")
    print(f"  |r| 最大值: {r_range:.3f}")
    
    # 定义气泡属性函数 - 只用大小表示相关程度，颜色表示显著性
    def get_bubble_attributes(row):
        corr, q_val = row['r'], row['q_value']
        
        # 计算相关系数的绝对值
        abs_corr = abs(corr)
        
        # 定义气泡大小范围：根据实际数据的r范围映射
        min_size = 50  # 最小气泡大小
        max_size = 800  # 最大气泡大小
        
        # 线性映射：|r| = 0 -> min_size, |r| = r_range -> max_size
        size = min_size + (max_size - min_size) * (abs_corr / r_range)
        
        # 颜色只表示显著性水平，使用紫色到深紫色的渐变
        if q_val > 0.05 or pd.isna(q_val):
            color = '#e0e0e0'  # 不显著：浅灰色
        else:
            # 显著的用紫色系，越显著颜色越深
            if q_val < 0.0001:
                color = '#4a148c'  # 极显著：深紫色
            elif q_val < 0.001:
                color = '#6a1b9a'  # 很显著：紫色
            elif q_val < 0.01:
                color = '#8e24aa'  # 显著：中紫色
            else:  # q_val < 0.05
                color = '#ab47bc'  # 边缘显著：浅紫色
        
        return pd.Series([size, color, abs_corr])
    
    results_df[['Size', 'Color', 'abs_r']] = results_df.apply(get_bubble_attributes, axis=1)
    
    # 创建图形
    fig = plt.figure(figsize=(14, 6), dpi=150)
    fig.patch.set_facecolor('white')
    
    # 主图区域
    ax_main = fig.add_axes([0.08, 0.15, 0.68, 0.75])
    
    # 绘制气泡
    for _, row in results_df.iterrows():
        lesion_idx = lesion_vars.index(row['Lesion'])
        clinical_idx = clinical_vars.index(row['Clinical'])
        
        ax_main.scatter(x=lesion_idx, y=clinical_idx, 
                       s=row['Size'], c=row['Color'], 
                       linewidths=0, zorder=2, alpha=0.8)
    
    # 设置坐标轴
    ax_main.set_xlim(-0.5, len(lesion_vars) - 0.5)
    ax_main.set_ylim(-0.5, len(clinical_vars) - 0.5)
    ax_main.set_xticks(range(len(lesion_vars)))
    ax_main.set_xticklabels(lesion_labels, rotation=45, ha='right', fontsize=9)
    ax_main.set_yticks(range(len(clinical_vars)))
    ax_main.set_yticklabels(clinical_labels, fontsize=10, fontweight='bold')
    
    # 添加网格
    ax_main.grid(True, which='major', axis='both', linestyle='-', 
                linewidth=0.5, color='gray', alpha=0.3, zorder=1)
    ax_main.set_axisbelow(True)
    
    # 设置边框
    for spine in ['top', 'right', 'bottom', 'left']:
        ax_main.spines[spine].set_linewidth(1.5)
        ax_main.spines[spine].set_color('black')
    
    ax_main.set_title('Lesion Variables vs Clinical Scores - Correlation Bubble Chart', 
                     fontsize=13, fontweight='bold', pad=15)
    
    # 添加图例
    # 图例1: 相关系数大小（气泡大小）- 根据实际数据范围
    ax_leg_size = fig.add_axes([0.78, 0.50, 0.18, 0.40])
    ax_leg_size.set_xlim(0, 1)
    ax_leg_size.set_ylim(0, 1)
    ax_leg_size.axis('off')
    
    # 显示不同相关系数对应的气泡大小，使用实际数据范围
    # 生成合理的图例值
    if r_range >= 0.6:
        size_values = [0.2, 0.4, 0.6, r_range]
    elif r_range >= 0.4:
        size_values = [0.1, 0.2, 0.3, r_range]
    else:
        size_values = [0.05, 0.1, 0.15, r_range]
    
    y_positions = np.linspace(0.75, 0.15, len(size_values))
    
    min_size = 50
    max_size = 800
    
    for size_val, y_pos in zip(size_values, y_positions):
        bubble_size = min_size + (max_size - min_size) * (size_val / r_range)
        ax_leg_size.scatter(0.25, y_pos, s=bubble_size, c='#8e24aa', alpha=0.6, edgecolors='none')
        ax_leg_size.text(0.55, y_pos, f'|r| = {size_val:.2f}', 
                        va='center', fontsize=10, fontweight='bold')
    
    ax_leg_size.text(0.5, 0.95, 'Correlation\nMagnitude', 
                    ha='center', va='top', fontsize=11, fontweight='bold')
    
    # 图例2: 显著性水平（颜色）
    ax_leg_sig = fig.add_axes([0.78, 0.10, 0.18, 0.35])
    ax_leg_sig.set_xlim(0, 1)
    ax_leg_sig.set_ylim(0, 1)
    ax_leg_sig.axis('off')
    
    sig_colors = ['#ab47bc', '#8e24aa', '#6a1b9a', '#4a148c']
    sig_labels = ['q < 0.05', 'q < 0.01', 'q < 0.001', 'q < 0.0001']
    y_positions_sig = np.linspace(0.70, 0.20, len(sig_colors))
    
    for color, label, y_pos in zip(sig_colors, sig_labels, y_positions_sig):
        ax_leg_sig.scatter(0.2, y_pos, s=200, c=color, alpha=0.8, edgecolors='none')
        ax_leg_sig.text(0.45, y_pos, label, va='center', fontsize=10, fontweight='bold')
    
    # 添加不显著的图例
    ax_leg_sig.scatter(0.2, 0.05, s=200, c='#e0e0e0', alpha=0.8, edgecolors='none')
    ax_leg_sig.text(0.45, 0.05, 'n.s. (q ≥ 0.05)', va='center', fontsize=10)
    
    ax_leg_sig.text(0.5, 0.95, 'Significance\nLevel (FDR)', 
                   ha='center', va='top', fontsize=11, fontweight='bold')
    
    # 保存图形
    output_file = os.path.join(output_dir, 'lesion_clinical_bubble_chart.pdf')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"\n气泡图已保存: {output_file}")
    
    # 同时保存PNG版本
    output_file_png = os.path.join(output_dir, 'lesion_clinical_bubble_chart.png')
    plt.savefig(output_file_png, bbox_inches='tight', dpi=300)
    print(f"PNG版本已保存: {output_file_png}")
    
    plt.close()

def main():
    """主函数"""
    print("="*80)
    print("Lesion vs Clinical Visualization")
    print("="*80)
    
    # 定义路径
    scatter_data_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\correlation_analysis_results\scatter_data"
    correlation_csv = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\correlation_analysis_results\1_lesion_vs_clinical.csv"
    output_dir = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\correlation_analysis_results"
    
    # 检查路径是否存在
    if not os.path.exists(scatter_data_dir):
        print(f"错误：找不到scatter_data目录: {scatter_data_dir}")
        return
    
    if not os.path.exists(correlation_csv):
        print(f"错误：找不到相关性CSV文件: {correlation_csv}")
        return
    
    # 1. 加载散点图数据
    data_dict = load_scatter_data(scatter_data_dir)
    
    if len(data_dict) == 0:
        print("错误：没有找到有效的数据文件")
        return
    
    # 2. 绘制散点图矩阵
    plot_scatter_matrix(data_dict, output_dir)
    
    # 3. 绘制气泡图
    plot_bubble_chart(correlation_csv, output_dir)
    
    print("\n" + "="*80)
    print("可视化完成！")
    print("="*80)

if __name__ == '__main__':
    main()

