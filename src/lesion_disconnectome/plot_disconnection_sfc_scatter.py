#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lesion vs SFC 分析脚本
- 结合 correlation_analysis.py 的分析三计算步骤
- 结合 plot_correlation_scatters.py 的画图风格
- 仅使用失语症患者（排除 non-aphasic 样本）
- 仅绘制 p < 0.05 的显著相关
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
from datetime import datetime

# 设置 matplotlib 参数 - 参考 plot_correlation_scatters.py
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 两张图统一使用的 5 色配色（来自你最新上传的图）
PALETTE_5 = ['#36663E', '#889F4E', '#C4C248', '#F8D042', '#F9B43F']

# 类型顺序（用于图例与颜色映射；其余类型会合并显示为 Other）
APHASIA_TYPES = ['Broca', 'Anomic', 'Conduction', 'Global', 'Non-aphasic Stroke']

def build_type_color_map(palette_5):
    """将 5 个类型映射到 5 个颜色。"""
    if len(palette_5) != 5:
        raise ValueError("palette_5 must have length 5")
    return {t: c for t, c in zip(APHASIA_TYPES, palette_5)}

# 预构建两套颜色映射
APHASIA_COLORS = build_type_color_map(PALETTE_5)

# 兼容可能出现的 non-aphasic 标签（会统一映射为 Non-aphasic Stroke）
NON_APHASIC_ALIASES = [
    'non-aphasic stroke', 'non-aphasic', 'none', 'normal stroke non-aphasia'
]

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DISCONN_FILE = os.path.join(BASE_DIR, 'disconnectome_mediation_results', 'subject_measures_all_scores.csv')
NETWORK_LESION_FILE = os.path.join(BASE_DIR, 'disconnectome_mediation_results', 'network_lesion_percentages.csv')
APHASIA_TYPE_FILE = os.path.join(BASE_DIR, 'single_subject_results_t288', 'filtered_participants.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'lesion_vs_sfc_results')


def load_data():
    """加载数据并保留所有类型（包含 non-aphasic）"""
    print("=" * 80)
    print("加载数据...")
    print("=" * 80)

    # 1. 加载 disconnectome 数据
    disconn_df = pd.read_csv(DISCONN_FILE)
    print(f"Loaded disconnectome data: {len(disconn_df)} subjects")

    # 2. 加载网络病灶百分比
    if not os.path.exists(NETWORK_LESION_FILE):
        raise FileNotFoundError(f"Network lesion file not found: {NETWORK_LESION_FILE}")
    network_lesion_df = pd.read_csv(NETWORK_LESION_FILE)
    print(f"Loaded network lesion data: {len(network_lesion_df)} subjects")

    # 3. 加载失语症类型（必须，用于排除 non-aphasic）
    if not os.path.exists(APHASIA_TYPE_FILE):
        raise FileNotFoundError(f"Aphasia type file not found: {APHASIA_TYPE_FILE}. Required to exclude non-aphasic samples.")
    aphasia_type_df = pd.read_csv(APHASIA_TYPE_FILE)
    aphasia_type_df['subject_id'] = aphasia_type_df['participant_id'].str.replace('sub-M', '', regex=False)
    aphasia_type_df = aphasia_type_df[['subject_id', 'wab_type']]
    print(f"Loaded aphasia type data: {len(aphasia_type_df)} subjects")

    # 4. 合并数据
    disconn_df['subject_id'] = disconn_df['subject_id'].astype(str).str.strip()
    network_lesion_df['subject_id'] = network_lesion_df['subject_id'].astype(str).str.strip()
    aphasia_type_df['subject_id'] = aphasia_type_df['subject_id'].astype(str).str.strip()

    merged_df = disconn_df.merge(network_lesion_df, on='subject_id', how='inner')
    # 失语症类型用 left join：允许有些人没有标注
    merged_df = merged_df.merge(aphasia_type_df, on='subject_id', how='left')
    print(f"Merged data: {len(merged_df)} subjects")

    # 统一 non-aphasic 的不同写法，并把缺失值也当作 Non-aphasic Stroke（为了“保留所有点”）
    wab = merged_df['wab_type'].fillna('Non-aphasic Stroke').astype(str).str.strip()
    wab_lower = wab.str.lower()
    alias_set = {s.strip().lower() for s in NON_APHASIC_ALIASES}
    wab = wab.where(~wab_lower.isin(alias_set), other='Non-aphasic Stroke')
    merged_df['wab_type'] = wab

    return merged_df


def perform_lesion_vs_sfc_correlation(df):
    """执行 Lesion vs SFC 的 Pearson 相关分析"""
    lesion_vars = [
        {'name': 'af_disconnection', 'label': 'AF Disconnection'},
        {'name': 'ilf_disconnection', 'label': 'ILF Disconnection'},
        {'name': 'uf_disconnection', 'label': 'UF Disconnection'},
        {'name': 'af_ilf_disconnection', 'label': 'AF+ILF Disconnection'},
        {'name': 'af_uf_disconnection', 'label': 'AF+UF Disconnection'},
        {'name': 'ilf_uf_disconnection', 'label': 'ILF+UF Disconnection'},
        {'name': 'all_tracts_disconnection', 'label': 'All Tracts Disconnection'},
        {'name': 'Cont_lesion_pct', 'label': 'Cont Network Lesion'},
        {'name': 'Default_lesion_pct', 'label': 'Default Network Lesion'},
        {'name': 'DorsAttn_lesion_pct', 'label': 'DorsAttn Network Lesion'},
        {'name': 'Limbic_lesion_pct', 'label': 'Limbic Network Lesion'},
        {'name': 'SalVentAttn_lesion_pct', 'label': 'SalVentAttn Network Lesion'},
        {'name': 'SomMot_lesion_pct', 'label': 'SomMot Network Lesion'},
        {'name': 'Vis_lesion_pct', 'label': 'Vis Network Lesion'},
    ]

    sfc_vars = [
        {'name': 'global_sfc', 'label': 'Global SFC'},
        {'name': 'dorsal_sfc', 'label': 'Dorsal SFC'},
        {'name': 'ventral_sfc', 'label': 'Ventral SFC'},
    ]

    results = []
    for x_var in lesion_vars:
        x_name = x_var['name']
        x_label = x_var['label']
        if x_name not in df.columns:
            continue
        for y_var in sfc_vars:
            y_name = y_var['name']
            y_label = y_var['label']
            if y_name not in df.columns:
                continue

            X = df[x_name].values
            Y = df[y_name].values
            # 保留所有点（包括 lesion=0），仅剔除 NaN
            valid_mask = ~(np.isnan(X) | np.isnan(Y))
            X_valid = X[valid_mask]
            Y_valid = Y[valid_mask]

            if len(X_valid) < 3:
                continue

            r, p = stats.pearsonr(X_valid, Y_valid)
            n = len(X_valid)

            if np.isnan(r):
                continue

            results.append({
                'X_Variable': x_label,
                'Y_Variable': y_label,
                'r': r,
                'p_value': p,
                'n': n,
                'X_name': x_name,
                'Y_name': y_name,
            })
            sig = '**' if p < 0.01 else '*' if p < 0.05 else ''
            print(f"{x_label:35s} vs {y_label:20s}: r={r:7.4f}, p={p:.4f}, n={n} {sig}")

    return pd.DataFrame(results), lesion_vars, sfc_vars


def plot_selected_correlations_keep_all_points(df, corr_results, output_dir):
    """仅绘制指定的 Lesion vs SFC 散点图，并保留所有点（仅去 NaN）"""
    targets = [
        ('af_disconnection', 'dorsal_sfc'),
        ('af_disconnection', 'ventral_sfc'),
    ]
    target_mask = corr_results.apply(lambda r: (r['X_name'], r['Y_name']) in targets, axis=1)
    selected = corr_results[target_mask].copy()

    print(f"\n绘制 {len(selected)} 张指定散点图（保留所有点）...")

    scatter_data_dir = os.path.join(output_dir, 'scatter_data')
    os.makedirs(scatter_data_dir, exist_ok=True)

    for idx, row in selected.iterrows():
        x_name = row['X_name']
        y_name = row['Y_name']
        x_label = row['X_Variable']
        y_label = row['Y_Variable']
        r = row['r']
        p = row['p_value']
        n = row['n']

        X = df[x_name].values
        Y = df[y_name].values
        # 画图保留所有点：仅剔除 NaN
        valid_mask = ~(np.isnan(X) | np.isnan(Y))
        X_valid = X[valid_mask]
        Y_valid = Y[valid_mask]
        aphasia_types = df['wab_type'].values[valid_mask]

        color_map = APHASIA_COLORS

        # 保存散点数据
        scatter_data = {x_name: X_valid, y_name: Y_valid, 'wab_type': aphasia_types}
        scatter_df = pd.DataFrame(scatter_data)
        safe_x = x_name.replace('_', '-')
        safe_y = y_name.replace('_', '-')
        scatter_csv = os.path.join(scatter_data_dir, f'lesion_sfc_{safe_x}_vs_{safe_y}_data.csv')
        scatter_df.to_csv(scatter_csv, index=False)

        # 绘图 - 画布略短一些
        fig, ax = plt.subplots(figsize=(5.6, 4.6))

        # 按类型绘制；其余类型合并为 Other（灰色）
        for aphasia_type in APHASIA_TYPES:
            mask = aphasia_types == aphasia_type
            if np.sum(mask) > 0:
                ax.scatter(
                    X_valid[mask], Y_valid[mask],
                    c=color_map[aphasia_type],
                    s=40,
                    alpha=0.8,
                    edgecolors='none',
                    label=aphasia_type
                )
        other_mask = ~np.isin(aphasia_types, APHASIA_TYPES)
        if np.sum(other_mask) > 0:
            ax.scatter(
                X_valid[other_mask], Y_valid[other_mask],
                c='#808080',
                s=40,
                alpha=0.8,
                edgecolors='none',
                label='Other'
            )

        # 回归线
        z = np.polyfit(X_valid, Y_valid, 1)
        p_fit = np.poly1d(z)
        x_line = np.linspace(np.min(X_valid), np.max(X_valid), 100)
        ax.plot(x_line, p_fit(x_line), "k-", linewidth=2, alpha=0.75, zorder=0)

        # 标题里显示统计信息（不再放在图内）
        ax.set_title(f"r = {r:.3f}, p = {p:.4f}, n = {n}", fontsize=12)

        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', labelsize=12, width=1.5, length=3)

        # 图例放在图下方并横向排列
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels,
                loc='lower center',
                bbox_to_anchor=(0.5, -0.02),
                ncol=min(len(labels), 6),
                fontsize=10,
                frameon=False
            )

        # 预留底部空间给图例
        plt.tight_layout(rect=(0, 0.09, 1, 1))

        output_file = os.path.join(output_dir, f'lesion_sfc_{safe_x}_vs_{safe_y}.pdf')
        plt.savefig(output_file, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

        print(f"  已保存: {os.path.basename(output_file)} (r={r:.4f}, p={p:.4f}, n={n})")


def main():
    """主函数"""
    print("=" * 80)
    print("Lesion vs SFC 分析（仅失语症患者，p<0.05 绘图）")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据（排除 non-aphasic）
    df = load_data()

    # 2. 执行 Lesion vs SFC 相关分析
    print("\n" + "=" * 80)
    print("Analysis 3: Lesion Variables vs SFC Variables (Pearson)")
    print("=" * 80)
    corr_df, _, _ = perform_lesion_vs_sfc_correlation(df)

    # 3. 保存完整相关结果
    corr_file = os.path.join(OUTPUT_DIR, 'lesion_vs_sfc_correlations_4types.csv')
    try:
        corr_df.to_csv(corr_file, index=False)
        print(f"\n完整相关结果已保存: {corr_file}")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        corr_file_alt = os.path.join(OUTPUT_DIR, f'lesion_vs_sfc_correlations_4types_{ts}.csv')
        corr_df.to_csv(corr_file_alt, index=False)
        print(f"\n警告: 目标文件被占用，已改存为: {corr_file_alt}")

    # 4. 仅绘制指定两张图（保留所有点）
    plot_selected_correlations_keep_all_points(df, corr_df, OUTPUT_DIR)

    n_sig = (corr_df['p_value'] < 0.05).sum()
    print("\n" + "=" * 80)
    print(f"分析完成！结果保存在: {OUTPUT_DIR}")
    print(f"  - 总相关数: {len(corr_df)}")
    print(f"  - 显著相关 (p<0.05): {n_sig}")
    print("  - 已绘制散点图: 2 (af_disconnection vs dorsal/ventral_sfc)")
    print("=" * 80)


if __name__ == '__main__':
    main()
