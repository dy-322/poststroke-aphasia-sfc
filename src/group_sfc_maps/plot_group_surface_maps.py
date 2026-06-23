import matplotlib
matplotlib.use('Qt5Agg')

# 设置全局字体为 Arial
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['pdf.fonttype'] = 42  # 确保 PDF 中字体可编辑

import matplotlib

matplotlib.use('Qt5Agg')


def merge_images(images_dict, output_file, trim_edges=False):
    """
    Merges a list of images into a single figure and saves it, based on provided image information.

    Parameters:
    - images_dict: List of dictionaries, each containing 'file_path' and 'title' for each image.
    - output_file: Path where the merged image should be saved.
    - trim_edges: Whether to trim white edges from the merged image before saving.
    # Example usage:
        images_dict = [
            {'file_path': os.path.join(plot_path, 'CT_d_all.png'), 'title': 'Cortical'},
            {'file_path': os.path.join(plot_path, 'SV_d_all.png'), 'title': 'Subcortical'}]
        output_file = os.path.join(plot_path, 'CT_SV_d_all.png')
        merge_images(images_dict, output_file, trim_edges=True)
    """
    n_images = len(images_dict)
    fig, axs = plt.subplots(n_images, 1, figsize=(12, 7 * n_images))

    for i, img_info in enumerate(images_dict):
        img = mpimg.imread(img_info['file_path'])
        axs[i].imshow(img)
        axs[i].axis('off')
        title_text = img_info.get('title', '')
        axs[i].set_title(title_text, y=0.7, fontdict={'family': 'sans-serif', 'fontsize': 18}, )
        # Adding a subtle marker (barely visible)
        axs[i].text(0.5, 0.65, '|', fontsize=12, color=(0.995, 0.995, 0.995), ha='center', transform=axs[i].transAxes)
        axs[i].text(0.5, 0.75, '|', fontsize=22, color=(0.995, 0.995, 0.995), ha='center', transform=axs[i].transAxes)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight', dpi=300)

    if trim_edges:
        # Load the saved image, trim edges, and save again
        merged_img = mpimg.imread(output_file)
        trimmed_img = trim_white_edges(merged_img)
        plt.imsave(output_file, np.ascontiguousarray(trimmed_img), dpi=300)


def trim_white_edges(img_array):
    """
    Trim white edges from an image with pixel values normalized to the 0 to 1 range.

    Parameters:
    - img_array: Image array with pixel values in the range [0, 1].

    Returns:
    - Cropped image array with white edges removed.
    """
    # Detect pixels that are not completely white (assuming white is [1, 1, 1])
    mask = np.any(img_array < 1, axis=-1)

    # Find the rows and columns that contain non-white pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Find the boundaries
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        # If no non-white pixels found, return original
        return img_array

    # Crop using slicing (more efficient and preserves dtype)
    cropped_img = img_array[row_indices[0]:row_indices[-1] + 1,
                  col_indices[0]:col_indices[-1] + 1]

    # Ensure contiguous array and correct dtype
    return np.ascontiguousarray(cropped_img)


import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.colorbar import ColorbarBase
from matplotlib.ticker import MaxNLocator
import matplotlib.image as mpimg
import os
import tempfile
from surfplot import Plot
from brainspace.datasets import load_parcellation
from brainspace.mesh.mesh_io import read_surface
from neuromaps.datasets import fetch_fslr
from nilearn import datasets, plotting
from neurocat.plotting import draw_and_save as ds

folder_path = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\single_subject_results_t288"
atlas_path = r"D:\download\Schaefer2018_400Parcels_7Networks_order_Tian_Subcortex_S3_MNI152NLin6Asym_1mm.nii.gz"

for file in os.listdir(folder_path):
    if "txt" in file and "average" in file:
        sfc_txt_path = os.path.join(folder_path, file)
    else:
        continue
    # sfc_txt_path = r"D:\sc-fc_coupling\scfc-coupling-main\scfc-coupling-main\single_subject_results_1\Anomic_average_sfc.txt"
    print(sfc_txt_path)
    sfc_values_raw = np.loadtxt(sfc_txt_path, skiprows=1, usecols=1)
    sfc_first50 = sfc_values_raw[:50]
    sfc_rest = sfc_values_raw[50:]

    global_vmin = 0.15
    global_vmax = 0.5

    atlas_img = nib.load(atlas_path)
    atlas_data = atlas_img.get_fdata().astype(int)
    roi_vol = np.full(atlas_data.shape, np.nan, dtype=float)
    for lbl in range(1, 51):
        roi_vol[atlas_data == lbl] = sfc_first50[lbl - 1]
    roi_img = nib.Nifti1Image(roi_vol, atlas_img.affine, atlas_img.header)

    bg_img = datasets.load_mni152_template(resolution=1)
    bg_data = bg_img.get_fdata()
    fig_nilearn = plt.figure(figsize=(40, 8))

    bg_img_final = nib.Nifti1Image(bg_data,
                                   bg_img.affine,
                                   bg_img.header)
    nilearn_disp = plotting.plot_stat_map(
        roi_img,
        bg_img=bg_img_final,
        threshold=0.0,
        cmap='RdBu_r',
        vmax=global_vmax,
        vmin=global_vmin,
        display_mode='z',
        cut_coords=[-25, -20, -10, 0, 10, 20, 22],
        draw_cross=False,
        colorbar=False,
        figure=fig_nilearn,
        black_bg=False,
        annotate=False,
    )

    temp_dir = tempfile.gettempdir()
    temp_nilearn = os.path.join(temp_dir, 'temp_nilearn.png')
    fig_nilearn.savefig(temp_nilearn, bbox_inches='tight', dpi=300, pad_inches=0)

    img_nilearn_temp = mpimg.imread(temp_nilearn)
    nilearn_width = img_nilearn_temp.shape[1]
    nilearn_height = img_nilearn_temp.shape[0]

    color_range = (global_vmin, global_vmax)

    temp_surface = os.path.join(temp_dir, 'temp_surface.png')

    surface_height = int(nilearn_width * 0.2)
    surface_size = (nilearn_width, surface_height)

    ds(
        layer1=sfc_rest,  # 400 个 cortical ROI，ds() 会自动识别并转换为全脑数据
        colorbar='RdBu_r',  # 使用与原来相同的颜色映射
        color_range=color_range,
        fig_name=temp_surface.replace('.png', ''),  # 去掉扩展名，函数会自动添加
        layout='row',  # 保持原来的 row 布局
        outline=True,  # 添加轮廓（类似原来的 atlas outline）
        trim=True,  # 修剪白边
        tp='fslr',  # 模板类型
        mesh='veryinflated',  # 网格类型
        save_png_only=True,  # 只保存 PNG，不生成 PDF，不包含 colorbar
        display=False,  # 不自动弹出
        size=(6000, 1000)  # 自定义尺寸以匹配 nilearn 的实际宽度
    )

    img_surface = mpimg.imread(temp_surface)
    img_nilearn = mpimg.imread(temp_nilearn)

    img_surface_trimmed = trim_white_edges(img_surface)
    img_nilearn_trimmed = trim_white_edges(img_nilearn)

    fig_merged = plt.figure(figsize=(70, 25))
    gs = gridspec.GridSpec(2, 2, figure=fig_merged,
                           width_ratios=[97, 3],
                           hspace=0.1, wspace=0.05)

    ax1 = fig_merged.add_subplot(gs[0, 0])
    ax1.imshow(img_surface_trimmed)
    ax1.axis('off')

    ax2 = fig_merged.add_subplot(gs[1, 0])
    ax2.imshow(img_nilearn_trimmed)
    ax2.axis('off')

    ax_cbar = fig_merged.add_subplot(gs[:, 1])  # 跨越所有行
    norm = mpl.colors.Normalize(vmin=global_vmin, vmax=global_vmax)
    cbar = ColorbarBase(ax_cbar, cmap='RdBu_r', norm=norm,
                        orientation='vertical')

    # 计算中间值并手动设置刻度
    mid_val = (global_vmin + global_vmax) / 2
    ax_cbar.set_yticks([global_vmin, global_vmax])

    cbar.set_label('SFC Values', rotation=270, labelpad=30, fontsize=70, fontname='Arial')
    ax_cbar.tick_params(labelsize=50, direction='in', length=8, width=80)
    # 设置 colorbar 刻度标签字体
    for label in ax_cbar.get_yticklabels():
        label.set_fontname('Arial')

    plt.tight_layout()

    output_path = os.path.join(folder_path, f"{file.split('_')[0]}_surface_t288.pdf")
    # output_path = os.path.join(folder_path, f"anomic_surface.pdf")
    fig_merged.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.show()

    os.remove(temp_surface)
    os.remove(temp_nilearn)
