import os

import numpy as np
import scanpy as sc
from scipy import sparse

from dataset import SpatialDataManager

model_path = "BCA_pre"
section_id = "151507"

show_gene = ["GRB7","VIM","COL1A2","EMILIN1"]

#绘制ground_truth
data_manager = SpatialDataManager(selection_id =section_id, train_ratio=0.5, seed=42, neighbor_ratio=4)
gt_counts = np.log2(data_manager.count_mtx.loc[:, data_manager.selected_genes].values+1)
all_coords = data_manager.in_tissue_coords
adata_gt = sc.AnnData(gt_counts)
adata_gt.obsm["coord"] = all_coords
adata_gt.var_names = data_manager.selected_genes

# 绘制预测结果
# adata_sample = sc.read_h5ad(f"{model_path}/{section_id}_final_adata.h5ad")
adata_sample = sc.read_h5ad(f"{model_path}/E1_final_adata.h5ad")





#Visium分辨率画图
sc.set_figure_params(dpi=300, figsize=(2.8, 3))
sc.pl.embedding(adata_sample,
                basis="coord",
                color=show_gene,
                s=50,
                show=True,
                save=f"{section_id}_{model_path}_pre_4X.png",
                )

#HD分辨率画图
# sc.set_figure_params(dpi=300, figsize=(5.7, 4.6))
# sc.pl.embedding(adata_sample,
#                 basis="coord",
#                 color=show_gene,
#                 s=25,
#                 show=True,
#                 save=f"{section_id}_{model_path}.png",
#                 )
#Xneium分辨率画图
# sc.set_figure_params(dpi=300, figsize=(6.2, 3))
# sc.pl.embedding(adata_sample,
#                 basis="coord",
#                 color=show_gene,
#                 s=20,
#                 show=True,
#                 save=f"{section_id}_{model_path}.png",
#                 )

# sc.set_figure_params(dpi=300, figsize=(5.8, 4))
# sc.pl.embedding(adata_sample,
#                 basis="coord",
#                 color=show_gene,
#                 s=10,
#                 show=True,
#                 save=f"{section_id}_{model_path}.png",
#                 )
