import os

import numpy as np
import scanpy as sc
from scipy import sparse

from dataset import SpatialDataManager

model_path = "BCA_pre"
section_id = "E1"
# show_gene = ["CALM2","CALM3","GFAP","ENC1","CKB"]  #151507
# show_gene = ["Cryab","Rgs9","Tmeff2","Ptprn"]  #MB_HD
# show_gene = ["Slc17a7", "Nrn1", "Hpca"]  # MB_HD
# show_gene = ["PNMT","PIP","TSKU","CLDN4"]  #HBC_HD
# show_gene = ["GPC3","VTN","AKR1C3"]    #HL_X

# show_gene = ["ADH4","CFB","GATM"]     #HL_X2
# show_gene = ["COX5B","COX6B1","TPI1","GUK1","MZT2B"]
# show_gene = ["PARK7","LZIC","RBP7","EXOSC10","SDHB"]
show_gene = ["GRB7","VIM","COL1A2","EMILIN1"]

#绘制ground_truth
# data_manager = SpatialDataManager(selection_id =section_id, train_ratio=0.5, seed=42, neighbor_ratio=4)
# gt_counts = np.log2(data_manager.count_mtx.loc[:, data_manager.selected_genes].values+1)
# all_coords = data_manager.in_tissue_coords
# adata_gt = sc.AnnData(gt_counts)
# adata_gt.obsm["coord"] = all_coords
# adata_gt.var_names = data_manager.selected_genes

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
# sc.set_figure_params(dpi=400, figsize=(6.2, 3))
# sc.pl.embedding(adata_sample,
#                 basis="coord",
#                 color=show_gene,
#                 s=20,
#                 show=True,
#                 save=f"{section_id}_{model_path}.png",
#                 )

# sc.set_figure_params(dpi=400, figsize=(5.8, 4))
# sc.pl.embedding(adata_sample,
#                 basis="coord",
#                 color=show_gene,
#                 s=10,
#                 show=True,
#                 save=f"{section_id}_{model_path}.png",
#                 )