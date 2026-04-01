import os
from pathlib import Path

import pandas as pd
import scanpy as sc
import anndata
import numpy as np
import torch
from GraphST.GraphST import GraphST
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from GraphST.utils import clustering

# ==== 参数配置 ====

palette = {
    'Layer_1': '#1f77b4',
    'Layer_2': '#ff7f0e',
    'Layer_3': '#49b192',
    'Layer_4': '#d62728',
    'Layer_5': '#aa40fc',
    'Layer_6': '#8c564b',
    'WM': '#e377c2',
}
palette1 = {
    0: '#1f77b4',
    1: '#ff7f0e',
    2: '#49b192',
    3: '#d62728',
    4: '#aa40fc',
    5: '#8c564b',
    6: '#e377c2',
}
palette2 = {
    1: '#1f77b4',
    2: '#ff7f0e',
    3: '#49b192',
    4: '#d62728',
    5: '#aa40fc',
    6: '#8c564b',
    7: '#e377c2',
}
# Turth Label
selection_id = '151673'
model_path = "HistoSGE_pre"
method = "GraphST"
num_clusters = 7   # 可根据模型自动设置聚类数
data_root = Path(f'D:\DL_project\diffusion_st_prediction_2\dataset\DLPFC/{selection_id}')

Turth = sc.read_visium(data_root , count_file=f"filtered_feature_bc_matrix.h5")
Turth.var_names_make_unique()
# 真实标签读取
truth_path = data_root / f"{selection_id}_truth.txt"
truth_df = pd.read_csv(truth_path, sep='\t', header=None, index_col=0)
truth_df.columns = ['Ground Truth']
Turth.obs['layer'] = truth_df.loc[Turth.obs_names, 'Ground Truth']

# ==== 读取数据 ====
adata = Turth
if isinstance(adata.X, np.matrix):
    adata.X = np.array(adata.X)

# ==== 聚类类别数 ====


print(adata.obsm["spatial"])
# 聚类绘图并保存

clusters=None
sample_adata = sc.read_h5ad(f"{model_path}/{selection_id}_final_adata.h5ad")
if method == "Kmeans":
    sc.pp.pca(sample_adata, n_comps=70)
    #使用 KMeans 聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    sample_adata.obs["cluster"] = kmeans.fit_predict(sample_adata.obsm["X_pca"])#使用基因主成分分析时这里填入X_pca
    sample_adata.obs["cluster"] = pd.Categorical(sample_adata.obs["cluster"])
    clusters = sample_adata.obs["cluster"]
# 使用层次聚类
# clustering = AgglomerativeClustering(n_clusters=num_clusters)
# labels = clustering.fit_predict(sample_adata.obsm["embedding"])
# sample_adata.obs["cluster"] = pd.Categorical(labels)

# GraphST  SEDR   STAGATE
elif method == "GraphST":
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['R_LANG'] = 'en_US.UTF-8'
    # 方法1: 使用原始字符串（推荐）
    os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.2"
    # Cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # n_clusters
    # define model
    model = GraphST(sample_adata, device=device)
    # train model
    sample_adata = model.train()
    # set radius to specify the number of neighbors considered during refinement
    radius = 50
    tool = 'mclust' # mclust, leiden, and louvain
    # clustering
    if tool == 'mclust':
       clustering(sample_adata, num_clusters, radius=radius, method=tool, refinement=True)
       # For DLPFC dataset, we use optional refinement step.
    elif tool in ['leiden', 'louvain']:
       clustering(sample_adata, num_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01, refinement=False)
    # 获取聚类标签
    clusters = sample_adata.obs['domain']
    sample_adata.obs['cluster']=sample_adata.obs['domain']
elif method == "STAGATE":
    os.environ['LANG'] = 'en_US.UTF-8'
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['R_LANG'] = 'en_US.UTF-8'
    # 方法1: 使用原始字符串（推荐）
    os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.2"
    # Cuda
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # n_clusters
    # define model
    from Cluster.STAGATE_pyG import Cal_Spatial_Net,Stats_Spatial_Net,Train_STAGATE
    import Cluster
    Cal_Spatial_Net(sample_adata, rad_cutoff=150)
    Stats_Spatial_Net(sample_adata)
    sample_adata =Train_STAGATE.train_STAGATE(sample_adata)
    # train model
    # set radius to specify the number of neighbors considered during refinement
    radius = 50
    tool = 'mclust'  # mclust, leiden, and louvain
    # clustering
    if tool == 'mclust':
        sample_adata.obsm['emb'] =  sample_adata.obsm['STAGATE']
        print(sample_adata.obsm['emb'].shape)
        clustering(sample_adata, num_clusters, radius=radius, method=tool, refinement=True)
        # For DLPFC dataset, we use optional refinement step.
    elif tool in ['leiden', 'louvain']:
        clustering(sample_adata, num_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01,
                   refinement=False)
    # 获取聚类标签
    clusters = sample_adata.obs['domain']
    sample_adata.obs['cluster'] = sample_adata.obs['domain']
elif method == "stMask":
    def train_one(args, adata, tissue_name=' '):
        net = stm.stMASK(adata,
                         tissue_name=tissue_name,
                         num_clusters=args.n_clusters,
                         genes_model='pca',
                         top_genes=args.top_genes,
                         rad_cutoff=200,
                         k_cutoff=args.k_cutoff,
                         graph_model='KNN',
                         device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
                         learning_rate=args.learning_rate,
                         weight_decay=args.weight_decay,
                         max_epoch=args.max_epoch,
                         gradient_clipping=args.gradient_clipping,
                         feat_mask_rate=args.feat_mask_rate,
                         edge_drop_rate=args.edge_drop_rate,
                         hidden_dim=args.hidden_dim,
                         latent_dim=args.latent_dim,
                         bn=args.bn,
                         att_dropout_rate=args.att_dropout_rate,
                         fc_dropout_rate=args.fc_dropout_rate,
                         use_token=args.use_token,
                         rep_loss=args.rep_loss,
                         rel_loss=args.rel_loss,
                         alpha=args.alpha,
                         lam=args.lam,
                         random_seed=args.seed,
                         nps=args.nps)
        net.train()
        method = "mclust"
        net.process(method=method)
        adata = net.get_adata()
        return adata


    from Cluster.stMask import stMask as stm
    from Cluster.stMask import utils  # ← Add this
    print(sample_adata)
    args = utils.build_args()

    args.hidden_dim, args.latent_dim = 512, 256
    args.max_epoch = 500
    args.lam = 1.3
    args.feat_mask_rate = 0.2
    args.edge_drop_rate = 0.2
    args.top_genes = 2000
    args.k_cutoff = 12
    args.n_clusters = 7
    sample_adata = train_one(args, sample_adata)
    print(sample_adata)
    clusters = sample_adata.obs['mclust']
    sample_adata.obs['cluster'] = sample_adata.obs['mclust']


#绘制可视化
coords = sample_adata.obsm["spatial"]
x, y = coords[:, 0], coords[:, 1]
# 顺时针旋转 90 度
# 顺时针 90° 等价于 (x, y) -> (y, -x)
x_rot, y_rot = y, -x
# 自己画散点图
plt.figure(figsize=(6, 6))  # 可调节图片比例，比如 (8,6)
for cluster_id in np.unique(clusters):
    idx = clusters == cluster_id
    cluster_num = int(float(cluster_id))
    plt.scatter(
        x_rot[idx], y_rot[idx],
        s=18,
        c=palette1.get(cluster_num, "#000000"),  # 将 cluster_id 转换为 int 类型
        label=f"Cluster {cluster_num}",
        alpha=0.8
    )

plt.gca().set_aspect('equal', adjustable='box')  # 保持比例
plt.axis("off")
# plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))  # 如果想显示图例
plt.title(f"{method} Clustering ", fontsize=14)
# 保存图片（可以是 png/jpg/pdf）
plt.savefig(f"figures/{model_path}_{selection_id}_{method}.png", dpi=300, bbox_inches="tight")
plt.close()

#计算ARI指数
if "coord" not in adata.obsm.keys():
    adata.obsm["coord"] = adata.obs[["array_row", "array_col"]].to_numpy()
if "coord" not in sample_adata.obsm.keys():
    sample_adata.obsm["coord"] = sample_adata.obs[["array_row", "array_col"]].to_numpy()
# ===== 提取坐标和标签 =====
coords1 = pd.DataFrame(
    adata.obsm["coord"],
    index=adata.obs_names,
    columns=["row", "col"]
)
true_labels = adata.obs["layer"]
coords2 = pd.DataFrame(
    sample_adata.obsm["coord"],
    index=sample_adata.obs_names,
    columns=["row", "col"]
)
pred_labels = sample_adata.obs["cluster"]
# ===== 用坐标对齐 =====
merged = coords1.join(true_labels).merge(
    coords2.join(pred_labels),
    on=["row", "col"],
    how="inner",
    suffixes=("_true", "_pred")
)
print(merged.head())
# ===== 取出对齐后的标签 =====
true_aligned = merged["layer"]
pred_aligned = merged["cluster"]
# 去掉 NaN
mask = (~true_aligned.isna()) & (~pred_aligned.isna())
true_aligned = true_aligned[mask]
pred_aligned = pred_aligned[mask]
# ===== 计算 ARI =====
ari = adjusted_rand_score(true_aligned, pred_aligned)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")