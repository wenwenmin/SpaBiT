import os
import gzip
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree
import scanpy as sc


# -------------------------
# Dataset（训练/测试用，按样本返回）
# -------------------------
class CustomDataset(Dataset):
    """
    返回按样本的子集，用于你的主干网络训练/测试（batch 方式）。
    多返回一个 global_index，方便根据全局索引从 GAT 预计算嵌入里取 cond。
    """
    def __init__(self, data, local_ebd, coords, spatials, neighbor_data, global_indices, labels=None):
        self.data = data                  # [N_sub, G]
        self.local_ebd = local_ebd        # [N_sub, C_img]
        self.coords = coords              # [N_sub, 2]
        self.spatials = spatials          # [N_sub, 2]
        self.neighbor_data = neighbor_data  # [N_sub, K, G]（log2 后）
        self.global_indices = global_indices  # [N_sub]，指向全图的索引
        self.labels = labels              # [N_sub] or None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return (
                self.data[idx],
                self.local_ebd[idx],
                self.coords[idx],
                self.spatials[idx],
                self.neighbor_data[idx],
                self.labels[idx],
                self.global_indices[idx],
            )
        else:
            return (
                self.data[idx],
                self.local_ebd[idx],
                self.coords[idx],
                self.spatials[idx],
                self.neighbor_data[idx],
                self.global_indices[idx],
            )


# -------------------------
# DataManager
# -------------------------
class SpatialDataManager:
    """
    四个对外函数：
      - get_gat_dataset()：训练/前向 GAT 用（全图一次性输入）
      - get_train_dataset()：主干网络训练用（子集 + batch）
      - get_test_dataset()：主干网络测试用（子集 + batch）
      - get_input_size()：输入维度信息（保持原有接口）
    读取部分分支逻辑完全保留。
    """
    def __init__(self, selection_id, train_ratio=0.5, seed=42, neighbor_ratio=4):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.slice_name = selection_id
        self.data_path = os.path.join(project_root, 'dataset', selection_id)
        self.gene_list_filename = "selected_gene_list.txt"
        self.train_ratio = train_ratio
        self.seed = seed
        self.neighbors = neighbor_ratio  # KNN 邻居个数（你要求的 4）
        np.random.seed(self.seed)

        # 加载基础数据（读取分支保持不动）
        self._load_base_data()

        # —— 读取后：构建全图 KNN 邻接、为 GAT 构造全图特征（训练 spot 用真值，测试 spot 用 4 邻训练均值）
        self._build_global_graph_inputs()

    # ==========================
    # 1) 读取（分支保持不动）
    # ==========================
    def _load_base_data(self):
        """加载并预处理基础数据（⚠️ 分支逻辑不动）"""
        self.labels = None

        if self.slice_name =='MP' or self.slice_name =='HS'or self.slice_name =='HBC_HD'\
            or self.slice_name =='MB_HD'or self.slice_name =='MB_HD_train'or self.slice_name =='HBC_HD_train'\
            or self.slice_name == 'HL_X'or self.slice_name == 'HL_X2'or self.slice_name =='HL_X_train'or self.slice_name =='HL_X2_train':
            self.count_mtx = pd.read_csv(os.path.join(self.data_path, "counts.csv"), index_col=0)

        elif self.slice_name == 'Alex'or self.slice_name == 'Alex2'or self.slice_name == 'Alex3':
            matrix_dir = os.path.join(self.data_path, 'filtered_count_matrix')
            features_path = os.path.join(matrix_dir, 'features.tsv.gz')
            barcodes_path = os.path.join(matrix_dir, 'barcodes.tsv.gz')
            matrix_path = os.path.join(matrix_dir, 'matrix.mtx.gz')

            with gzip.open(features_path, 'rt') as f:
                features = pd.read_csv(f, sep='\t', header=None)
            if features.shape[1] == 1:
                features[1] = features[0]
                features[2] = "Gene Expression"

            adata = sc.read_mtx(matrix_path).T  # spots × genes
            with gzip.open(barcodes_path, 'rt') as f:
                barcodes = [line.strip() for line in f]
            adata.obs_names = barcodes
            adata.var_names = features[1].values

            # 归一化 + log1p 预处理
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            self.count_mtx = pd.DataFrame(
                adata.X.toarray(),
                index=adata.obs_names,
                columns=adata.var_names
            )

            label_path = os.path.join(self.data_path, "label.txt")
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    labels = [int(x) for x in f.read().strip().split()]
                    self.labels = np.array(labels, dtype=int)
                print(f"[INFO] Loaded {len(self.labels)} labels from {label_path}")
            else:
                print(f"[WARNING] Label file not found at {label_path}")
                self.labels = None

        else:
            adata = sc.read_10x_h5(os.path.join(self.data_path, "filtered_feature_bc_matrix.h5"))
            adata.var_names_make_unique()
            self.count_mtx = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)

        # ====== 位置文件（保持逻辑不动）======
        if self.slice_name=='MBSP' or self.slice_name == '151507'or self.slice_name == 'AMBC'or self.slice_name == 'HOC'or self.slice_name == 'HCC'\
                or self.slice_name == 'HIC'or self.slice_name == 'MBC'or self.slice_name == 'AMOB'or self.slice_name == 'VIHBC'\
                or self.slice_name == 'Alex'or self.slice_name == 'Alex2'or self.slice_name == 'Alex3':
            position_df = pd.read_csv(os.path.join(self.data_path, 'spatial/tissue_positions_list.csv'), header=None)
        elif self.slice_name == 'MP'or self.slice_name =='F1'or self.slice_name =='HS'or self.slice_name =='HBC_HD'or self.slice_name =='MB_HD'\
                or self.slice_name =='MB_HD_train'or self.slice_name =='HBC_HD_train'or self.slice_name == 'HL_X'\
                or self.slice_name == 'HL_X2'or self.slice_name =='HL_X_train'or self.slice_name =='HL_X2_train':
            position_df = pd.read_csv(os.path.join(self.data_path, 'tissue_positions_list.csv'), header=None)
        else:
            position_df = pd.read_csv(os.path.join(self.data_path, 'spatial/tissue_positions_list.csv'), header=None, skiprows=1)

        valid_spots = position_df[position_df[1] == 1][0].values
        # 只保留在组织中的 spot
        self.count_mtx = self.count_mtx.loc[valid_spots]

        # 基因列表
        self.selected_genes = list(
            np.genfromtxt(os.path.join(self.data_path, "processed", self.gene_list_filename), dtype=str)
        )

        # 坐标
        self.spot_coords = position_df[position_df[1] == 1][[2, 3]].values.astype(int)   # 网格坐标
        self.spot_spatials = position_df[position_df[1] == 1][[4, 5]].values.astype(int) # 像素坐标
        all_coords = np.array(position_df[[2, 3]].values).astype(int)  # 仅用于下采样网格

        # ====== 下采样划分训练/测试（保持你原逻辑）======
        n_spots = len(self.count_mtx)
        self.in_tissue_coords = np.array(self.spot_coords)
        delta_x = 1
        if self.slice_name == 'E1'or self.slice_name == 'MP'or self.slice_name =='F3'or self.slice_name =='HS':
            delta_y = 1
        else:
            delta_y = 2
        x_min = min(all_coords[:, 0]) + min(all_coords[:, 0]) % 2
        y_min = min(all_coords[:, 1]) + min(all_coords[:, 1]) % 2
        lr_x, lr_y = np.mgrid[x_min:max(all_coords[:, 0]) + delta_x:2 * delta_x,
                              y_min:max(all_coords[:, 1]):2 * delta_y]
        lr_spot_index = []
        lr_xy = [list(i) for i in list(np.vstack((lr_x.reshape(-1), lr_y.reshape(-1))).T)]
        for i in range(self.in_tissue_coords.shape[0]):
            if list(self.in_tissue_coords[i]) in lr_xy:
                lr_spot_index.append(i)
        self.train_mask = np.zeros(n_spots, dtype=bool)
        self.train_mask[lr_spot_index] = True

        # 图像局部特征
        self.img_local_ebd = torch.load(os.path.join(self.data_path, "processed/local_ebd.pt"), map_location="cpu")
        gat_path = os.path.join(self.data_path, "processed", "gat_ebd.pt")
        if os.path.exists(gat_path):
            self.neighbor_ebd = torch.load(gat_path, map_location="cpu")
            print(f"[INFO] Loaded precomputed GAT embeddings from {gat_path}")
        else:
            print(f"[WARNING] GAT embedding file not found at {gat_path}. Using zeros instead.")
            # 用与 local_ebd 相同形状的零张量代替，保持维度一致
            self.neighbor_ebd = None

        # 建一个全局索引数组，后面切子集时要用
        self.global_indices = np.arange(n_spots, dtype=int)

    # ==========================
    # 2) 全图输入：GAT 用
    # ==========================
    def _build_global_graph_inputs(self):
        """
        构建：
          - self.gat_features: 全图节点特征（训练用原值，测试用4邻训练均值），再 log2(x+1)
          - self.gat_adj: 全图 0/1 邻接（每点连接 4 个最近邻，无向）
        """
        # --- 基于 selected_genes 的表达矩阵 ---
        all_expr = self.count_mtx[self.selected_genes].values  # [N, G]
        train_expr = all_expr[self.train_mask]                  # [N_tr, G]

        # --- 对测试集，用 4 个训练邻居的均值填充 ---
        K = self.neighbors  # 一般为 4
        K = min(K, train_expr.shape[0]) if train_expr.shape[0] > 0 else 0
        gat_features = np.zeros_like(all_expr)

        gat_features[self.train_mask] = train_expr
        if K > 0:
            tree_tr = KDTree(self.spot_coords[self.train_mask])
            dists, nidx = tree_tr.query(self.spot_coords[~self.train_mask], k=K)
            neighbor_expr = train_expr[nidx]                  # [N_te, K, G]
            test_filled = neighbor_expr.mean(axis=1)          # [N_te, G]
            gat_features[~self.train_mask] = test_filled
        else:
            # 极端情况下（没有训练点），退化为原值
            gat_features[~self.train_mask] = all_expr[~self.train_mask]

        # log2(x+1)
        gat_features = np.log2(gat_features + 1.0).astype(np.float32)
        self.gat_features = torch.from_numpy(gat_features)     # [N, G]

        # --- 全图 KNN=4 邻接（无向，0/1） ---
        N = self.spot_coords.shape[0]
        tree_all = KDTree(self.spot_coords)
        # k = 1(自身) + 4(邻居)
        _, knn_idx = tree_all.query(self.spot_coords, k=min(1 + self.neighbors, N))
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in knn_idx[i, 1:]:   # 跳过自身
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        self.gat_adj = torch.from_numpy(adj)                   # [N, N]，0/1，无向

    # ==========================
    # 3) GAT 训练/前向专用
    # ==========================
    def get_gat_dataset(self):
        """
        返回一个“全图包”，用于 GAT 训练或一次性前向编码。
        - features: [N, G]（训练原值，测试 = 4 邻训练均值，log2 后）
        - adj: [N, N] 0/1 无向邻接
        - coords: [N, 2]
        - spatials: [N, 2]
        - train_mask: [N] bool（保留，方便需要时做监督或评估）
        - labels: [N] or None
        - global_indices: [N] (0..N-1)
        """
        pkg = {
            "features": self.gat_features.clone(),                    # torch.FloatTensor
            "adj": self.gat_adj.clone(),                              # torch.FloatTensor (0/1)
            "coords": torch.from_numpy(self.spot_coords.copy()).long(),
            "spatials": torch.from_numpy(self.spot_spatials.copy()).long(),
            "train_mask": torch.from_numpy(self.train_mask.copy()),
            "labels": torch.from_numpy(self.labels.copy()).long() if self.labels is not None else None,
            "global_indices": torch.from_numpy(self.global_indices.copy()).long(),
            "selected_genes": self.selected_genes,                    # 便于外部核对
        }
        return pkg

    # ==========================
    # 4) 主干训练集（batch）
    # ==========================
    def get_train_dataset(self):
        """
        训练子集样本（主干网络训练用）：
          - data: 训练 spot 的（log2 后）真实表达
          - local_ebd: 图像特征 local_ebd.pt
          - neighbor_data: GAT 输出的图结构嵌入 gat_ebd.pt
          - coords / spatials: 坐标信息
          - global_index: 训练子集在全图中的索引
          - labels: 若存在
        """
        train_mask = self.train_mask
        train_spots = self.count_mtx.index[train_mask]
        train_count = self.count_mtx.loc[train_spots, self.selected_genes]  # [N_tr, G]
        train_coords = self.spot_coords[train_mask]
        train_spatials = self.spot_spatials[train_mask]
        train_img_local = self.img_local_ebd[train_mask]
        train_indices = self.global_indices[train_mask]

        # log2 标准化
        data_log = np.log2(train_count.values.astype(np.float32) + 1.0).astype(np.float32)

        # ✅ 关键：neighbor_data 改为 GAT 输出的嵌入
        neighbor_emb = self.neighbor_ebd[train_mask]

        train_labels = None
        if self.labels is not None:
            train_labels = torch.tensor(self.labels[train_mask], dtype=torch.long)

        return CustomDataset(
            data=torch.from_numpy(data_log),
            local_ebd=train_img_local.float(),
            coords=torch.from_numpy(train_coords).long(),
            spatials=torch.from_numpy(train_spatials).long(),
            neighbor_data=neighbor_emb.float(),  # 👈 改为 GAT embedding
            global_indices=torch.from_numpy(train_indices).long(),
            labels=train_labels
        )

    # ==========================
    # 5) 主干测试集（batch）
    # ==========================
    def get_test_dataset(self):
        """
        测试子集样本：
          - data: 测试 spot 的 log2 基因表达（真实值，用于评估）
          - neighbor_data: GAT 输出的图嵌入 gat_ebd.pt
          - local_ebd, coords, spatials 同上
        """
        train_mask = self.train_mask
        test_mask = ~train_mask
        test_spots = self.count_mtx.index[test_mask]
        test_count = self.count_mtx.loc[test_spots, self.selected_genes]  # [N_te, G]
        test_coords = self.spot_coords[test_mask]
        test_spatials = self.spot_spatials[test_mask]
        test_img_local = self.img_local_ebd[test_mask]
        test_indices = self.global_indices[test_mask]

        # log2 标准化
        data_log = np.log2(test_count.values.astype(np.float32) + 1.0).astype(np.float32)

        # ✅ 同样替换 neighbor_data 为 GAT 嵌入
        neighbor_emb = self.neighbor_ebd[test_mask]

        test_labels = None
        if self.labels is not None:
            test_labels = torch.tensor(self.labels[test_mask], dtype=torch.long)

        return CustomDataset(
            data=torch.from_numpy(data_log),
            local_ebd=test_img_local.float(),
            coords=torch.from_numpy(test_coords).long(),
            spatials=torch.from_numpy(test_spatials).long(),
            neighbor_data=neighbor_emb.float(),  # 👈 改为 GAT embedding
            global_indices=torch.from_numpy(test_indices).long(),
            labels=test_labels
        )

    # ==========================
    # 6) 维度信息（保持接口）
    # ==========================
    def get_input_size(self):
        return {
            'input_gene_size': len(self.selected_genes),
            'cond_size': self.img_local_ebd.shape[1]
        }

    # 保持你原来的接口（可选）
    def get_test_spots(self):
        return self.count_mtx.index[~self.train_mask].tolist()
