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


class SpatialDataManager:

    def __init__(self, selection_id, train_ratio=0.5, seed=42, neighbor_ratio=4):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.slice_name = selection_id
        self.data_path = os.path.join(project_root, 'dataset', selection_id)
        self.gene_list_filename = "selected_gene_list.txt"
        self.train_ratio = train_ratio
        self.seed = seed
        self.neighbors = neighbor_ratio  
        np.random.seed(self.seed)

        self._load_base_data()

        self._build_global_graph_inputs()


    def _load_base_data(self):
     
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
 
        self.count_mtx = self.count_mtx.loc[valid_spots]


        self.selected_genes = list(
            np.genfromtxt(os.path.join(self.data_path, "processed", self.gene_list_filename), dtype=str)
        )

        
        self.spot_coords = position_df[position_df[1] == 1][[2, 3]].values.astype(int)   # 网格坐标
        self.spot_spatials = position_df[position_df[1] == 1][[4, 5]].values.astype(int) # 像素坐标
        all_coords = np.array(position_df[[2, 3]].values).astype(int)  # 仅用于下采样网格

  
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

 
        self.img_local_ebd = torch.load(os.path.join(self.data_path, "processed/local_ebd.pt"), map_location="cpu")
        gat_path = os.path.join(self.data_path, "processed", "gat_ebd.pt")
        if os.path.exists(gat_path):
            self.neighbor_ebd = torch.load(gat_path, map_location="cpu")
            print(f"[INFO] Loaded precomputed GAT embeddings from {gat_path}")
        else:
            print(f"[WARNING] GAT embedding file not found at {gat_path}. Using zeros instead.")

            self.neighbor_ebd = None

        self.global_indices = np.arange(n_spots, dtype=int)


    def _build_global_graph_inputs(self):
      
        all_expr = self.count_mtx[self.selected_genes].values  # [N, G]
        train_expr = all_expr[self.train_mask]                  # [N_tr, G]

        K = self.neighbors  # 一般为 4
        K = min(K, train_expr.shape[0]) if train_expr.shape[0] > 0 else 0
        gat_features = np.zeros_like(all_expr)

        gat_features[self.train_mask] = train_expr
        tree_tr = KDTree(self.spot_coords[self.train_mask])
        dists, nidx = tree_tr.query(self.spot_coords[~self.train_mask], k=K)
        neighbor_expr = train_expr[nidx]                  # [N_te, K, G]
        test_filled = neighbor_expr.mean(axis=1)          # [N_te, G]
        gat_features[~self.train_mask] = test_filled

        # log2(x+1)
        gat_features = np.log2(gat_features + 1.0).astype(np.float32)
        self.gat_features = torch.from_numpy(gat_features)     # [N, G]


        N = self.spot_coords.shape[0]
        tree_all = KDTree(self.spot_coords)

        _, knn_idx = tree_all.query(self.spot_coords, k=min(1 + self.neighbors, N))
        adj = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j in knn_idx[i, 1:]:  
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        self.gat_adj = torch.from_numpy(adj)             

    def get_gat_dataset(self):
       
        pkg = {
            "features": self.gat_features.clone()
            "coords": torch.from_numpy(self.spot_coords.copy()).long(),
            "spatials": torch.from_numpy(self.spot_spatials.copy()).long(),
            "train_mask": torch.from_numpy(self.train_mask.copy()),
            "labels": torch.from_numpy(self.labels.copy()).long() if self.labels is not None else None,
            "global_indices": torch.from_numpy(self.global_indices.copy()).long(),
            "selected_genes": self.selected_genes,                    # 便于外部核对
        }
        return pkg

 
    def get_train_dataset(self):
        
        train_mask = self.train_mask
        train_spots = self.count_mtx.index[train_mask]
        train_count = self.count_mtx.loc[train_spots, self.selected_genes]  # [N_tr, G]
        train_coords = self.spot_coords[train_mask]
        train_spatials = self.spot_spatials[train_mask]
        train_img_local = self.img_local_ebd[train_mask]
        train_indices = self.global_indices[train_mask]

        data_log = np.log2(train_count.values.astype(np.float32) + 1.0).astype(np.float32)
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

   
    def get_test_dataset(self):
       
        train_mask = self.train_mask
        test_mask = ~train_mask
        test_spots = self.count_mtx.index[test_mask]
        test_count = self.count_mtx.loc[test_spots, self.selected_genes]  # [N_te, G]
        test_coords = self.spot_coords[test_mask]
        test_spatials = self.spot_spatials[test_mask]
        test_img_local = self.img_local_ebd[test_mask]
        test_indices = self.global_indices[test_mask]


        data_log = np.log2(test_count.values.astype(np.float32) + 1.0).astype(np.float32)
        neighbor_emb = self.neighbor_ebd[test_mask]

        test_labels = None
        if self.labels is not None:
            test_labels = torch.tensor(self.labels[test_mask], dtype=torch.long)

        return CustomDataset(
            data=torch.from_numpy(data_log),
            local_ebd=test_img_local.float(),
            coords=torch.from_numpy(test_coords).long(),
            spatials=torch.from_numpy(test_spatials).long(),
            neighbor_data=neighbor_emb.float(), 
            global_indices=torch.from_numpy(test_indices).long(),
            labels=test_labels
        )

    def get_input_size(self):
        return {
            'input_gene_size': len(self.selected_genes),
            'cond_size': self.img_local_ebd.shape[1]
        }

    def get_test_spots(self):
        return self.count_mtx.index[~self.train_mask].tolist()


