import gzip

import torch
from PIL import Image
import pandas as pd
import numpy as np
import h5py
import scanpy as sc
import os
from UNI.uni.get_encoder.get_encoder import get_encoder
from huggingface_hub import login
from conch.open_clip_custom import create_model_from_pretrained
import json
from sklearn.neighbors import KDTree
from tqdm import tqdm
Image.MAX_IMAGE_PIXELS = None  # 取消限制

class DLPFCProcessor:
    def __init__(self, slice_id):
        self.slice_id = slice_id
        self.data_dir = f'dataset/DLPFC/{slice_id}'
        self.spatial_dir = os.path.join(self.data_dir, 'spatial')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)

    def get_spot_coords(self):
        """读取position文件获取spot坐标映射"""
        position_df = pd.read_csv(os.path.join(self.spatial_dir, 'tissue_positions_list.csv'), header=None)
        # 创建spot名称到坐标的映射
        spot_to_coord = {}
        for _, row in position_df.iterrows():
            spot_name = row[0]  # 第一列是spot的barcode
            coord = (int(row[4]), int(row[5]))  # 第5、6列是像素坐标
            spot_to_coord[spot_name] = coord
        return spot_to_coord

    def process_gene_list(self):
        """处理基因列表"""
        # 读取h5文件
        adata = sc.read_10x_h5(os.path.join(self.data_dir, 'filtered_feature_bc_matrix.h5'))
        count_mtx = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)

        # 数据预处理
        # 1. 过滤掉表达量为0的基因
        genes_with_expr = count_mtx.columns[count_mtx.sum() > 0]
        filtered_df = count_mtx[genes_with_expr]

        # 2. 标准化处理
        norm_df = filtered_df.div(filtered_df.sum(axis=1), axis=0) * 10000
        norm_df = np.log1p(norm_df)

        # 3. 计算每个基因的均值和标准差
        gene_means = norm_df.mean()
        gene_stds = norm_df.std()

        # 4. 选择高变基因
        genes_by_mean = gene_means.sort_values(ascending=False)
        genes_by_std = gene_stds.sort_values(ascending=False)

        num_genes = 1000
        high_mean_genes = set(genes_by_mean.head(num_genes).index)
        high_std_genes = set(genes_by_std.head(num_genes).index)

        selected_genes = sorted(list(high_mean_genes.intersection(high_std_genes)))
        selected_genes = [gene for gene in selected_genes
                        if not gene.startswith(("MT-", "mt-", "RP", "rp"))]
        selected_genes = selected_genes[:300]
        # adata = sc.read_10x_h5(os.path.join(self.data_dir, 'filtered_feature_bc_matrix.h5'))


        # # 选取高变基因，使用 Seurat v3 方法
        # adata.var_names_make_unique()
        # sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
        # #
        # # # 获取高变基因名
        # selected_genes = adata.var[adata.var['highly_variable']].index.tolist()
        #
        # # 可选：排除线粒体和核糖体基因
        # selected_genes = [gene for gene in selected_genes if not gene.startswith(("MT-", "mt-", "RP", "rp"))]
        #
        # # 如果过滤后不足300个，可补充（按方差排序）
        if len(selected_genes) < 300:
            remaining_genes = adata.var.loc[~adata.var_names.isin(selected_genes)]
            additional_genes = remaining_genes.sort_values("variances", ascending=False).index.tolist()
            selected_genes += [g for g in additional_genes if not g.startswith(("MT-", "mt-", "RP", "rp"))]
            selected_genes = selected_genes[:300]

        print(f"Selected {len(selected_genes)} highly variable genes")
        
        # 保存基因列表
        gene_list_path = os.path.join(self.processed_dir, 'selected_gene_list.txt')
        with open(gene_list_path, 'w') as f:
            for gene in selected_genes:
                f.write(f"{gene}\n")

        #为SpaVit模型筛选训练基因
        # all_genes_sorted = genes_by_std.index.tolist()
        # selected_indices = [all_genes_sorted.index(gene) for gene in selected_genes if gene in all_genes_sorted]
        #
        # max_idx = max(selected_indices) if selected_indices else 0
        # next_300_genes = []
        #
        # for gene in all_genes_sorted[max_idx + 1:]:
        #     if gene not in selected_genes and not gene.startswith(("MT-", "mt-", "RP", "rp")):
        #         next_300_genes.append(gene)
        #         if len(next_300_genes) >= 300:
        #             break
        #
        # train_gene_list_path = os.path.join(self.processed_dir, 'train_selected_gene_list.txt')
        # with open(train_gene_list_path, 'w') as f:
        #     for gene in next_300_genes:
        #         f.write(f"{gene}\n")
        #
        # print(f"Selected {len(next_300_genes)} genes for training extension set")
        
        return selected_genes

    def process_embeddings(self):
        """处理图像嵌入"""
        # Load image
        image = Image.open(os.path.join(self.spatial_dir, f'{self.slice_id}_full_image.tif'))

        # 首先读取position文件来确定有效spots的顺序
        position_df = pd.read_csv(os.path.join(self.spatial_dir, 'tissue_positions_list.csv'), header=None, skiprows=1)
        valid_spots = position_df[position_df[1] == 1][0].values  # 这样确保与数据加载类中的顺序一致
        coords = position_df[position_df[1] == 1][[4, 5]].values.astype(int)  # 使用像素坐标

        # 构建KDTree用于邻居搜索
        tree = KDTree(coords)

        # Initialize device and model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        token = ""
        login(token=token, add_to_git_credential=False)
        model_UNI, transform_UNI = get_encoder(enc_name='uni', device=device)

        # Initialize embeddings storage
        all_local_ebd = []
        neighbor_ebd = []
        global_ebd = []

        # all_local_aug_ebd = []
        # neighbor_aug_ebd = []
        # global_aug_ebd = []

        # 按照valid_spots的顺序处理每个spot
        print("Generating local embeddings...")
        for spot_idx, spot in enumerate(tqdm(valid_spots)):
            x, y = coords[spot_idx]  # 直接使用对应索引的坐标
            patch = crop_patch(image, x, y)

            # Get regular and augmented embeddings
            local_emb = get_img_embd_uni(patch, model_UNI, transform_UNI, device)
            # aug_emb = patch_augmentation_embd(patch, model_UNI, transform_UNI, device)

            all_local_ebd.append(local_emb)
            # all_local_aug_ebd.append(aug_emb)

        # 将local embeddings转换为tensor
        all_local_ebd = torch.cat(all_local_ebd, dim=0)  # [N, 1024]
        # all_local_aug_ebd = torch.cat(all_local_aug_ebd, dim=0)  # [N, 7, 1024]

        # 然后处理neighbor和global embeddings
        print("Generating neighbor and global embeddings...")
        for idx in tqdm(range(len(valid_spots))):
            # Get neighbor and global indices
            _, n_idx = tree.query([coords[idx]], k=9)  # self + 8 neighbors
            _, g_idx = tree.query([coords[idx]], k=49)  # self + 48 neighbors

            # Regular embeddings
            neighbor_embs = all_local_ebd[n_idx[0]].unsqueeze(0)  # [1, 9, 1024]
            global_embs = all_local_ebd[g_idx[0]].unsqueeze(0)    # [1, 49, 1024]
            neighbor_ebd.append(neighbor_embs)
            global_ebd.append(global_embs)

            # Augmented embeddings - 调整维度顺序
            # neighbor_aug = all_local_aug_ebd[n_idx[0]]  # [9, 7, 1024]
            # neighbor_aug = neighbor_aug.permute(1, 0, 2)  # [7, 9, 1024]
            # neighbor_aug_embs = neighbor_aug.unsqueeze(0)  # [1, 7, 9, 1024]
            # neighbor_aug_ebd.append(neighbor_aug_embs)

            # global_aug = all_local_aug_ebd[g_idx[0]]  # [49, 7, 1024]
            # global_aug = global_aug.permute(1, 0, 2)  # [7, 49, 1024]
            # global_aug_embs = global_aug.unsqueeze(0)  # [1, 7, 49, 1024]
            # global_aug_ebd.append(global_aug_embs)

        # Stack all embeddings
        all_neighbor_ebd = torch.cat(neighbor_ebd, dim=0)
        all_global_ebd = torch.cat(global_ebd, dim=0)
        # all_neighbor_aug_ebd = torch.cat(neighbor_aug_ebd, dim=0)
        # all_global_aug_ebd = torch.cat(global_aug_ebd, dim=0)

        # Save embeddings
        print("Saving embeddings...")
        torch.save(all_local_ebd.cpu(), os.path.join(self.processed_dir, 'local_ebd.pt'))
        torch.save(all_neighbor_ebd.cpu(), os.path.join(self.processed_dir, 'neighbor_ebd.pt'))
        torch.save(all_global_ebd.cpu(), os.path.join(self.processed_dir, 'global_ebd.pt'))
        # torch.save(all_local_aug_ebd.cpu(), os.path.join(self.processed_dir, 'local_aug_ebd.pt'))
        # torch.save(all_neighbor_aug_ebd.cpu(), os.path.join(self.processed_dir, 'neighbor_aug_ebd.pt'))
        # torch.save(all_global_aug_ebd.cpu(), os.path.join(self.processed_dir, 'global_aug_ebd.pt'))

        print("Final data size:")
        print(f"Local embeddings: {all_local_ebd.shape}")
        print(f"Neighbor embeddings: {all_neighbor_ebd.shape}")
        print(f"Global embeddings: {all_global_ebd.shape}")
        # print(f"Local augmented embeddings: {all_local_aug_ebd.shape}")
        # print(f"Neighbor augmented embeddings: {all_neighbor_aug_ebd.shape}")  # 应该是 [N, 7, 9, 1024]
        # print(f"Global augmented embeddings: {all_global_aug_ebd.shape}")     # 应该是 [N, 7, 49, 1024]

class BCProcessor:
    def __init__(self, bc_id='MBSP'):
        self.bc_id = bc_id
        self.data_dir = f'dataset/{self.bc_id}'
        self.spatial_dir = os.path.join(self.data_dir, 'spatial')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        # 文件名适配
        # self.h5_file = os.path.join(self.data_dir, f'filtered_feature_bc_matrix.h5')
        self.img_file = os.path.join(self.data_dir, f'image.tif')
        self.position_file = os.path.join(self.spatial_dir, 'tissue_positions_list.csv')

    def get_spot_coords(self):
        position_df = pd.read_csv(self.position_file, header=None)
        spot_to_coord = {}
        for _, row in position_df.iterrows():
            spot_name = row[0]
            coord = (int(row[4]), int(row[5]))
            spot_to_coord[spot_name] = coord
        return spot_to_coord

    def process_gene_list(self):
        if self.bc_id == 'MBSP' or self.bc_id == 'HCC' or self.bc_id == 'HIC'or self.bc_id == 'VIHBC':
            adata = sc.read_10x_h5(self.h5_file)
            count_mtx = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)
        else:
            count_mtx = pd.read_csv(os.path.join(self.data_dir, "counts.csv"), index_col=0)

        genes_with_expr = count_mtx.columns[count_mtx.sum() > 0]
        filtered_df = count_mtx[genes_with_expr]
        norm_df = filtered_df.div(filtered_df.sum(axis=1), axis=0) * 10000
        norm_df = np.log1p(norm_df)
        gene_means = norm_df.mean()
        gene_stds = norm_df.std()
        genes_by_mean = gene_means.sort_values(ascending=False)
        genes_by_std = gene_stds.sort_values(ascending=False)

        num_genes = 1000
        high_mean_genes = set(genes_by_mean.head(num_genes).index)
        high_std_genes = set(genes_by_std.head(num_genes).index)
        selected_genes = sorted(list(high_mean_genes.intersection(high_std_genes)))
        selected_genes = [gene for gene in selected_genes if not gene.startswith(("MT-", "mt-", "RP", "rp"))]

        # 记录排序后的高变基因顺序（用于 MP 特殊处理）
        all_genes_sorted = genes_by_std.index.tolist()

        # 获取最终用于测试的 300 个基因
        selected_genes = selected_genes[:300]

        gene_list_path = os.path.join(self.processed_dir, 'selected_gene_list.txt')
        with open(gene_list_path, 'w') as f:
            for gene in selected_genes:
                f.write(f"{gene}\n")

        # ========== 筛选训练用基因 ========== #
        if self.bc_id == 'MP':
            # 如果是 MP，从 high_std_genes 中顺序继续往后取
            used_genes_set = set(selected_genes)
            next_300_genes = []
            for gene in all_genes_sorted:
                if gene not in used_genes_set and not gene.startswith(("MT-", "mt-", "RP", "rp")):
                    next_300_genes.append(gene)
                    if len(next_300_genes) >= 300:
                        break
        else:
            # 其他切片保持原逻辑
            selected_indices = [all_genes_sorted.index(gene) for gene in selected_genes if gene in all_genes_sorted]
            max_idx = max(selected_indices) if selected_indices else 0
            next_300_genes = []
            for gene in all_genes_sorted[max_idx + 1:]:
                if gene not in selected_genes and not gene.startswith(("MT-", "mt-", "RP", "rp")):
                    next_300_genes.append(gene)
                    if len(next_300_genes) >= 300:
                        break

        train_gene_list_path = os.path.join(self.processed_dir, 'train_selected_gene_list.txt')
        with open(train_gene_list_path, 'w') as f:
            for gene in next_300_genes:
                f.write(f"{gene}\n")

        print(f"Selected {len(next_300_genes)} genes for training extension set")
        return selected_genes

    def process_embeddings(self):
        image = Image.open(self.img_file)
        position_df = pd.read_csv(self.position_file, header=None)
        valid_spots = position_df[position_df[1] == 1][0].values
        coords = position_df[position_df[1] == 1][[4, 5]].values.astype(int)
        tree = KDTree(coords)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        token = ""
        login(token=token, add_to_git_credential=False)
        model_UNI, transform_UNI = get_encoder(enc_name='uni', device=device)
        all_local_ebd = []
        neighbor_ebd = []
        global_ebd = []
        print("Generating local embeddings...")
        for spot_idx, spot in enumerate(tqdm(valid_spots)):
            x, y = coords[spot_idx]
            patch = crop_patch(image, x, y)
            if patch.mode != 'RGB':
                patch = patch.convert('RGB')
            local_emb = get_img_embd_uni(patch, model_UNI, transform_UNI, device)
            all_local_ebd.append(local_emb)
        all_local_ebd = torch.cat(all_local_ebd, dim=0)
        print("Generating neighbor and global embeddings...")
        for idx in tqdm(range(len(valid_spots))):
            _, n_idx = tree.query([coords[idx]], k=9)
            _, g_idx = tree.query([coords[idx]], k=49)
            neighbor_embs = all_local_ebd[n_idx[0]].unsqueeze(0)
            global_embs = all_local_ebd[g_idx[0]].unsqueeze(0)
            neighbor_ebd.append(neighbor_embs)
            global_ebd.append(global_embs)
        all_neighbor_ebd = torch.cat(neighbor_ebd, dim=0)
        all_global_ebd = torch.cat(global_ebd, dim=0)
        print("Saving embeddings...")
        torch.save(all_local_ebd.cpu(), os.path.join(self.processed_dir, 'local_ebd.pt'))
        # torch.save(all_neighbor_ebd.cpu(), os.path.join(self.processed_dir, 'neighbor_ebd.pt'))
        # torch.save(all_global_ebd.cpu(), os.path.join(self.processed_dir, 'global_ebd.pt'))
        print("Final data size:")
        print(f"Local embeddings: {all_local_ebd.shape}")
        print(f"Neighbor embeddings: {all_neighbor_ebd.shape}")
        print(f"Global embeddings: {all_global_ebd.shape}")


def crop_patch(image, center_x, center_y, size=110):
    """从图像中裁剪patch，如果超出边界则补白"""
    half_size = size // 2
    left = center_x - half_size
    top = center_y - half_size
    right = center_x + half_size
    bottom = center_y + half_size

    # 触发补白逻辑
    if left < 0 or top < 0 or right > image.width or bottom > image.height:
        new_image = Image.new('RGB', (size, size), 'white')

        valid_left = max(0, left)
        valid_top = max(0, top)
        valid_right = min(image.width, right)
        valid_bottom = min(image.height, bottom)

        # ✅ 添加合法性检查
        if valid_right <= valid_left or valid_bottom <= valid_top:
            print(f"[WARNING] Invalid crop: center=({center_x}, {center_y}), image size=({image.width}, {image.height})")
            return new_image  # 返回白图或 None

        region = image.crop((valid_left, valid_top, valid_right, valid_bottom))
        paste_left = abs(min(0, left))
        paste_top = abs(min(0, top))
        new_image.paste(region, (paste_left, paste_top))
        return new_image

    else:
        return image.crop((left, top, right, bottom))
def get_img_embd_uni(patch, model, transform, device):
    """获取UNI模型的图像嵌入"""
    base_width = 224
    patch_resized = patch.resize((base_width, base_width), Image.Resampling.LANCZOS)
    img_transformed = transform(patch_resized).unsqueeze(dim=0)
    with torch.inference_mode():
        feature_emb = model(img_transformed.to(device))
    return torch.clone(feature_emb)

def patch_augmentation_embd(patch, model, transform, device):
    """获取patch增强后的嵌入"""
    embd_dim = 1024  # UNI model dimension
    num_transpose = 7
    patch_aug_embd = torch.zeros(num_transpose, embd_dim)
    
    for trans in range(num_transpose):
        patch_transposed = patch.transpose(trans)
        patch_embd = get_img_embd_uni(patch_transposed, model, transform, device)
        patch_aug_embd[trans, :] = torch.clone(patch_embd)
    
    return patch_aug_embd.unsqueeze(0)


class Otherprocessor:
    def __init__(self, bc_id='MBSP'):
        self.bc_id = bc_id
        self.data_dir = f'dataset/{self.bc_id}'

        self.processed_dir = os.path.join(self.data_dir, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        # 文件名适配
        if self.bc_id == 'Alex2' or self.bc_id == 'Alex'or self.bc_id == 'Alex3':
            matrix_dir = os.path.join(self.data_dir, 'filtered_count_matrix')
            # === 手动读取并兼容 features.tsv ===
            features_path = os.path.join(matrix_dir, 'features.tsv.gz')
            barcodes_path = os.path.join(matrix_dir, 'barcodes.tsv.gz')
            matrix_path = os.path.join(matrix_dir, 'matrix.mtx.gz')
            # 读取 features.tsv.gz
            with gzip.open(features_path, 'rt') as f:
                features = pd.read_csv(f, sep='\t', header=None)
            # 如果只有一列，则临时补齐为三列（不修改原文件）
            if features.shape[1] == 1:
                features[1] = features[0]
                features[2] = "Gene Expression"
            # === 使用临时 DataFrame 创建 AnnData ===
            adata = sc.read_mtx(matrix_path).T  # 转置成 spots × genes
            with gzip.open(barcodes_path, 'rt') as f:
                barcodes = [line.strip() for line in f]
            adata.obs_names = barcodes
            adata.var_names = features[1].values  # 第二列为基因名
            # === 归一化 + log1p 预处理 ===
            # sc.pp.normalize_total(adata, target_sum=1e4)
            # sc.pp.log1p(adata)
            # === 转换为 DataFrame 方便后续处理 ===
            self.count_mtx = pd.DataFrame(
                adata.X.toarray(),
                index=adata.obs_names,
                columns=adata.var_names
            )
            self.img_file = os.path.join(self.data_dir, f'image.tif')
            self.position_file = os.path.join(self.data_dir,'spatial', 'tissue_positions_list.csv')
        else:
            self.counts_file = os.path.join(self.data_dir, 'counts.csv')
            self.img_file = os.path.join(self.data_dir, f'image.jpg')
            self.position_file = os.path.join(self.data_dir, 'tissue_positions_list.csv')

    def get_spot_coords(self):
        position_df = pd.read_csv(self.position_file, header=None)
        spot_to_coord = {}
        for _, row in position_df.iterrows():
            spot_name = row[0]
            coord = (int(row[4]), int(row[5]))
            spot_to_coord[spot_name] = coord
        return spot_to_coord

    def process_gene_list(self):

        # count_mtx = pd.read_csv(self.counts_file, index_col=0)
        count_mtx = self.count_mtx
        # 标准的预处理流程
        adata = sc.AnnData(count_mtx)
        adata.var_names_make_unique()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)



        # 选取高变基因，使用 Seurat v3 方法
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)

        # 获取高变基因名
        selected_genes = adata.var[adata.var['highly_variable']].index.tolist()

        # genes_with_expr = count_mtx.columns[count_mtx.sum() > 0]
        # filtered_df = count_mtx[genes_with_expr]
        # norm_df = filtered_df.div(filtered_df.sum(axis=1), axis=0) * 10000
        # norm_df = np.log1p(norm_df)
        # gene_means = norm_df.mean()
        # gene_stds = norm_df.std()
        # genes_by_mean = gene_means.sort_values(ascending=False)
        # genes_by_std = gene_stds.sort_values(ascending=False)
        # num_genes = 1000
        # high_mean_genes = set(genes_by_mean.head(num_genes).index)
        # high_std_genes = set(genes_by_std.head(num_genes).index)
        # selected_genes = sorted(list(high_mean_genes.intersection(high_std_genes)))
        # selected_genes = [gene for gene in selected_genes if not gene.startswith(("MT-", "mt-", "RP", "rp"))]
        # selected_genes = selected_genes[:300]
        # print(f"Selected {len(selected_genes)} highly variable genes")
        gene_list_path = os.path.join(self.processed_dir, 'selected_gene_list.txt')
        with open(gene_list_path, 'w') as f:
            for gene in selected_genes:
                f.write(f"{gene}\n")
        return selected_genes

    def process_embeddings(self):
        image = Image.open(self.img_file)
        position_df = pd.read_csv(self.position_file, header=None)
        valid_spots = position_df[position_df[1] == 1][0].values
        coords = position_df[position_df[1] == 1][[4, 5]].values.astype(int)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        token = ""
        login(token=token, add_to_git_credential=False)
        model_UNI, transform_UNI = get_encoder(enc_name='uni', device=device)
        all_local_ebd = []
        print("Generating local embeddings...")
        for spot_idx, spot in enumerate(tqdm(valid_spots)):
            x, y = coords[spot_idx]
            patch = crop_patch(image, x, y)
            if patch.mode != 'RGB':
                patch = patch.convert('RGB')
            local_emb = get_img_embd_uni(patch, model_UNI, transform_UNI, device)
            all_local_ebd.append(local_emb)
        all_local_ebd = torch.cat(all_local_ebd, dim=0)
        print("Generating neighbor and global embeddings...")
        print("Saving embeddings...")
        torch.save(all_local_ebd.cpu(), os.path.join(self.processed_dir, 'local_ebd.pt'))
        print(f"Local embeddings: {all_local_ebd.shape}")


def main():
    # 处理指定切片
    # section_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
    #                 "151675", "151676"]
    section_list = ["151507","151673","151674"]
    # section_list = ["Alex3"]
    for section_id in section_list:
        other_processor = DLPFCProcessor(section_id)



    #     # 处理基因列表
    #     print("Processing gene list...")
    #     selected_genes = processor.process_gene_list()
    #     print(f"Saved {len(selected_genes)} genes to selected_gene_list.txt")

        # 处理图像嵌入
        # print("\nProcessing image embeddings...")
        # bc_processor = BCProcessor(section_id)
    # selected_genes = bc_processor.process_gene_list()
    # print(f"Saved {len(selected_genes)} genes to selected_gene_list.txt")
    #     other_processor.process_embeddings()
    #other_processor = Otherprocessor('HS')
        selected_genes = other_processor.process_gene_list()
        # print(f"Saved {len(selected_genes)} genes to selected_gene_list.txt")
        # other_processor.process_embeddings()


if __name__ == '__main__':
    main() 