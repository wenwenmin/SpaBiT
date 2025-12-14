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

def crop_patch(image, center_x, center_y, size=224):
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

        if valid_right <= valid_left or valid_bottom <= valid_top:
            print(f"[WARNING] Invalid crop: center=({center_x}, {center_y}), "
                  f"image size=({image.width}, {image.height})")
            return new_image

        region = image.crop((valid_left, valid_top, valid_right, valid_bottom))
        paste_left = abs(min(0, left))
        paste_top = abs(min(0, top))
        new_image.paste(region, (paste_left, paste_top))
        return new_image
    else:
        return image.crop((left, top, right, bottom))


def get_resnet_encoder(device):
    resnet = models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()
    resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return resnet, transform


def get_densenet_encoder(device):
    """预训练 DenseNet121 (ImageNet)，输出 1024 维特征"""
    densenet = models.densenet121(pretrained=True)
    densenet.classifier = torch.nn.Identity()
    densenet.to(device)
    densenet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return densenet, transform


# ============ 通用图像编码类 ============

class DLPFCProcessor:

    def __init__(self, slice_id: str):
        self.slice_name = slice_id
        dlpfc_12 = {
            '151507', '151508', '151509', '151510',
            '151669', '151670', '151671', '151672',
            '151673', '151674', '151675', '151676'
        }

        current_dir = os.path.dirname(os.path.abspath(__file__))
        search_dir = current_dir
        project_root = current_dir 

        while True:
            candidate = os.path.join(search_dir, 'dataset')
            if os.path.isdir(candidate):
                project_root = search_dir
                break

            parent = os.path.dirname(search_dir)
            if parent == search_dir: 
                project_root = current_dir
                break

            search_dir = parent


        if self.slice_name in dlpfc_12:
            # DLPFC 12: .../dataset/DLPFC/151507
            self.data_path = os.path.join(project_root, 'dataset', 'DLPFC', self.slice_name)
        else:
            # 其他：.../dataset/HBC_HD
            self.data_path = os.path.join(project_root, 'dataset', self.slice_name)

        self.processed_dir = os.path.join(self.data_path, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)


    def _load_position_df(self):

        spatial_no_header = {'MBSP', '151507', 'HCC', 'HIC'}
        root_no_spatial = {'MP', 'HS', 'HBC_HD', 'MB_HD', 'HL_X', 'HL_X2'}

        if self.slice_name in spatial_no_header:
            pos_path = os.path.join(self.data_path, 'spatial', 'tissue_positions_list.csv')
            position_df = pd.read_csv(pos_path, header=None)

        elif self.slice_name in root_no_spatial:
            pos_path = os.path.join(self.data_path, 'tissue_positions_list.csv')
            position_df = pd.read_csv(pos_path, header=None)

        else:

            pos_path = os.path.join(self.data_path, 'spatial', 'tissue_positions_list.csv')
            position_df = pd.read_csv(pos_path, header=None, skiprows=1)

        return position_df

    def _get_valid_spots_and_coords(self):
        position_df = self._load_position_df()
        valid_mask = position_df[1] == 1
        valid_spots = position_df.loc[valid_mask, 0].values
        coords = position_df.loc[valid_mask, [4, 5]].values.astype(int)
        return valid_spots, coords

    

    def _load_full_image(self):
        dlpfc_12 = {
            '151507', '151508', '151509', '151510',
            '151669', '151670', '151671', '151672',
            '151673', '151674', '151675', '151676'
        }

        candidates = []

        if self.slice_name in dlpfc_12:
            candidates.append(os.path.join(self.data_path, 'spatial',
                                           f'{self.slice_name}_full_image.tif'))

            candidates.append(os.path.join(self.data_path, 'spatial', 'image.tif'))
            candidates.append(os.path.join(self.data_path, 'spatial', 'image.jpg'))
        else:

            if self.slice_name in {'MP', 'MBSP'}:
   
                candidates.append(os.path.join(self.data_path, 'image.tif'))
                candidates.append(os.path.join(self.data_path, 'spatial', 'image.tif'))
            else:
                candidates.append(os.path.join(self.data_path, 'image.jpg'))
                candidates.append(os.path.join(self.data_path, 'spatial', 'image.jpg'))

        candidates.extend([
            os.path.join(self.data_path, 'spatial', 'tissue_hires_image.png'),
            os.path.join(self.data_path, 'image.png'),
        ])

   
        seen = set()
        unique_candidates = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                unique_candidates.append(p)

        for img_path in unique_candidates:
            if os.path.exists(img_path):
                return Image.open(img_path).convert("RGB")

        raise FileNotFoundError(
            f"[ERROR] Cannot find image file for slice {self.slice_name} in {self.data_path}.\n"
            f"Tried: {unique_candidates}"
        )


    def _get_image_encoder(self, encoder_type: str, device: torch.device):
        encoder_type = encoder_type.lower()

        if encoder_type == 'uni':
            token = ""
            login(token=token, add_to_git_credential=False)
            model_uni, transform_uni = get_encoder(enc_name='uni', device=device)
            model_uni.eval()

            def encode_fn(patch: Image.Image):
                patch_resized = patch.resize((224, 224), Image.Resampling.LANCZOS)
                img = transform_uni(patch_resized).unsqueeze(0).to(device)
                with torch.inference_mode():
                    feat = model_uni(img)  # [1, 1024]
                return feat.detach().cpu()

            save_name = 'local_ebd.pt'

        elif encoder_type == 'conch':
            token = ""
            conch_model, conch_preprocess = create_model_from_pretrained(
                'conch_ViT-B-16',
                "hf_hub:MahmoodLab/conch",
                device=device,
                hf_auth_token=token
            )
            conch_model.eval()

            def encode_fn(patch: Image.Image):
                patch_resized = patch.resize((224, 224), Image.Resampling.LANCZOS)
                img = conch_preprocess(patch_resized).unsqueeze(0).to(device)
                with torch.inference_mode():
                    feat = conch_model.encode_image(
                        img,
                        proj_contrast=False,
                        normalize=False
                    )  # [1, 512]
                return feat.detach().cpu()
            save_name = 'local_ebd_conch.pt'

        elif encoder_type == 'resnet':
            resnet_model, resnet_transform = get_resnet_encoder(device)
            resnet_model.eval()
            def encode_fn(patch: Image.Image):
                patch_resized = patch.resize((224, 224), Image.Resampling.LANCZOS)
                img = resnet_transform(patch_resized).unsqueeze(0).to(device)
                with torch.inference_mode():
                    feat = resnet_model(img)  # [1, 2048]
                return feat.detach().cpu()
            save_name = 'local_ebd_resnet.pt'

        elif encoder_type in ('densenet', 'densenet121'):
            densenet_model, densenet_transform = get_densenet_encoder(device)
            densenet_model.eval()
            def encode_fn(patch: Image.Image):
                patch_resized = patch.resize((224, 224), Image.Resampling.LANCZOS)
                img = densenet_transform(patch_resized).unsqueeze(0).to(device)
                with torch.inference_mode():
                    feat = densenet_model(img)  # [1, 1024]
                return feat.detach().cpu()

            save_name = 'local_ebd_densenet121.pt'
        else:
            raise ValueError(
                f"Unknown encoder_type: {encoder_type} "
                f"(supported: 'uni', 'conch', 'resnet', 'densenet121')"
            )
        return encode_fn, save_name



    def process_image_embeddings(self, encoder_type: str = 'uni'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = self._load_full_image()
        valid_spots, coords = self._get_valid_spots_and_coords()
        encode_fn, save_name = self._get_image_encoder(encoder_type, device)
        all_embeds = []
        print(f"[{self.slice_name}] Generating {encoder_type} local embeddings ...")
        for (x, y) in tqdm(coords, desc=f"{self.slice_name} spots"):
            patch = crop_patch(image, x, y, size=224)
            emb = encode_fn(patch)  # [1, D]
            all_embeds.append(emb)
        all_embeds = torch.cat(all_embeds, dim=0)
        out_path = os.path.join(self.processed_dir, save_name)
        torch.save(all_embeds, out_path)
        print(f"[{self.slice_name}] {encoder_type} local embeddings saved to: {out_path}")
        print(f"           shape: {all_embeds.shape}")


    def process_gene_list(self):

        adata = sc.read_10x_h5(os.path.join(self.data_dir, 'filtered_feature_bc_matrix.h5'))
        count_mtx = pd.DataFrame(adata.X.toarray(), columns=adata.var_names, index=adata.obs_names)
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
        selected_genes = [gene for gene in selected_genes
                        if not gene.startswith(("MT-", "mt-", "RP", "rp"))]
        selected_genes = selected_genes[:300]
        # adata = sc.read_10x_h5(os.path.join(self.data_dir, 'filtered_feature_bc_matrix.h5'))
        if len(selected_genes) < 300:
            remaining_genes = adata.var.loc[~adata.var_names.isin(selected_genes)]
            additional_genes = remaining_genes.sort_values("variances", ascending=False).index.tolist()
            selected_genes += [g for g in additional_genes if not g.startswith(("MT-", "mt-", "RP", "rp"))]
            selected_genes = selected_genes[:300]

        print(f"Selected {len(selected_genes)} highly variable genes")
        

        gene_list_path = os.path.join(self.processed_dir, 'selected_gene_list.txt')
        with open(gene_list_path, 'w') as f:
            for gene in selected_genes:
                f.write(f"{gene}\n")
        return selected_genes


def get_resnet_encoder(device):
    resnet = models.resnet50(pretrained=True)
    resnet.fc = torch.nn.Identity()  
    resnet.to(device)
    resnet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    return resnet, transform
def get_densenet_encoder(device):
    densenet = models.densenet121(pretrained=True)
    densenet.classifier = torch.nn.Identity()
    densenet.to(device)
    densenet.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),   
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  
            std=[0.229, 0.224, 0.225]
        )
    ])
    return densenet, transform


def main():
    section_list = []
    for section_id in section_list:
        processor = DLPFCProcessor(section_id)
        processor.process_image_embeddings('uni')
        processor.process_gene_list


if __name__ == '__main__':
    main() 
