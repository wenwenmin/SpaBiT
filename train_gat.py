import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# === 导入自定义模块 ===
from dataset import SpatialDataManager          # 从项目根目录导入
from models import GAT_Regressor                # 从 Baseline/HistoSGE/models.py 导入

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def train_gat_regression(
    selection_id="151508",
    n_hidden=256,
    n_out=1024,
    n_heads=4,
    dropout=0.2,
    alpha=0.01,
    lr=1e-3,
    weight_decay=5e-4,
    num_epochs=300
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # === 计算相对路径 ===
    current_dir = os.path.dirname(os.path.abspath(__file__))         # 当前文件 Baseline/HistoSGE
    project_root = os.path.dirname(os.path.dirname(current_dir))     # 回到 diffusion_st_prediction_2
    data_dir = os.path.join(project_root, "dataset", selection_id)
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # === 加载全图数据 ===
    data_manager = SpatialDataManager(selection_id=selection_id, train_ratio=0.5, seed=42, neighbor_ratio=6)
    gat_data = data_manager.get_gat_dataset()
    features = gat_data["features"].to(device)   # [N, G]
    target = gat_data["features"].to(device)     # 目标是基因表达自身
    adj = gat_data["adj"].to(device)

    # === 构建稀疏边索引 edge_index ===
    src, dst = torch.nonzero(adj > 0, as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0).to(device)  # shape [2, E]

    # === 训练 mask（True = 真实 spot, False = 填充 spot） ===
    train_mask = torch.tensor(data_manager.train_mask, dtype=torch.bool).to(device)
    print(f"[INFO] Real (train) spots: {train_mask.sum().item()} / {len(train_mask)}")

    # === 初始化模型 ===
    nfeat = features.shape[1]
    model = GAT_Regressor(
        nfeat=nfeat,
        nhid=n_hidden,
        nout=n_out,
        dropout=dropout,
        alpha=alpha,
        nheads=n_heads
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss(reduction="none")  # 不平均，后面按 mask 取样本

    print(f"[INFO] Model initialized: {nfeat} → {n_hidden}×{n_heads} → {n_out}")
    print("[INFO] Start training GAT ...")

    # === 训练阶段 ===
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training GAT (masked regression)"):
        optimizer.zero_grad()

        # 前向传播
        emb,gene = model(features, edge_index)
        pred = gene  # 预测基因表达

        # Masked loss
        loss_all = criterion(pred, target).mean(dim=1)  # 每个节点损失
        # loss = loss_all[train_mask].mean()              # 仅真实 spot 计算损失
        loss = loss_all.mean()
        # 反向传播
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]  Masked MSE Loss: {loss.item():.6f}")

    print("[INFO] Training finished ✅")

    # === 保存模型 ===
    model_path = os.path.join(processed_dir, "gat_regressor_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model weights saved to {model_path}")

    # === 生成稳定嵌入并保存 ===
    model.eval()
    with torch.no_grad():
        gat_emb,gene = model(features, edge_index)   # [N, 1024]
    emb_path = os.path.join(processed_dir, "gat_ebd.pt")
    torch.save(gene.cpu(), emb_path)
    print(f"[INFO] GAT embeddings saved to {emb_path}")

    return model, gat_emb.cpu()


def test_gat_regression(selection_id="151508"):
    """加载模型并重新生成节点嵌入（验证模型是否能正确加载）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_dir = os.path.join(project_root, "dataset", "DLPFC", selection_id)
    processed_dir = os.path.join(data_dir, "processed")

    # === 加载数据 ===
    data_manager = SpatialDataManager(selection_id=selection_id)
    gat_data = data_manager.get_gat_dataset()
    features = gat_data["features"].to(device)
    adj = gat_data["adj"].to(device)

    # === 构建边索引 ===
    src, dst = torch.nonzero(adj > 0, as_tuple=True)
    edge_index = torch.stack([src, dst], dim=0).to(device)

    # === 初始化模型并加载权重 ===
    nfeat = features.shape[1]
    model = GAT_Regressor(nfeat=nfeat, nhid=256, nout=1024, dropout=0.2, alpha=0.01, nheads=4).to(device)
    model_path = os.path.join(processed_dir, "gat_regressor_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        emb = model(features, edge_index)
    print(f"[INFO] Loaded GAT embedding shape: {emb.shape}")
    print("First 2 node embeddings:\n", emb[:2].cpu().numpy())
    return emb.cpu()


if __name__ == "__main__":
    section_list = ["MBSP","HCC","HIC","MP","HS","HBC_HD","MB_HD","HL_X","HL_X2"]
    for section_id in section_list:
        setup_seed(5)
        train_gat_regression(
            selection_id=section_id,
            n_hidden=650,
            n_out=1024,
            n_heads=6,
            dropout=0.2,
            alpha=0.01,
            lr=1e-4,
            num_epochs=1
        )
        # test_gat_regression(section_id)
