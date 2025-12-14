import os

from functorch.einops import rearrange
from scipy.stats import pearsonr
import anndata as ad
from utils import *
import scanpy as sc
from model import ConvNetwork as Model
import process_data


def build_adata_from_prediction(pred_hr, hr_x, hr_y, in_tissue_mask, gene_list):
    """
    将预测表达矩阵转换为 AnnData，保留每个 spot 的坐标和基因名。
    - pred_hr: shape [n_genes, H, W]
    - hr_x, hr_y: np.mgrid 网格坐标 (H, W)
    - in_tissue_mask: shape [H, W]
    - gene_list: list of gene names, length = n_genes
    """
    pred_hr = pred_hr.cpu().numpy()  # [n_genes, H, W]
    in_tissue_mask = in_tissue_mask.cpu().numpy()

    n_genes, H, W = pred_hr.shape
    coords = []
    expr = []

    for i in range(H):
        for j in range(W):
            if in_tissue_mask[i, j] > 0:
                coords.append([hr_x[i, j], hr_y[i, j]])
                expr.append(pred_hr[:, i, j])  # 每个spot的n_genes表达

    coords = np.array(coords)
    expr = np.array(expr)  # shape = [n_spots, n_genes]

    # 构造 AnnData 对象
    adata = ad.AnnData(X=expr)
    adata.obs['array_row'] = coords[:, 0]
    adata.obs['array_col'] = coords[:, 1]
    adata.obsm['coord'] = coords  # 添加空间坐标信息
    adata.var_names = gene_list

    return adata
def decode_patches(pre_hr): #将多通道patch解码为高分辨基因图像
    g, _, h, w = pre_hr.shape
    out = torch.zeros((g, h * 2, w * 2), device=pre_hr.device)
    out[:, 0::2, 0::2] = pre_hr[:, 0, :, :]  # 左上
    out[:, 0::2, 1::2] = pre_hr[:, 1, :, :]  # 右上
    out[:, 1::2, 0::2] = pre_hr[:, 2, :, :]  # 左下
    out[:, 1::2, 1::2] = pre_hr[:, 3, :, :]  # 右下
    return out
def train_and_test(selection_id=151507):
    data_path=f"data\\DLPFC\\{selection_id}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adata = sc.read_visium(path=data_path, count_file='filtered_feature_bc_matrix.h5')
    adata.var_names_make_unique()
    train_gene_list = list(np.genfromtxt(os.path.join(data_path, "processed", "train_selected_gene_list.txt"), dtype=str))
    test_gene_list = list(np.genfromtxt(os.path.join(data_path, "processed", "selected_gene_list.txt"), dtype=str))
    #划分好训练和测试所需要的所有数据
    coords = adata.obs[['array_row', 'array_col']]
    valid_train_genes = [g for g in train_gene_list if g in adata.var_names]
    train_counts = adata[:, valid_train_genes].X.toarray()
    train_counts = np.log2(train_counts + 1)
    valid_test_genes = [g for g in test_gene_list if g in adata.var_names]
    test_counts = adata[:, valid_test_genes].X.toarray()
    test_counts = np.log2(test_counts + 1)

    train_lr, train_hr, in_tissue_matrix,_,_ = get10Xtrainset(train_counts, coords)
    train_lr_h, train_lr_w = train_lr.shape[1], train_lr.shape[2]
    in_tissue_matrix = torch.Tensor(in_tissue_matrix).to(device)
    b, h, w = train_hr.shape
    train_lr = data_pad(train_lr, 4)    #对原基因图进行0填充

    train_lr = torch.Tensor(train_lr.reshape((b, 1, int(train_lr.shape[1]), int(train_lr.shape[2]))))
    train_hr = torch.Tensor(train_hr.reshape((b, 1, h, w)))

    net = Model(patch_size=4, embed_dim=64).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, betas=(0.5, 0.6), eps=1e-6)
    for epoch in range(5):
        loss_running = 0
        idx = 0
        for b_id, data in enumerate(data_iter(train_lr, train_hr, 512), 0):
            idx += 1
            lr, hr = data
            lr, hr = lr.to(device), hr.to(device)
            pre_hr = net(lr)

            loss = criterion(pre_hr, hr, in_tissue_matrix, train_lr_h=train_lr_h, train_lr_w=train_lr_w,transformer=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_running += loss.item()
        print(f'epoch:{epoch + 1}, loss:{loss_running / idx}')
    torch.save(net.state_dict(), f'{selection_id}_p4h8.params')



    net.load_state_dict(torch.load(f'{selection_id}_p4h8.params'))
    net.eval()


    test_lr, test_hr, in_tissue_matrix,hr_x,hr_y = get10Xtrainset(test_counts, coords)
    test_lr_h, test_lr_w = test_lr.shape[1], test_lr.shape[2]
    in_tissue_matrix = torch.Tensor(in_tissue_matrix).to(device)
    b, h, w = test_hr.shape
    test_lr = data_pad(test_lr, 4)    #对原基因图进行0填充

    test_lr = torch.Tensor(test_lr.reshape((b, 1, int(test_lr.shape[1]), int(test_lr.shape[2]))))
    test_hr = torch.Tensor(test_hr.reshape((b, 1, h, w)))
    # 获取低分辨率输入（padding后）和高分辨率ground truth
    test_lr=test_lr.to(device)


    # 预测
    with torch.no_grad():
        pre_hr = net(test_lr)
        pre_hr = pre_hr[:, :, 0:train_lr_h, 0:train_lr_w]  # 裁剪掉padding部分

    # 原始高分辨率目标裁剪成 4 个子图
    gt_sub1 = test_hr[:, 0, 0::2, 0::2]  # 左上
    gt_sub2 = test_hr[:, 0, 0::2, 1::2]  # 右上
    gt_sub3 = test_hr[:, 0, 1::2, 0::2]  # 左下
    gt_sub4 = test_hr[:, 0, 1::2, 1::2]  # 右下

    mask1 = in_tissue_matrix[0::2, 0::2]
    mask2 = in_tissue_matrix[0::2, 1::2]
    mask3 = in_tissue_matrix[1::2, 0::2]
    mask4 = in_tissue_matrix[1::2, 1::2]

    gt_parts = [gt_sub1, gt_sub2, gt_sub3, gt_sub4]
    pred_parts = [pre_hr[:, i, :, :] for i in range(4)]
    mask_parts = [mask1, mask2, mask3, mask4]

    # 合并所有有效像素点（组织内）用于评估
    all_gt = []
    all_pred = []

    for i in range(4):
        gt = gt_parts[i].cpu().numpy()
        pred = pred_parts[i].cpu().numpy()
        mask = mask_parts[i].cpu().numpy()

        for b_id in range(gt.shape[0]):
            valid = mask > 0
            all_gt.append(gt[b_id][valid])
            all_pred.append(pred[b_id][valid])

    # 拼接所有样本的有效值
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    # 计算评估指标
    pcc = pearsonr(all_gt, all_pred)[0]
    mse = np.mean((all_gt - all_pred) ** 2)
    mae = np.mean(np.abs(all_gt - all_pred))

    print(f"Test Result on Retained Downsampled Points:")
    print(f"  PCC : {pcc:.4f}")
    print(f"  MSE : {mse:.4f}")
    print(f"  MAE : {mae:.4f}")
    pre_hr_full = decode_patches(pre_hr)
    final_adata = build_adata_from_prediction(pre_hr_full, hr_x, hr_y, in_tissue_matrix, test_gene_list)
    save_path='DIST_pre'
    final_adata.write_h5ad(f"{save_path}/{selection_id}_final_adata.h5ad")
    return pcc, mse, mae

if __name__ == '__main__':
    section_list = [
                                   "151675", "151676"]  # 假设有多个 section_id
    results = {
        "PCC": [],
        "MSE": [],
        "MAE": []
    }
    for section_id in section_list:
        pcc, mse, mae = train_and_test(section_id)
        results["PCC"].append(round(pcc, 4))
        results["MSE"].append(round(mse, 4))
        results["MAE"].append(round(mae, 4))

    df = pd.DataFrame(results, index=["PCC", "MSE", "MAE"]).T  # 转置，以便每个 section_id 成为列
    df.columns = section_list  # 设置列名为 section_list

    # 保存为 CSV 文件
    df.to_csv("SpaViT_section_metrics.csv")
    print("\n✅ 保存完毕: SpaViT_section_metrics.csv")