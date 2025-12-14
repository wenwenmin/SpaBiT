import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
from scipy.stats import pearsonr
import numpy as np
from models import SpaBiT

from dataset import SpatialDataManager
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import scanpy as sc
import pandas as pd
def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def train(selection_id='151507', num_genges=None, train_loader=None):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpaBiT(in_features=1024 , depth=4, trans_heads=4,num_genes=num_genges, dropout= 0.).to(device)

    model_optim = optim.Adam(model.parameters(), lr=0.001)
    model_sche = optim.lr_scheduler.StepLR(model_optim, step_size=500, gamma=1)
    loss_fn = nn.MSELoss ()

    num_epochs = 1000
    with tqdm(range(num_epochs), total=num_epochs, desc='Epochs') as epoch:
        for j in epoch:
            train_re_loss = []

            for gene, local_emb, coord, spatial, neighbor_data, labels in train_loader:
                gene,local_emb,neighbor_data , coord =gene.to(device), local_emb.to(device),neighbor_data.to(device), coord.to(device)
                # neighbor_mean = neighbor_data.mean(dim=1).to(device)

                model_optim.zero_grad()
                _, xrecon = model(local_emb,neighbor_data,coord)
                recon_loss = loss_fn(xrecon, gene)  # + 0.1 * L1_loss(xrecon,exp)  # + 0.1 * sliced_wasserstein_distance(xrecon, exp, 1000, device=device)
                recon_loss.backward()
                model_optim.step()
                model_sche.step()
                train_re_loss.append(recon_loss.item())
                epoch_info = 'recon_loss: %.5f' % \
                             (torch.mean(torch.FloatTensor(train_re_loss)))
                epoch.set_postfix_str(epoch_info)
    torch.save(model, selection_id + "_train_" + ".ckpt")

def test_model(model, test_loader,selection_id, device):
  
    model.eval()
    preds=None
    h_embs = None
    truth=[]
    coords_all=[]
    spatial_all=[]
    # labels_all=[]
   
    with torch.no_grad(): 
        for gene, local_emb, coord ,spatial, neighbor_data, labels in test_loader:
            gene, local_emb,neighbor_data , coord = gene.to(device), local_emb.to(
                device),neighbor_data.to(device), coord.to(device)
            # neighbor_mean = neighbor_data.mean(dim=1).to(device)
           
            h,pred = model(local_emb,neighbor_data,coord)
            coords_all.append(coord.cpu().numpy())
            spatial_all.append(spatial)
            # labels_all.append(labels.cpu().numpy())
            if preds is None:
                preds = pred.squeeze()
                # h_embs = h.squeeze()
                truth = gene.cpu().numpy()
            else:
                pred = pred.squeeze()
                # h = h.squeeze()
                # h_embs = torch.cat((h_embs, h), dim=0)
                preds = torch.cat((preds, pred), dim=0)
                truth = np.concatenate((truth, gene.cpu().numpy()), axis=0)
        generate_profile = preds.squeeze().cpu().detach().numpy()
        # hidden_emb = h_embs.squeeze().cpu().detach().numpy()
        coords_all = np.concatenate(coords_all, axis=0)
        spatial_all = np.concatenate(spatial_all, axis=0)
        # labels_all = np.concatenate(labels_all, axis=0)
        adata_stage = sc.AnnData(generate_profile)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        data_dir = os.path.join(project_root, 'dataset','DLPFC', selection_id)   
        with open(os.path.join(data_dir, 'processed', 'selected_gene_list.txt'), 'r') as f:
            used_genes = [line.strip() for line in f.readlines() if line.strip()]
        adata_stage.var.index = used_genes
        adata_stage.obsm["coord"] = coords_all
        adata_stage.obsm["spatial"] = spatial_all
        # adata_stage.obsm["embedding"] = hidden_emb
        # adata_stage.obs["tumor"] = labels_all

        adata_ground_truth = sc.AnnData(truth)
        adata_ground_truth.var.index = used_genes
        # adata_stage.write(selection_id+"_stage.h5ad")
        # adata_ground_truth.write(selection_id+"_ground_truth.h5ad")

        pr_stage = np.zeros(shape=(adata_stage.shape[1]))
        mse_values = np.zeros(adata_stage.shape[1])
        mae_values = np.zeros(adata_stage.shape[1])
        for i in tqdm(range(len(used_genes))):
            pred_i = adata_stage[:, used_genes[i]].X
            true_i = adata_ground_truth[:, used_genes[i]].X
            pr_stage[i] = pearsonr(pred_i.toarray().squeeze(), true_i.toarray().squeeze())[0]
            mse_values[i] = mean_squared_error(pred_i.toarray().squeeze(), true_i.toarray().squeeze())
            mae_values[i] = mean_absolute_error(pred_i.toarray().squeeze(), true_i.toarray().squeeze())
        #
        pr_stage_common = pr_stage[~np.isnan(pr_stage)]
        print("PCC:", selection_id, np.mean(pr_stage_common))
        print("AVG MSE:", selection_id, np.mean(mse_values))
        print("AVG MAE:", selection_id, np.mean(mae_values))

        # target_genes = ["Cryab","Rgs9","Tmeff2","Ptprn"]  
        # print(f"\nPCC values for specific genes in {selection_id}:")
        # for gene_name in target_genes:
        #     if gene_name in used_genes:
        #         idx = used_genes.index(gene_name)
        #         print(f"{gene_name}: PCC = {pr_stage[idx]:.4f}")
        #     else:
        #         print(f"{gene_name}: Not found in used_genes.")
        #
        # 
        # top5_idx = np.argsort(-pr_stage)[:5]
        # print(f"\nTop 5 Genes with highest PCC in {selection_id}:")
        # for idx in top5_idx:
        #     print(f"{used_genes[idx]}: PCC = {pr_stage[idx]:.4f}")
    return adata_stage,np.mean(pr_stage_common), np.mean(mse_values), np.mean(mae_values),  #pr_stage_common

if __name__ == '__main__':

    # seed_list = [1, 8, 24, 50, 222, 333, 2001, 2048,3234,761]
    seed_list=[8]
    # num_runs = 1
    num_runs = len(seed_list)
   
    all_pcc_list = []
    all_mse_list = []
    all_mae_list = []

    section_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                   "151675", "151676"]
    for run_idx, seed in enumerate(seed_list):
        setup_seed(seed)
        print(f"Running iteration {run_idx + 1}/{num_runs}")

        pcc_list = []
        mse_list = []
        mae_list = []
        for section_id in section_list:
            data_manager = SpatialDataManager(selection_id =section_id, train_ratio=0.5, seed=42, neighbor_ratio=6)
            train_dataset =  data_manager.get_train_dataset()
            test_dataset = data_manager.get_test_dataset()
            num_genges = train_dataset.data.shape[1]
            train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=True)
            train(selection_id=section_id,num_genges= num_genges,train_loader=train_loader)

            test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4, shuffle=True)
            model = torch.load(section_id+"_train_"+".ckpt")
            pred_adata,pcc,mse,mae= test_model(model, test_loader,section_id, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            train_counts = data_manager.count_mtx.loc[data_manager.train_mask, data_manager.selected_genes].values
            train_coords = data_manager.in_tissue_coords [data_manager.train_mask]
            train_spatials = data_manager.spot_spatials[data_manager.train_mask]
            train_adata = sc.AnnData(train_counts)
            train_adata.obsm["coord"] = train_coords
            train_adata.obsm["spatial"] = train_spatials
            # train_adata.obs["tumor"] = data_manager.labels[data_manager.train_mask]
            train_adata.var_names = data_manager.selected_genes 
            # # 拼接 test + train
            final_adata = pred_adata.concatenate(train_adata, index_unique=None)
            # 保存 final_adata
            # final_adata.write(section_id + "_final_adata.h5ad")
            # train_adata.write(section_id + "_train_adata.h5ad")
            # pred_adata.write(section_id + "_final_adata.h5ad")
            # train_adata.write(section_id + "_train_adata.h5ad")


            # all_pcc_table[section_id] = pcc_values.tolist()

            pcc_list.append(pcc)
            mse_list.append(mse)
            mae_list.append(mae)
            
        all_pcc_list.append(pcc_list)
        all_mse_list.append(mse_list)
        all_mae_list.append(mae_list)
    #
    pcc_array = np.array(all_pcc_list)  
    mse_array = np.array(all_mse_list)
    mae_array = np.array(all_mae_list)
    
    
    pcc_mean = np.mean(pcc_array, axis=0)
    pcc_std = np.std(pcc_array, axis=0)
    mse_mean = np.mean(mse_array, axis=0)
    mse_std = np.std(mse_array, axis=0)
    mae_mean = np.mean(mae_array, axis=0)
    mae_std = np.std(mae_array, axis=0)
    #
   
    results_dict = {}
    for i, section_id in enumerate(section_list):
        pcc_str = f"{pcc_mean[i]:.4f} ± {pcc_std[i]:.4f}"
        mse_str = f"{mse_mean[i]:.4f} ± {mse_std[i]:.4f}"
        mae_str = f"{mae_mean[i]:.4f} ± {mae_std[i]:.4f}"
        results_dict[section_id] = [pcc_str, mse_str, mae_str]
    #
    df = pd.DataFrame(results_dict, index=["PCC", "MSE", "MAE"])
    df.to_csv("GAT_DLPFC_results.csv")
    #
    # print("实验完成，结果已保存")
    # df_pcc_table = pd.DataFrame.from_dict(all_pcc_table, orient='index')
    # df_pcc_table.index.name = 'Dataset'
    #     # df_pcc_table.to_csv(f"BCA_PCC_MP_table.csv")
    #
    #     df = pd.DataFrame({
    #         section_list[i]: [pcc_list[i], mse_list[i], mae_list[i]]
    #         for i in range(len(section_list))
    #     }, index=["PCC", "MSE", "MAE"])
    #     df = df.round(4)

    #     df.to_csv("VisiumHD_results_results.csv")
