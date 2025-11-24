import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataset import *
from torch.nn import MultiheadAttention

from transformer import Transformer


class SparseGraphAttentionLayer(nn.Module):
    """显存友好的稀疏版 GAT 层"""
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.01, concat=True):
        super(SparseGraphAttentionLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a_src = nn.Parameter(torch.empty(out_features, 1))
        self.a_dst = nn.Parameter(torch.empty(out_features, 1))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.concat = concat
        self.dropout = dropout

    def forward(self, h, edge_index):
        # h: [N, F], edge_index: [2, E]
        Wh = self.W(h)  # [N, F']
        src, dst = edge_index
        Wh_src = Wh[src]
        Wh_dst = Wh[dst]

        e = self.leakyrelu((Wh_src @ self.a_src + Wh_dst @ self.a_dst).squeeze(-1))
        # 对每个目标节点归一化
        attention = torch.zeros_like(e)
        num_nodes = h.size(0)
        attention = torch.exp(e - e.max())
        attention_sum = torch.zeros(num_nodes, device=h.device).index_add_(0, dst, attention)
        attention = attention / (attention_sum[dst] + 1e-9)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # 聚合邻居
        out = torch.zeros_like(Wh)
        out.index_add_(0, dst, attention.unsqueeze(-1) * Wh_src)
        return F.elu(out) if self.concat else out
class GAT_Regressor(nn.Module):
    """显存友好的 GAT 回归模型"""
    def __init__(self, nfeat, nhid, nout, nheads=4, dropout=0.2, alpha=0.01):
        super(GAT_Regressor, self).__init__()
        self.attentions = nn.ModuleList([
            SparseGraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True)
            for _ in range(nheads)
        ])
        self.out_att = SparseGraphAttentionLayer(nhid * nheads, nout, dropout, alpha, concat=False)
        self.dropout = dropout
        self.gene_head = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, nfeat)
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, edge_index) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        emb = self.out_att(x, edge_index)
        gene = self.gene_head(emb)
        return emb,gene

class Feed(nn.Module):
    def __init__(self, gene_number, X_dim):
        super(Feed, self).__init__()
        self.fc6 = nn.Linear(X_dim, 1024)
        self.fc6_bn = nn.BatchNorm1d(1024)
        self.fc7 = nn.Linear(1024, 2048)
        self.fc7_bn = nn.BatchNorm1d(2048)
        self.fc8 = nn.Linear(2048, 2048)
        self.fc8_bn = nn.BatchNorm1d(2048)
        self.fc9 = nn.Linear(2048, gene_number)

    def forward(self, z, relu):
        h6 = F.relu(self.fc6_bn(self.fc6(z)))
        h7 = F.relu(self.fc7_bn(self.fc7(h6)))
        h8 = F.relu(self.fc8_bn(self.fc8(h7)))
        if relu:
            return F.relu(self.fc9(h8))
        else:
            return self.fc9(h8)


class CrossAttentionLayer(nn.Module):
    def __init__(self, embed_dim, q_dim, kv_dim, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.query_proj = nn.Linear(q_dim, embed_dim)
        self.key_proj = nn.Linear(kv_dim, embed_dim)
        self.value_proj = nn.Linear(kv_dim, embed_dim)

        self.output_proj = nn.Linear(embed_dim, embed_dim)

        # 添加LayerNorm和Dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, key, value, mask=None):
        """
        :param query: (batch_size, query_len, embed_dim)
        :param key: (batch_size, key_len, embed_dim)
        :param value: (batch_size, value_len, embed_dim)
        :param mask: (batch_size, query_len, key_len)
        :return: output, attention_weights
        """
        B, N1, _ = query.shape
        _, N2, _ = key.shape

        Q = self.query_proj(query).reshape(B, N1, self.num_heads, int(self.embed_dim / self.num_heads)).permute(0, 2, 1, 3)  # (batch, num_heads, N, dim)
        K = self.key_proj(key).reshape(B, N2, self.num_heads, int(self.embed_dim / self.num_heads)).permute(0, 2, 1, 3)  # (batch, num_heads, N, dim)
        V = self.value_proj(value).reshape(B, N2, self.num_heads, int(self.embed_dim / self.num_heads)).permute(0, 2, 1, 3)  # (batch, num_heads, N, dim)

        att = (Q @ K.transpose(-2, -1)) * self.scale  # (batch, num_heads, N, N)
        att = att.softmax(dim=-1)
        att = self.dropout(att)
        attention_output = (att @ V).transpose(1, 2).flatten(2)  # B,N,dim

        output_ = self.output_proj(attention_output)  # (batch_size, query_len, embed_dim)
        output_ = self.norm(output_)
        return output_

class SpaBiT(nn.Module):
    def __init__(self, in_features, depth, trans_heads, num_genes=1000, dropout=0.):
        super(SpaBiT, self).__init__()
        self.hidden_dim = in_features
        self.x_embed = nn.Embedding(512, in_features)
        self.y_embed = nn.Embedding(512, in_features)
        self.trans = Transformer(dim=in_features, depth=depth, heads=trans_heads, dim_head=64, mlp_dim=in_features,
                                 dropout=dropout)
        self.crossAttention = MultiheadAttention(1024, num_heads=8, batch_first=True)
        self.cross_attention1 = CrossAttentionLayer(embed_dim=in_features, q_dim=self.hidden_dim, kv_dim=self.hidden_dim,
                                                    num_heads=8)
        self.cross_attention2 = CrossAttentionLayer(embed_dim=in_features, q_dim=self.hidden_dim, kv_dim=self.hidden_dim,
                                                    num_heads=8)
        self.fusion_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.gene_head = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, num_genes)
        )
        self.neighbor_emb = nn.Sequential(
            nn.Linear(num_genes, 2048),
            nn.ReLU(),
            nn.LayerNorm(2048),
            nn.Linear(2048, in_features)
        )

    def forward(self, local_emb, neighbor,coord):
        # centers_x = self.x_embed(coord[:, 0].long())
        # centers_y = self.y_embed(coord[:, 1].long())
        # pos = (centers_y + centers_x).unsqueeze(0)
        local_emb = local_emb.unsqueeze(1)
        # neighbor = neighbor.unsqueeze(1)

        neighbor = self.neighbor_emb(neighbor).unsqueeze(1)

        img_attn = self.cross_attention1(local_emb, neighbor, neighbor)
        gene_attn = self.cross_attention2(neighbor, local_emb, local_emb)

        fused_feat = self.fusion_proj(torch.cat([img_attn, gene_attn], dim=-1))
        fused_feat = torch.permute(fused_feat, (1, 0, 2))

        # x = pos + local_emb
        # h = self.trans(x)
        h = self.trans(fused_feat)

        x = self.gene_head(h)
        x = x.squeeze(0)

        return h, F.relu(x)
