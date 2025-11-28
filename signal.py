# coding: utf-8
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from abstract_recommender import GeneralRecommender
from loss import EmbLoss
import math
import os
import torch.nn.functional as F

class LAD(nn.Module):
    def __init__(self, emb_dim=64, time_emb_dim=128, hidden_dim=256, num_modalities=4,
                 num_attention_heads=1, num_transformer_blocks=1, 
                 dim_feedforward_ratio=1, 
                 dropout=0.1):
    
        super().__init__()

        self.emb_dim = emb_dim
        self.num_modalities = num_modalities
        self.time_emb_dim = time_emb_dim
        self.hidden_dim = hidden_dim  

        self.register_buffer('emb_factors', self._precompute_emb_factors(self.time_emb_dim))

        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )

        self.initial_fusion_net = nn.Sequential(
            nn.Linear(emb_dim + time_emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,  
            nhead=num_attention_heads,  
            dim_feedforward=hidden_dim * dim_feedforward_ratio,  
            dropout=dropout,
            batch_first=True 
        )
        self.transformer_blocks = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_transformer_blocks  
        )

        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, emb_dim), 
        )

    def _precompute_emb_factors(self, time_emb_dim):

        half_dim = time_emb_dim // 2
        if half_dim == 0:
            return torch.empty(0)

        indices = torch.arange(half_dim)
        denom = max(half_dim - 1, 1.0)  
        val = math.log(10000) / denom
        return torch.exp(-indices * val)

    def forward(self, x, t):
        
        t_float = t.float().unsqueeze(-1)  

        emb_prod = t_float * self.emb_factors.unsqueeze(0)

        t_emb_sincos = torch.cat([torch.sin(emb_prod), torch.cos(emb_prod)], dim=-1)


        t_emb = self.time_proj(t_emb_sincos)  # [N*4, time_emb_dim]


        x_fused = torch.cat([x, t_emb], dim=1)  # [N*4, emb_dim + time_emb_dim]

  
        x_processed = self.initial_fusion_net(x_fused)  # [N*4, hidden_dim]

       
        batch_size_original = x.shape[0] // self.num_modalities
        
        x_reshaped_for_attn = x_processed.view(batch_size_original, self.num_modalities, self.hidden_dim)

        
        x_attended = self.transformer_blocks(x_reshaped_for_attn)  # [N, 4, hidden_dim]

        
        x_output_flat = x_attended.view(x.shape[0], self.hidden_dim)  # [N*4, hidden_dim]

     
        output = self.output_net(x_output_flat)  # [N*4, emb_dim]

        return output

class SIGNAL(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SIGNAL, self).__init__(config, dataset)

        self.embedding_dim = config['embedding_size']
        self.lambda_coeff = config['lambda_coeff']
        self.cf_model = config['cf_model']

        self.Heteromodal_coeff = config['Heteromodal_coeff']
        self.knn_k = config['knn_k']
        self.rho = config['rho']
        self.n_layers = config['n_mm_layers']

        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.kl_weight = config['kl_weight']
        self.intra_weight = config['intra_weight']
        self.conflicts_weights = config['conflicts_weights']

        self.cl_a = config['cl_a']
        self.cl_r = config['cl_r']
        self.cl_d = config['cl_d']
        self.dropout = config['dropout']
        self.build_item_graph = True
        self.noise_ratio = config['noise_ratio']  

        self.n_nodes = self.n_users + self.n_items

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        self.reg_loss = EmbLoss()

        if self.v_feat is not None:
            #self.v_feat = self.add_noise_to_features(self.v_feat)
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=True)
            self.image_trs =  nn.Linear(self.v_feat.shape[1], 256)
            nn.init.xavier_normal_(self.image_trs.weight)

        if self.t_feat is not None:
            #self.t_feat = self.add_noise_to_features(self.t_feat)
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=True)
            self.text_trs = nn.Linear(self.t_feat.shape[1], 256)
            nn.init.xavier_normal_(self.text_trs.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.item_graph_dict = np.load(os.path.join(dataset_path, config['item_graph_dict_file']),allow_pickle=True).item()
        __, self.session_adj = self.get_session_adj()
        self.image_adj, self.text_adj = self.get_knn_adj_mat_denoise(self.image_embedding.weight.detach(), self.text_embedding.weight.detach(),self.session_adj)

        self.image_adj_copy = self.image_adj
        self.text_adj_copy = self.text_adj
        self.session_adj_copy = self.session_adj

        self.text_adj = self.Heteromodal_information_introduction(self.text_adj_copy, self.session_adj_copy, self.image_adj_copy)
        self.session_adj = self.Heteromodal_information_introduction(self.session_adj_copy, self.image_adj_copy, self.text_adj_copy)
        self.image_adj = self.Heteromodal_information_introduction(self.image_adj_copy, self.session_adj_copy, self.text_adj_copy)
        self.fusion_adj =  ((self.text_adj+ self.image_adj + self.session_adj)/3).coalesce()
        self.gate_layer = nn.Linear(64 * 3, 3)
        self.gate_layer_ii = nn.Linear(64 * 2, 2)

        self.noise_coeff = config['noise_coeff']
        self.diff_weight = config['diff_weight']
        self.use_diffusion = config['use_diffusion']
        self.diffusion_steps = config['diffusion_steps'] 
        self.denoise_steps = config['denoise_steps'] 
        self.denoise_net = LAD() if self.use_diffusion else None
        self.register_buffer('alphas_cumprod', self._linear_schedule(self.diffusion_steps))

        self.res_uu = config['res_uu']
        self.res_ii = config['res_ii']
        self.res_dif = config['res_dif']


    def add_noise_to_features(self, features):
        if self.noise_ratio <= 0:
            return features

        std = torch.std(features)
        base_noise = torch.randn_like(features) #* std

        noisy_features = features * (1 - self.noise_ratio) + base_noise * self.noise_ratio

        return noisy_features

    def _linear_schedule(self, T):
        beta_start = 0.001 * self.noise_coeff
        beta_end   = 0.02 * self.noise_coeff

        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1 - betas
        return torch.cumprod(alphas, dim=0)  
    def denoise_step_one(self, x, t,alpha_t):
        pred_noise = self.denoise_net(x, t)
       
        x0_hat = (x - torch.sqrt(1 - alpha_t) * pred_noise) / (torch.sqrt(alpha_t) +  0)
        del pred_noise
        return x0_hat

    def denoise_step_mul(self, x, t, next_t=None):
        pred_noise = self.denoise_net(x, t)
        alpha_t = self.alphas_cumprod[t].view(-1, 1)

        if next_t[:1] == 0:  
            x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / (torch.sqrt(alpha_t) + 0)
            return x0
           
        alpha_next = self.alphas_cumprod[next_t].view(-1, 1)
        x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / (torch.sqrt(alpha_t) + 0)
        x = torch.sqrt(alpha_next) * x0 + torch.sqrt(1 - alpha_next) * pred_noise
   
        return x

    def denoise(self,x):
        indices = list(reversed(range(0, self.denoise_steps+1)))
        for i_loop in range(len(indices)):
            step = torch.tensor([indices[i_loop]], dtype=torch.long, device=x.device).expand(x.size(0))  # 转换为 Tensor
            if i_loop + 1 < len(indices):
                next_t = indices[i_loop + 1]
            else:
                break
            next_t = torch.tensor([next_t], dtype=torch.long, device=x.device).expand(x.size(0))
            x_hat = self.denoise_step_mul(x, step, next_t)
        
        return x_hat

    def gen_noise(self,x,i,t,v,s):
       
        ori_features = [i,t,v,s]
        size =i.shape[0]
        coefficient_matrices = [
            torch.randn((size,1)).to(self.device)  
            for _ in range(4)
        ]

        new_feature = sum(
            feature * matrix
            for feature, matrix in zip(ori_features, coefficient_matrices)
        )
        noise = torch.cat((new_feature,new_feature,new_feature,new_feature), dim=0)

        return noise

    def SNR(self, t):
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])

    def Heteromodal_information_introduction(self, cur_modal, het_modal1, het_modal2):
     
        attn_het1 = self.calc_structure_attention(cur_modal, het_modal1)
        attn_het2 = self.calc_structure_attention(cur_modal, het_modal2)

        attn_self = torch.full((cur_modal.size(0),1), (1), device=cur_modal.device)

        attention_scores = torch.cat([attn_self, attn_het1, attn_het2], dim=1)  # [N, 3]

        weights = torch.softmax(attention_scores, dim=1)
        w_self, w_het1, w_het2 = weights[:, 0], weights[:, 1], weights[:, 2]

        cur_dense = cur_modal.to_dense()
        fuse = (w_self.unsqueeze(1) * cur_dense +
                  w_het1.unsqueeze(1) * het_modal1.to_dense() +
                  w_het2.unsqueeze(1) * het_modal2.to_dense())

        result = self.Heteromodal_coeff * cur_dense + ( 1 - self.Heteromodal_coeff ) * fuse

        return result.to_sparse().coalesce()

    def calc_structure_attention(self, source, target):
        keep_src = self.keep_topk_mask(source,self.knn_k)
        keep_tgt = self.keep_topk_mask(target,self.knn_k)

        raw_attn = torch.sum((keep_src * keep_tgt).to_dense() == 1, dim=1).unsqueeze(1) / self.knn_k

        return raw_attn

    def keep_topk_mask(self, x, k):
        x = x.to_dense()
        values, indices = torch.topk(x, k=k, dim=1)
        mask = torch.zeros_like(x)
        mask.scatter_(1, indices, 1)
        return mask
    
    def get_knn_adj_mat_denoise(self, v_embeddings, t_embeddings, adj_s):
        v_context_norm = v_embeddings.div(torch.norm(v_embeddings, p=2, dim=-1, keepdim=True))
        v_sim = torch.mm(v_context_norm, v_context_norm.transpose(1, 0))

        t_context_norm = t_embeddings.div(torch.norm(t_embeddings, p=2, dim=-1, keepdim=True))
        t_sim = torch.mm(t_context_norm, t_context_norm.transpose(1, 0))

        mask_v = v_sim <  v_sim.mean(dim=1)
        mask_t = t_sim <  t_sim.mean(dim=1)

        t_sim[mask_v] = 0
        v_sim[mask_t] = 0
        t_sim[mask_t] = 0
        v_sim[mask_v] = 0

        index_x = []
        index_v = []
        index_t = []


        for i in range(self.n_items):
            item_num = len(torch.nonzero(t_sim[i]))
            if item_num <= self.knn_k:
                _, v_knn_ind = torch.topk(v_sim[i], item_num)
                _, t_knn_ind = torch.topk(t_sim[i], item_num)
            else:
                _, v_knn_ind = torch.topk(v_sim[i], self.knn_k)
                _, t_knn_ind = torch.topk(t_sim[i], self.knn_k)

            index_x.append(torch.ones_like(v_knn_ind) * i)
            index_v.append(v_knn_ind)
            index_t.append(t_knn_ind)

        index_x = torch.cat(index_x, dim=0).cuda()
        index_v = torch.cat(index_v, dim=0).cuda()
        index_t = torch.cat(index_t, dim=0).cuda()

        adj_size = (self.n_items, self.n_items)
        del v_sim, t_sim

        v_indices = torch.stack((torch.flatten(index_x), torch.flatten(index_v)), 0)
        t_indices = torch.stack((torch.flatten(index_x), torch.flatten(index_t)), 0)

        return self.compute_normalized_laplacian(v_indices, adj_size), self.compute_normalized_laplacian(t_indices, adj_size)


    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        #print("row_sum: ", row_sum)
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        #print("values: ", values)
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_session_adj(self):
        index_x = []
        index_y = []
        values = []
        for i in range(self.n_items):
            index_x.append(i)
            index_y.append(i)
            values.append(1)
            if i in self.item_graph_dict.keys():
                item_graph_sample = self.item_graph_dict[i][0]
                item_graph_weight = self.item_graph_dict[i][1]

                for j in range(len(item_graph_sample)):
                    index_x.append(i)
                    index_y.append(item_graph_sample[j])
                    values.append(item_graph_weight[j])
        index_x = torch.tensor(index_x, dtype=torch.long)
        index_y = torch.tensor(index_y, dtype=torch.long)
        indices = torch.stack((index_x, index_y), 0).to(self.device)
        # norm
        return indices, self.compute_normalized_laplacian(indices, (self.n_items, self.n_items))

    def DCL(self, sim1, sim2, sim3, sim4,temperature=0.2):
        def batch_topk(matrix, k):
            k2 = 2 * k
            topk_vals, _ = torch.topk(matrix, k2 + 1, dim=1)
            return topk_vals[:, k], topk_vals[:, k2]

        with torch.no_grad():
            strong_sim, weak_sim = zip(*[batch_topk(s, self.knn_k) for s in [sim1, sim2, sim3, sim4]])

            strong_mask = torch.stack([
                sim1 > strong_sim[0].unsqueeze(1),
                sim2 > strong_sim[1].unsqueeze(1),
                sim3 > strong_sim[2].unsqueeze(1),
                sim4 > strong_sim[3].unsqueeze(1)
            ]).to(torch.int8)  # 使用int8节省内存

            weak_mask = torch.stack([
                sim1 > weak_sim[0].unsqueeze(1),
                sim2 > weak_sim[1].unsqueeze(1),
                sim3 > weak_sim[2].unsqueeze(1),
                sim4 > weak_sim[3].unsqueeze(1)
            ]).to(torch.int8)

            # 
            grade = (strong_mask + weak_mask).sum(dim=0)

            # 
            alls_mask = grade >= self.rho
            mul_mask = grade >= 2  

        exp_scores = (
                torch.exp(sim1 / temperature) +
                torch.exp(sim2 / temperature) +
                torch.exp(sim3 / temperature)
                + torch.exp(sim4 / temperature)

        )

    
        all_score = torch.where(alls_mask, exp_scores, 0).sum(dim=1)
        mul_score =  torch.where(mul_mask, exp_scores, 0).sum(dim=1)
        ttl_score = exp_scores.sum(dim=1) 
        
        term1 = all_score / ttl_score
        # term2 = mul_score / ttl_score
        term2 = (mul_score-all_score) / (ttl_score-all_score)
       
        cl_loss = -torch.log(term1) - torch.log(term2) 
        return cl_loss.mean()

    def Basic_CL(self, e_id, e_mul, temperature=0.2):

        e_id_e_mul_T = torch.mm(e_id, e_mul.transpose(0, 1))  # [B, B]
        total_score_matrix = torch.exp(e_id_e_mul_T / temperature)

        diag_sum = torch.einsum('ij,ij->i', e_id, e_mul)
        pos_score = torch.exp(diag_sum / temperature)  # [B]

        total_score = total_score_matrix.sum(dim=1)  # [B]

        epsilon = 1e-10
        term1 = pos_score / (total_score  + epsilon) + epsilon
        loss = -self.cl_a * torch.log(term1)
        return loss.mean()

    def gate_fuse(self, e_v, e_t, e_s):
        combined = torch.cat([e_v, e_t, e_s], dim=1)  # [batch_size, 64*3]
        gates = torch.sigmoid(self.gate_layer(combined))  # [batch_size, 3]
        gated_v = gates[:, 0:1] * e_v
        gated_t = gates[:, 1:2] * e_t
        gated_s = gates[:, 2:3] * e_s
        #print(gates[0])   ##[0.9787, 0.9867, 0.9919]
        fused = gated_v + gated_t + gated_s
        return fused

    def gate_fuse_ii(self, i1, i2):
        combined = torch.cat([i1, i2], dim=1)  # [batch_size, 64*2]
        gates = torch.sigmoid(self.gate_layer_ii(combined))  # [batch_size, 2]
        gated_i1 = gates[:, 0:1] * i1
        gated_i2 = gates[:, 1:2] * i2
        fused = gated_i1 + gated_i2
        return fused

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix

        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users),
                             [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col),
                                  [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def diff_agg(self,i,t,v,s):
        batch_m = i.size(0)
        e_i,e_t,e_v,e_s = i.detach(),t.detach(),v.detach(),s.detach()
        tar_mule = self.gate_fuse(e_t, e_v, e_s)
        all_tar_e = self.gate_fuse_ii(tar_mule, e_i)
        e_mul= torch.cat((e_i,e_t,e_v,e_s), dim=0)

        e_guide_4 = torch.cat((e_i, all_tar_e, all_tar_e, all_tar_e), dim=0)

        de_mul = self.denoise(e_mul)
        d_i, d_t, d_v, d_s = torch.split(de_mul, [batch_m] * 4, dim=0)

        step = torch.randint(1, self.denoise_steps+1, (e_mul.size(0),)).to(self.device)
        alpha_t = self.alphas_cumprod[step].view(-1, 1)
    
        noise = self.gen_noise(e_mul, e_i, e_t, e_v, e_s)
        noisy_emb = torch.sqrt(alpha_t) * e_mul + torch.sqrt(1 - alpha_t) * noise

        pred_e = self.denoise_step_one(noisy_emb, step, alpha_t)

        diff_weight = self.SNR(step - 1) - self.SNR(step)
        diff_weight = torch.where((step == 0), 1.0, diff_weight)
        diff_loss = (diff_weight * (F.mse_loss(pred_e, e_guide_4, reduction='none').mean(dim=1))).mean()
       
        add_all_ori_e = i + (t + v + s) / 3
        add_all_de_e = d_i + (d_t + d_v + d_s) / 3

        return self.res_dif * add_all_de_e + (1 - self.res_dif) * add_all_ori_e ,  diff_loss

    def forward(self):
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        del ego_embeddings, side_embeddings

        h_t = self.item_id_embedding.weight.clone()
        h_v = self.item_id_embedding.weight.clone()
        h_s = self.item_id_embedding.weight.clone()
        h_id = i_g_embeddings

        for i in range(self.n_layers):
            h_id = torch.sparse.mm(self.fusion_adj, i_g_embeddings)
            h_t = torch.sparse.mm(self.text_adj, h_t)
            h_v = torch.sparse.mm(self.image_adj, h_v)
            h_s = torch.sparse.mm(self.session_adj, h_s)

        return u_g_embeddings, (1 - self.res_ii) * i_g_embeddings + self.res_ii * h_id, h_t, h_v, h_s

    def calculate_loss(self, interaction):
        users = interaction[0]     
        pos_items = interaction[1]
        neg_items = interaction[2]

        user_embeddings, item_embeddings, h_t, h_v, h_s = self.forward()
        u_g_embeddings = user_embeddings[users]
        self.build_item_graph = False

        #u_idx = torch.unique(users, return_inverse=True, sorted=False) 
        i_idx = torch.unique(torch.cat((pos_items, neg_items)), return_inverse=True, sorted=False)
        #u_id = u_idx[0]
        i_id = i_idx[0]

        h_t_id_norm = F.normalize(h_t[i_id])
        h_v_id_norm = F.normalize(h_v[i_id])
        h_s_id_norm = F.normalize(h_s[i_id])
        hid_id_norm = F.normalize(item_embeddings[i_id])
       

        bcl_loss1 = self.Basic_CL(hid_id_norm, h_t_id_norm)
        bcl_loss2 = self.Basic_CL(hid_id_norm, h_v_id_norm)
        bcl_loss3 = self.Basic_CL(hid_id_norm, h_s_id_norm)
        bcl_loss = (bcl_loss1 + bcl_loss2 + bcl_loss3)/3
       
        sim_t = torch.mm(h_t_id_norm, h_t_id_norm.T)
        sim_v = torch.mm(h_v_id_norm, h_v_id_norm.T)
        sim_s = torch.mm(h_s_id_norm, h_s_id_norm.T)
        sim_id = torch.mm(hid_id_norm, hid_id_norm.T)
        
        dcl_loss = self.DCL(sim_v, sim_t, sim_s, sim_id)

        diff_e, diff_loss = self.diff_agg(item_embeddings[i_id], h_t[i_id], h_v[i_id], h_s[i_id])
        ia_embeddings = item_embeddings + (h_t + h_v + h_s) / 3.0
        ia_embeddings[i_id] = diff_e

        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss = self.bpr_loss(u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings)

        return batch_mf_loss + self.cl_d * dcl_loss + bcl_loss + self.lambda_coeff * self.reg_loss(u_g_embeddings, ia_embeddings) + self.diff_weight * diff_loss
                                

    def full_sort_predict(self, interaction):
        user = interaction[0]
        user_embeddings, item_embeddings, h_t, h_v, h_s = self.forward()  #
        user_e = user_embeddings[user, :]

        e_mul = torch.cat((item_embeddings, h_t, h_v, h_s), dim=0)
        de_mul= self.denoise(e_mul)
        item_embeddings, h_t, h_v, h_s = torch.split(de_mul, [self.n_items] * 4, dim=0)


        all_item_e = item_embeddings + (h_v + h_t + h_s) / 3.0

        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        p_maxi = torch.log(F.sigmoid(pos_scores - neg_scores))
        mf_loss = -torch.mean(p_maxi )

        return mf_loss