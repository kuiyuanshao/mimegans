# networks.py: Optimized neural network backbone with single-sample generation.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu", batch_first=True
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128):
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(num_steps, embedding_dim // 2), persistent=False)
        self.projection1 = nn.Linear(embedding_dim, embedding_dim)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        return self.projection2(F.silu(self.projection1(x)))

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K = x.shape
        y = x + self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.mid_projection(y) + self.cond_projection(cond_info)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = self.output_projection(torch.sigmoid(gate) * torch.tanh(filter))
        res, skip = torch.chunk(y, 2, dim=1)
        return (x + res) / math.sqrt(2.0), skip


class Diff_TwoPhase(nn.Module):
    def __init__(self, config_diff, inputdim=2):
        super().__init__()
        self.channels = config_diff["channels"]
        self.num_steps = config_diff["num_steps"]
        self.diffusion_embedding = DiffusionEmbedding(self.num_steps, config_diff["diffusion_embedding_dim"])
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        self.residual_layers = nn.ModuleList([
            ResidualBlock(config_diff["side_dim"], self.channels, config_diff["diffusion_embedding_dim"],
                          config_diff["nheads"])
            for _ in range(config_diff["layers"])
        ])

    def forward(self, x, cond_info, diffusion_step):
        x = F.relu(self.input_projection(x))
        diff_emb = self.diffusion_embedding(diffusion_step)
        skip_list = []
        for layer in self.residual_layers:
            x, skip = layer(x, cond_info, diff_emb)
            skip_list.append(skip)
        x = torch.sum(torch.stack(skip_list), dim=0) / math.sqrt(len(self.residual_layers))
        return self.output_projection2(F.relu(self.output_projection1(x))).squeeze(1)


class CSDI_TwoPhase(nn.Module):
    def __init__(self, target_dim, config, device, matched_state, p1_indices, p2_indices, is_binary, cat_groups):
        super().__init__()
        self.device, self.target_dim = device, target_dim
        self.p1_indices, self.p2_indices = p1_indices, p2_indices
        self.cat_groups = cat_groups

        self.register_buffer("matched_state_t", torch.tensor(matched_state).long())
        self.register_buffer("is_binary_t", torch.tensor(is_binary).bool())
        self.max_step = int(torch.max(self.matched_state_t).item())

        self.register_buffer("cond_mask", torch.ones(1, self.target_dim))
        self.cond_mask[:, self.p2_indices] = 0.0
        self.residual_pools = None  # (AuditSize, NumP2Cols) on device

        self.emb_feature_dim = config["model"]["featureemb"]
        self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_feature_dim + 1
        self.num_steps = config_diff["num_steps"]
        self.diffmodel = Diff_TwoPhase(config_diff, inputdim=2)

        self.beta = np.linspace(config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5,
                                self.num_steps) ** 2 if \
            config_diff["schedule"] == "quad" else np.linspace(config_diff["beta_start"], config_diff["beta_end"],
                                                               self.num_steps)
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)

        self.register_buffer("alpha_torch", torch.tensor(self.alpha).float())
        self.register_buffer("alpha_hat_torch", torch.tensor(self.alpha_hat).float())

    def forward(self, observed_data):
        return self.calc_loss(observed_data)

    def get_static_side_info(self, B):
        feat = self.embed_layer(torch.arange(self.target_dim).to(self.device)).unsqueeze(0).expand(B, -1, -1).permute(0,
                                                                                                                      2,
                                                                                                                      1)
        role = torch.zeros(B, 1, self.target_dim).to(self.device)
        role[:, 0, self.p2_indices] = 1.0
        return torch.cat([feat, role], dim=1)

    def calc_loss(self, observed_data):
        B = observed_data.shape[0]
        t = torch.randint(0, self.max_step + 1, [B], device=self.device)
        pool_size = self.residual_pools.shape[0]
        rand_idx = torch.randint(0, pool_size, (B, len(self.p2_indices)), device=self.device)
        sampled_noise = torch.gather(self.residual_pools, 0, rand_idx)
        noise = torch.zeros_like(observed_data)
        p2_obs = observed_data[:, self.p2_indices]
        binary_mask = self.is_binary_t.unsqueeze(0).expand(B, -1)
        binary_noise = sampled_noise * (1 - 2 * p2_obs) + (1 - sampled_noise) * (2 * p2_obs - 1)
        noise[:, self.p2_indices] = torch.where(binary_mask, binary_noise, sampled_noise)
        a_t = self.alpha_torch[t].unsqueeze(1)
        noisy_p2 = (a_t ** 0.5) * p2_obs + ((1 - a_t) ** 0.5) * noise[:, self.p2_indices]
        noisy_target = torch.zeros_like(observed_data)
        noisy_target[:, self.p2_indices] = noisy_p2
        diff_input = torch.cat([(observed_data * self.cond_mask).unsqueeze(1), noisy_target.unsqueeze(1)], dim=1)
        predicted = self.diffmodel(diff_input, self.get_static_side_info(B), t)
        loss = 0
        p2_pred = predicted[:, self.p2_indices]
        p2_target = noise[:, self.p2_indices]
        if (~self.is_binary_t).any():
            loss += F.mse_loss(p2_pred[:, ~self.is_binary_t], p2_target[:, ~self.is_binary_t])
        if self.is_binary_t.any():
            loss += F.binary_cross_entropy_with_logits(p2_pred[:, self.is_binary_t],
                                                       (p2_target[:, self.is_binary_t] + 1) / 2)
        return loss

    def impute_single(self, observed_data, observed_mask):
        """
        Generates ONE Multiple Imputation sample (B, K).
        NO tqdm bars here.
        """
        B, K = observed_data.shape
        side_info = self.get_static_side_info(B)
        audit_rows = (observed_mask[:, self.p2_indices[0]] == 1).nonzero(as_tuple=True)[0]

        alpha_T = self.alpha_torch[self.max_step]
        alpha_m = self.alpha_torch[self.matched_state_t]
        init_coeff = torch.sqrt(alpha_T / alpha_m).unsqueeze(0)
        init_std = torch.sqrt(1 - alpha_T / alpha_m).unsqueeze(0)

        cur = observed_data.clone()
        p1_obs = observed_data[:, self.p1_indices]
        pool_size = self.residual_pools.shape[0]
        rand_idx = torch.randint(0, pool_size, (B, len(self.p2_indices)), device=self.device)
        epsilon = torch.gather(self.residual_pools, 0, rand_idx)
        cur[:, self.p2_indices] = init_coeff * p1_obs + init_std * epsilon

        for t in range(self.max_step - 1, -1, -1):
            inject_mask = self.matched_state_t == (t + 1)
            if inject_mask.any():
                cur[:, self.p2_indices] = torch.where(inject_mask.unsqueeze(0), p1_obs, cur[:, self.p2_indices])

            inp = torch.cat([(observed_data * self.cond_mask).unsqueeze(1), (cur * (1 - self.cond_mask)).unsqueeze(1)],
                            dim=1)
            pred_p2 = self.diffmodel(inp, side_info, torch.tensor([t], device=self.device))[:, self.p2_indices]
            a_hat_t, a_t = self.alpha_hat_torch[t], self.alpha_torch[t]
            c1, c2 = 1 / torch.sqrt(a_hat_t), (1 - a_hat_t) / torch.sqrt(1 - a_t)

            p2_curr = cur[:, self.p2_indices]
            p2_next = torch.zeros_like(p2_curr)
            if (~self.is_binary_t).any():
                p2_next[:, ~self.is_binary_t] = c1 * (
                            p2_curr[:, ~self.is_binary_t] - c2 * pred_p2[:, ~self.is_binary_t])
            if self.is_binary_t.any():
                hat_x0 = torch.sigmoid(pred_p2[:, self.is_binary_t]) * 2 - 1
                derived_noise = (p2_curr[:, self.is_binary_t] - torch.sqrt(a_t) * hat_x0) / torch.sqrt(1 - a_t)
                p2_next[:, self.is_binary_t] = c1 * (p2_curr[:, self.is_binary_t] - c2 * derived_noise)
            if t > 0:
                sigma = torch.sqrt((1 - self.alpha_torch[t - 1]) / (1 - a_t) * (1 - a_hat_t))
                p2_next += sigma * torch.randn_like(p2_next)
            cur[:, self.p2_indices] = p2_next

        cur[:, self.p2_indices] = torch.where(self.is_binary_t.unsqueeze(0),
                                              (torch.sigmoid(cur[:, self.p2_indices]) > 0.5).float(),
                                              cur[:, self.p2_indices])
        for g in self.cat_groups:
            if g:
                probs = F.softmax(cur[:, g], dim=1)
                cur[:, g] = 0.0
                cur[torch.arange(B, device=self.device), torch.tensor(g, device=self.device)[
                    torch.argmax(probs, dim=1)]] = 1.0

        if len(audit_rows) > 0:
            gt_p2 = observed_data[audit_rows][:, self.p2_indices]
            cur[audit_rows.view(-1, 1), torch.tensor(self.p2_indices, device=self.device).view(1, -1)] = gt_p2

        return cur