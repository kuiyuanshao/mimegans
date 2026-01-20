# networks.py: Neural network backbone with Analog Bits Self-Conditioning.
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
        y = x + self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.mid_projection(y) + self.cond_projection(cond_info)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = self.output_projection(torch.sigmoid(gate) * torch.tanh(filter))
        res, skip = torch.chunk(y, 2, dim=1)
        return (x + res) / math.sqrt(2.0), skip


class Diff_TwoPhase(nn.Module):
    def __init__(self, config_diff, inputdim=3):
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
    def __init__(self, target_dim, config, device, matched_state, p1_indices, p2_indices, is_bit, cat_groups):
        super().__init__()
        self.device, self.target_dim = device, target_dim
        self.p1_indices, self.p2_indices = p1_indices, p2_indices
        self.cat_groups = cat_groups
        self.register_buffer("matched_state_t", torch.tensor(matched_state).long())
        self.register_buffer("is_bit_t", torch.tensor(is_bit).bool())
        self.max_step = int(torch.max(self.matched_state_t).item())

        self.register_buffer("cond_mask", torch.ones(1, self.target_dim))
        self.cond_mask[:, self.p2_indices] = 0.0
        self.residual_pools = None

        self.emb_feature_dim = config["model"]["featureemb"]
        self.embed_layer = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim)

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_feature_dim + 1
        self.num_steps = config_diff["num_steps"]
        self.diffmodel = Diff_TwoPhase(config_diff, inputdim=3)

        betas = np.linspace(config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps) ** 2
        self.alpha = np.cumprod(1. - betas)
        self.register_buffer("alpha_torch", torch.tensor(self.alpha).float())

    def forward(self, observed_data):
        """ Primary execution path for training. """
        return self.calc_loss(observed_data)

    def get_static_side_info(self, B):
        feat = self.embed_layer(torch.arange(self.target_dim).to(self.device)).unsqueeze(0).expand(B, -1, -1).permute(0,
                                                                                                                      2,
                                                                                                                      1)
        role = torch.zeros(B, 1, self.target_dim).to(self.device)
        role[:, 0, self.p2_indices] = 1.0
        return torch.cat([feat, role], dim=1)

    def calc_loss(self, observed_data):
        """ Implements Self-Conditioning recursive loss (Chen et al., 2023). """
        B = observed_data.shape[0]
        t = torch.randint(0, self.max_step + 1, [B], device=self.device)

        # Pull Audit Residuals
        rand_idx = torch.randint(0, self.residual_pools.shape[0], (B, len(self.p2_indices)), device=self.device)
        epsilon_audit = torch.gather(self.residual_pools, 0, rand_idx)

        a_t = self.alpha_torch[t].unsqueeze(1)
        p2_noisy = (a_t ** 0.5) * observed_data[:, self.p2_indices] + ((1 - a_t) ** 0.5) * epsilon_audit

        cond_obs = observed_data * self.cond_mask
        target_noisy = torch.zeros_like(observed_data)
        target_noisy[:, self.p2_indices] = p2_noisy

        self_cond = torch.zeros_like(observed_data)

        # Self-Conditioning Training Pass
        if torch.rand(1) < 0.5:
            with torch.no_grad():
                inp1 = torch.cat([cond_obs.unsqueeze(1), target_noisy.unsqueeze(1), self_cond.unsqueeze(1)], dim=1)
                eps1 = self.diffmodel(inp1, self.get_static_side_info(B), t)
                # Bridge: Derive x0 estimate from predicted noise
                x0_est = (target_noisy[:, self.p2_indices] - (1 - a_t) ** 0.5 * eps1[:, self.p2_indices]) / (a_t ** 0.5)
                self_cond[:, self.p2_indices] = x0_est.detach()

        # Final Training Pass
        inp2 = torch.cat([cond_obs.unsqueeze(1), target_noisy.unsqueeze(1), self_cond.unsqueeze(1)], dim=1)
        eps_final = self.diffmodel(inp2, self.get_static_side_info(B), t)

        return F.mse_loss(eps_final[:, self.p2_indices], epsilon_audit)

    def impute_single(self, observed_data, observed_mask):
        """ Denoising with recursive x0 feedback. """
        B = observed_data.shape[0]
        side_info = self.get_static_side_info(B)
        cur, p1_obs = observed_data.clone(), observed_data[:, self.p1_indices]

        # Initialize at T
        rand_idx = torch.randint(0, self.residual_pools.shape[0], (B, len(self.p2_indices)), device=self.device)
        noise_init = torch.gather(self.residual_pools, 0, rand_idx)
        cur[:, self.p2_indices] = (self.alpha_torch[self.max_step] ** 0.5) * p1_obs + (
                    (1 - self.alpha_torch[self.max_step]) ** 0.5) * noise_init

        x_pred_accum = torch.zeros_like(cur)

        for t in range(self.max_step - 1, -1, -1):
            inp = torch.cat([(observed_data * self.cond_mask).unsqueeze(1),
                             (cur * (1 - self.cond_mask)).unsqueeze(1),
                             x_pred_accum.unsqueeze(1)], dim=1)

            eps = self.diffmodel(inp, side_info, torch.tensor([t], device=self.device))[:, self.p2_indices]
            a_t = self.alpha_torch[t]

            # Recursive x0 feedback update
            x_pred_accum[:, self.p2_indices] = (cur[:, self.p2_indices] - (1 - a_t) ** 0.5 * eps) / (a_t ** 0.5)

            # Inverse Step
            cur[:, self.p2_indices] = (cur[:, self.p2_indices] - (1 - a_t) / (1 - a_t) ** 0.5 * eps) / (a_t ** 0.5)
            if t > 0:
                cur[:, self.p2_indices] += (1 - a_t) ** 0.5 * torch.randn_like(cur[:, self.p2_indices])

        # Hardening bits at the final step
        cur[:, self.p2_indices] = torch.where(self.is_bit_t.unsqueeze(0),
                                              (cur[:, self.p2_indices] > 0.0).float() * 2 - 1,
                                              cur[:, self.p2_indices])

        # Transductive ground-truth assignment
        audit_rows = (observed_mask[:, self.p2_indices[0]] == 1).nonzero(as_tuple=True)[0]
        if len(audit_rows) > 0:
            cur[audit_rows.view(-1, 1), torch.tensor(self.p2_indices, device=self.device).view(1, -1)] = \
            observed_data[audit_rows][:, self.p2_indices]

        return cur