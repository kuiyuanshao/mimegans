# networks.py
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
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward(self, x, side_info, diffusion_emb):
        B, C, L = x.shape
        x_in = x
        time_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x + time_emb
        y = y + self.cond_projection(side_info)[:, :C, :]
        y = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.mid_projection(y)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x_in + residual) / math.sqrt(2.0), skip


class HybridFeatureEmbedder(nn.Module):
    def __init__(self, schema, channels):
        super().__init__()
        self.schema = schema
        self.projections = nn.ModuleDict()
        for var in schema:
            name = var['name']
            # Input dim is vector size: 1 for numeric, Log2(K) for bits
            input_dim = var['dim']
            self.projections[name] = nn.Linear(input_dim, channels)

    def forward(self, feature_dict):
        embeddings_list = []
        for var in self.schema:
            name = var['name']
            emb = self.projections[name](feature_dict[name])
            embeddings_list.append(emb)
        return torch.stack(embeddings_list, dim=1).permute(0, 2, 1)


class RDDM_NET(nn.Module):
    def __init__(self, config, device, variable_schema):
        super().__init__()
        self.device = device
        self.channels = config["model"]["channels"]
        self.num_steps = config["diffusion"]["num_steps"]
        self.nheads = config["model"]["nheads"]
        self.layers = config["model"]["layers"]

        self.target_schema = [v for v in variable_schema if 'aux' not in v['type']]
        self.aux_schema = [v for v in variable_schema if 'aux' in v['type']]

        self.target_embedder = HybridFeatureEmbedder(self.target_schema, self.channels)
        self.aux_embedder = HybridFeatureEmbedder(self.aux_schema, self.channels)

        # [SELF-CONDITIONING] Input Mixer
        # Inputs: x_t (Signal) + P1 (Proxy) + SelfCond (Predicted x0)
        # 3 Inputs * Channels
        self.input_mixer = Conv1d_with_init(3 * self.channels, self.channels, 1)

        self.diffusion_embedding = DiffusionEmbedding(self.num_steps, self.channels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.channels, self.channels, self.channels, self.nheads) for _ in range(self.layers)
        ])

        self.output_heads = nn.ModuleDict()
        for var in self.target_schema:
            name = var['name']
            out_dim = var['dim']
            # Predict [Residual, Noise] -> 2 * dim
            self.output_heads[name] = nn.Linear(self.channels, 2 * out_dim)

    def forward(self, x_t_dict, t, p1_dict, aux_dict, self_cond_dict=None):
        # 1. Embed Inputs
        x_t_emb = self.target_embedder(x_t_dict)  # (B, C, L)
        p1_emb = self.target_embedder(p1_dict)  # (B, C, L)
        aux_emb = self.aux_embedder(aux_dict)  # (B, C, L_aux)
        dif_emb = self.diffusion_embedding(t)  # (B, C)

        # 2. Embed Self-Conditioning (or Zero)
        if self_cond_dict is None:
            # Create zeros with same shape as x_t_emb
            self_cond_emb = torch.zeros_like(x_t_emb)
        else:
            self_cond_emb = self.target_embedder(self_cond_dict)

        # 3. Concatenate Inputs
        # The order matters: Signal, Proxy, SelfCond
        combined_input = torch.cat([x_t_emb, p1_emb, self_cond_emb], dim=1)  # (B, 3C, L)
        mixed_input = self.input_mixer(combined_input)  # (B, C, L)

        # 4. Concatenate Aux
        sequence = torch.cat([mixed_input, aux_emb], dim=2)

        skip_accum = 0
        for block in self.res_blocks:
            sequence, skip = block(sequence, sequence, dif_emb)
            skip_accum += skip

        target_features = (skip_accum / math.sqrt(self.layers))[:, :, :len(self.target_schema)].permute(0, 2, 1)

        output_dict = {}
        for i, var in enumerate(self.target_schema):
            output_dict[var['name']] = self.output_heads[var['name']](target_features[:, i, :])
        return output_dict