# networks.py: Hybrid D3PM/RDDM Backbone
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
            # D3PM: Inputs are soft probabilities (size K) or scalars (size 1)
            input_dim = var['num_classes'] if 'categorical' in var['type'] else 1
            self.projections[name] = nn.Linear(input_dim, channels)

    def forward(self, feature_dict):
        embeddings_list = []
        for var in self.schema:
            name = var['name']
            emb = self.projections[name](feature_dict[name])
            embeddings_list.append(emb)
        return torch.stack(embeddings_list, dim=1).permute(0, 2, 1)


class CSDI_Hybrid(nn.Module):
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
        self.input_mixer = Conv1d_with_init(2 * self.channels, self.channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(self.num_steps, self.channels)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.channels, self.channels, self.channels, self.nheads) for _ in range(self.layers)
        ])

        self.output_heads = nn.ModuleDict()
        for var in self.target_schema:
            name = var['name']
            if var['type'] == 'categorical':
                self.output_heads[name] = nn.Linear(self.channels, var['num_classes'])
            else:
                self.output_heads[name] = nn.Linear(self.channels, 2)  # [Residual, Noise]

    def forward(self, x_t_dict, t, p1_dict, aux_dict):
        x_t_emb = self.target_embedder(x_t_dict)
        p1_emb = self.target_embedder(p1_dict)
        aux_emb = self.aux_embedder(aux_dict)
        dif_emb = self.diffusion_embedding(t)

        combined_input = torch.cat([x_t_emb, p1_emb], dim=1)
        mixed_input = self.input_mixer(combined_input)
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