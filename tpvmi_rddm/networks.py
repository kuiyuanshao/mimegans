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
    def __init__(self, num_steps, embedding_dim=128, dropout_rate = 0):
        super().__init__()
        self.register_buffer("embedding", self._build_embedding(num_steps, embedding_dim // 2), persistent=False)
        self.projection1 = nn.Linear(embedding_dim, embedding_dim)
        self.projection2 = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        return self.projection2(self.dropout(F.silu(self.projection1(self.dropout(x)))))

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, dropout_rate = 0):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, side_info, diffusion_emb):
        B, C, L = x.shape
        x_in = x
        time_emb = self.diffusion_projection(self.dropout(diffusion_emb)).unsqueeze(-1)
        y = x + time_emb
        y = y + self.cond_projection(self.dropout(side_info))[:, :C, :]
        y = self.feature_layer(self.dropout(y.permute(0, 2, 1))).permute(0, 2, 1)
        y = self.mid_projection(self.dropout(y))
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(self.dropout(y))
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x_in + residual) / math.sqrt(2.0), skip


class HybridFeatureEmbedder(nn.Module):
    def __init__(self, schema, channels, dropout_rate = 0):
        super().__init__()
        self.schema = schema
        self.projections = nn.ModuleDict()
        self.dropout = nn.Dropout(dropout_rate)
        for var in schema:
            name = var['name']
            input_dim = var['num_classes'] if 'categorical' in var['type'] else 1
            self.projections[name] = nn.Linear(input_dim, channels)

    def forward(self, feature_dict):
        embeddings_list = []
        for var in self.schema:
            name = var['name']
            emb = self.projections[name](self.dropout(feature_dict[name]))
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
        self.dropout_rate = config["model"]["dropout"]

        self.target_schema = [v for v in variable_schema if 'aux' not in v['type']]
        self.aux_schema = [v for v in variable_schema if 'aux' in v['type']]

        self.target_embedder = HybridFeatureEmbedder(self.target_schema, self.channels, dropout_rate = self.dropout_rate)
        self.aux_embedder = HybridFeatureEmbedder(self.aux_schema, self.channels, dropout_rate = self.dropout_rate)

        # Learnable Mask Token for absorbing state
        self.mask_token = nn.Parameter(torch.randn(1, self.channels, 1))

        # Input Mixer: Takes x_t (Masked) + P1 (Condition)
        self.input_mixer = Conv1d_with_init(2 * self.channels, self.channels, 1)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.diffusion_embedding = DiffusionEmbedding(self.num_steps, self.channels, self.dropout_rate)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(self.channels, self.channels, self.channels, self.nheads, self.dropout_rate) for _ in range(self.layers)
        ])

        self.output_heads = nn.ModuleDict()

        for var in self.target_schema:
            name = var['name']
            if var['type'] == 'categorical':
                self.output_heads[name] = nn.Linear(self.channels, var['num_classes'])
            else:
                # Numeric: Decoupled heads for Residual and Noise
                self.output_heads[name] = nn.ModuleDict({
                    'res': nn.Sequential(
                        nn.Linear(self.channels, self.channels),
                        nn.SiLU(),
                        nn.Dropout(self.dropout_rate),
                        nn.Linear(self.channels, 1)
                    ),
                    'eps': nn.Sequential(
                        nn.Linear(self.channels, self.channels),
                        nn.SiLU(),
                        nn.Dropout(self.dropout_rate),
                        nn.Linear(self.channels, 1)
                    )
                })

    def forward(self, x_t_dict, t, p1_dict, aux_dict, mask_dict):
        # 1. Embed Inputs
        x_t_emb = self.target_embedder(x_t_dict)
        p1_emb = self.target_embedder(p1_dict)
        aux_emb = self.aux_embedder(aux_dict)
        dif_emb = self.diffusion_embedding(t)

        # 2. Apply Input Masking
        mask_list = []
        for var in self.target_schema:
            m = mask_dict[var['name']]
            m_expanded = m.unsqueeze(1).expand(-1, self.channels, -1)
            mask_list.append(m_expanded)

        mask_seq = torch.cat(mask_list, dim=2)
        x_t_masked = x_t_emb * mask_seq + self.mask_token * (1.0 - mask_seq)

        # 3. Combine: Masked State + Proxy Condition
        combined_input = torch.cat([x_t_masked, p1_emb], dim=1)
        mixed_input = self.input_mixer(self.dropout(combined_input))

        sequence = torch.cat([mixed_input, aux_emb], dim=2)

        skip_accum = 0
        for block in self.res_blocks:
            sequence, skip = block(sequence, sequence, dif_emb)
            skip_accum += skip

        target_features = (skip_accum / math.sqrt(self.layers))[:, :, :len(self.target_schema)].permute(0, 2, 1)

        output_dict = {}
        for i, var in enumerate(self.target_schema):
            name = var['name']
            feature = target_features[:, i, :]

            if var['type'] == 'categorical':
                output_dict[name] = self.output_heads[name](self.dropout(feature))
            else:
                # [ARCH FIX] Run decoupled heads
                res_pred = self.output_heads[name]['res'](self.dropout(feature))
                eps_pred = self.output_heads[name]['eps'](self.dropout(feature))

                # Stack results to (B, 2) to match expected shape in tpvmi_rddm.py
                # Index 0: Residual, Index 1: Noise
                output_dict[name] = torch.cat([res_pred, eps_pred], dim=1)

        return output_dict