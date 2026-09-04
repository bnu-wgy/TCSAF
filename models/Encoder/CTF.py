import math
import torch
import numpy as np
import torch.nn as nn

args = {'n': 4,
        'L': 4,
        'D': 768,
        'mlp_dim': 1024,
        'input_size': 1024,
        'dropout_rate': 0.}

class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(args['D'], args['mlp_dim'])
        self.fc2 = nn.Linear(args['mlp_dim'], args['D'])
        self.act_fn = torch.nn.functional.gelu  # torch.nn.functional.relu
        self.dropout = nn.Dropout(args['dropout_rate'])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(args['D'], eps=1e-6)
        self.ffn_norm = nn.LayerNorm(args['D'], eps=1e-6)
        self.ffn = Mlp()
        self.norm = nn.LayerNorm(args['D'], eps=1e-6)
        self.attn = Attention()

    def forward(self, x1, x2, topo_bias=None):
        identity = x1
        x1 = self.attention_norm(x1)
        x2 = self.norm(x2)
        x1 = self.attn(x1, x2, topo_bias=topo_bias)
        x1 = x1 + identity

        identity = x1
        x1 = self.ffn_norm(x1)
        x1 = self.ffn(x1)
        x1 = x1 + identity

        return x1


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(args['D'], eps=1e-6)
        for _ in range(args['L']):
            layer = Block()
            self.layer.append(layer)

    def forward(self, x1, x2, topo_bias=None):
        for layer_block in self.layer:
            x1 = layer_block(x1, x2, topo_bias=topo_bias)
        encoded = self.encoder_norm(x1)
        return encoded


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.num_attention_heads = args['n']
        self.attention_head_size = int(args['D'] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args['D'], self.all_head_size)
        self.key = nn.Linear(args['D'], self.all_head_size)
        self.value = nn.Linear(args['D'], self.all_head_size)

        self.out = nn.Linear(args['D'], args['D'])
        self.attn_dropout = nn.Dropout(args['dropout_rate'])
        self.proj_dropout = nn.Dropout(args['dropout_rate'])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x1, x2, topo_bias=None):
        mixed_query_layer = self.query(x1)
        mixed_key_layer = self.key(x2)
        mixed_value_layer = self.value(x2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if topo_bias is not None:
            attention_scores = attention_scores + topo_bias

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


def _build_normalized_grid(size):
    coords = torch.linspace(0.0, 1.0, steps=size)
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1)
    grid = grid.view(size * size, 2)
    return grid


class TopologyBias(nn.Module):
    def __init__(self, hidden_size, num_classes, num_heads, feat_size1, feat_size2):
        super().__init__()
        self.num_heads = num_heads
        self.num_classes = num_classes
        # Eq. (3): both streams share the same semantic projection W_p.
        self.cls_proj = nn.Linear(hidden_size, num_classes)
        in_dim = num_classes + 2
        mid_dim = max(32, hidden_size // 8)
        # Eq. (4): the spatial-semantic embedding MLP is shared as well.
        self.embedding_mlp = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, num_heads)
        )
        grid1 = _build_normalized_grid(feat_size1)
        grid2 = _build_normalized_grid(feat_size2)
        self.register_buffer('grid1', grid1)
        self.register_buffer('grid2', grid2)

    def forward(self, embed_q, embed_k):
        B, nq, _ = embed_q.shape
        _, nk, _ = embed_k.shape

        if nq != self.grid1.shape[0] or nk != self.grid2.shape[0]:
            raise ValueError(
                'Token counts must match the configured feature grids: '
                f'got ({nq}, {nk}), expected '
                f'({self.grid1.shape[0]}, {self.grid2.shape[0]}).'
            )

        prob_q = torch.softmax(self.cls_proj(embed_q), dim=-1)
        prob_k = torch.softmax(self.cls_proj(embed_k), dim=-1)

        pos_q = self.grid1.unsqueeze(0).expand(B, nq, -1)
        pos_k = self.grid2.unsqueeze(0).expand(B, nk, -1)

        q_feat = torch.cat([pos_q, prob_q], dim=-1)
        k_feat = torch.cat([pos_k, prob_k], dim=-1)

        embed_q_topo = self.embedding_mlp(q_feat)
        embed_k_topo = self.embedding_mlp(k_feat)

        # [B, N_q, H] x [B, N_k, H] -> [B, H, N_q, N_k].
        # No additional scale is used, matching Eq. (4)-(5) in the paper.
        bias = torch.einsum('bqh,bkh->bhqk', embed_q_topo, embed_k_topo)
        return torch.tanh(bias)


class Embeddings(nn.Module):
    def __init__(self, in_channels=3, hidden_size=args['D'], img_size=args['input_size']):
        super(Embeddings, self).__init__()
        n_patches = img_size * img_size
        k_size = 1
        self.patch_embeddings = nn.Conv2d(in_channels, out_channels=hidden_size, kernel_size=k_size, stride=k_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(args['dropout_rate'])

    def forward(self, x):
        x = self.patch_embeddings(x) 
        x = x.flatten(2).transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings



class CTF(nn.Module):
    def __init__(self, in_channels1, in_channels2, feat_size1, feat_size2,
                 hidden_size=args['D'], num_classes=6, use_topo_bias=True):
        super(CTF, self).__init__()
        self.embed1 = Embeddings(in_channels1, hidden_size, feat_size1)
        self.embed2 = Embeddings(in_channels2, hidden_size, feat_size2)
        self.encoder = Encoder()
        self.use_topo_bias = use_topo_bias
        if use_topo_bias:
            self.topo_bias = TopologyBias(
                hidden_size=hidden_size,
                num_classes=num_classes,
                num_heads=args['n'],
                feat_size1=feat_size1,
                feat_size2=feat_size2
            )
        else:
            self.topo_bias = None

    def forward(self, x_l, x_g):
        embed1 = self.embed1(x_l)
        embed2 = self.embed2(x_g)
        topo_bias = None
        if self.topo_bias is not None:
            topo_bias = self.topo_bias(embed1, embed2)
        encoded = self.encoder(embed1, embed2, topo_bias=topo_bias)
        B, n_patch, hidden = encoded.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        encoded = encoded.permute(0, 2, 1)
        encoded = encoded.contiguous().view(B, hidden, h, w)
        return encoded
