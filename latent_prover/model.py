"""A small goal-conditioned proof autoencoder with a continuous bottleneck."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab: int
    max_length: int
    width: int = 128
    latent_dim: int = 32
    slots: int = 4
    layers: int = 2
    heads: int = 4
    dropout: float = 0.05


class ProofModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        width, heads = config.width, config.heads
        self.embedding = nn.Embedding(config.vocab, width, padding_idx=0)
        self.position = nn.Embedding(config.max_length, width)
        self.proof_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, heads, 2 * width, config.dropout, batch_first=True),
            config.layers, enable_nested_tensor=False,
        )
        self.goal_encoder = nn.TransformerEncoderLayer(
            width, heads, 2 * width, config.dropout, batch_first=True,
        )
        self.queries = nn.Parameter(torch.randn(config.slots, width) / width**0.5)
        self.pool = nn.MultiheadAttention(width, heads, batch_first=True)
        self.to_latent = nn.Linear(width, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, width)
        self.latent_type = nn.Parameter(torch.zeros(width))
        self.strategy_embeddings = nn.Embedding(2, config.slots * config.latent_dim)
        self.latent_shape = (config.slots, config.latent_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(width, heads, 2 * width, config.dropout, batch_first=True),
            config.layers,
        )
        self.output = nn.Linear(width, config.vocab)

    def embed(self, ids):
        positions = torch.arange(ids.shape[1], device=ids.device)
        return self.embedding(ids) + self.position(positions)

    def encode_proof(self, proof):
        pad = proof.eq(0)
        features = self.proof_encoder(self.embed(proof), src_key_padding_mask=pad)
        queries = self.queries.unsqueeze(0).expand(proof.shape[0], -1, -1)
        pooled, _ = self.pool(queries, features, features, key_padding_mask=pad, need_weights=False)
        return self.to_latent(pooled)

    def encode_label(self, batch_size, device):
        labels = torch.arange(batch_size, device=device).remainder(4).div(2, rounding_mode="floor")
        return self.strategy_embeddings(labels).reshape(batch_size, *self.latent_shape)

    def memory(self, goal, latent):
        mask = goal.eq(0)
        context = self.goal_encoder(self.embed(goal), src_key_padding_mask=mask)
        latent_mask = mask.new_zeros(latent.shape[:2])
        return (torch.cat((context, self.from_latent(latent) + self.latent_type), dim=1),
                torch.cat((mask, latent_mask), dim=1))

    def decode(self, target, memory, memory_pad):
        length = target.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=target.device).triu(1)
        states = self.decoder(self.embed(target), memory, tgt_mask=causal,
                              tgt_key_padding_mask=target.eq(0), memory_key_padding_mask=memory_pad)
        return self.output(states)

    def generate(self, goal, latent, bos, eos, forbidden, max_tokens):
        memory, pad = self.memory(goal, latent)
        ids = goal.new_full((goal.shape[0], 1), bos)
        done = torch.zeros(goal.shape[0], dtype=torch.bool, device=goal.device)
        for _ in range(max_tokens):
            logits = self.decode(ids, memory, pad)[:, -1]
            logits[:, forbidden] = -torch.inf
            next_id = torch.where(done, eos, logits.argmax(-1))
            ids = torch.cat((ids, next_id.unsqueeze(1)), dim=1)
            done |= next_id.eq(eos)
            if done.all():
                break
        return ids[:, 1:].tolist()


def strategy_loss(latent):
    flat = latent.flatten(1)
    rows = torch.arange(flat.shape[0], device=flat.device)
    # Four rows per goal: two renderings of each of two strategies.
    positive = F.cosine_similarity(flat, flat[rows ^ 1])
    negative = torch.stack((F.cosine_similarity(flat, flat[rows ^ 2]),
                            F.cosine_similarity(flat, flat[rows ^ 3])), dim=1)
    return (0.3 * (1 - positive) + F.relu(0.4 + negative - positive[:, None]).mean(1)).mean()
