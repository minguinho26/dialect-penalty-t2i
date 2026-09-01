"""latentguard_core.py — LatentGuard shared core for text filter.

Module extracting byte-identical code (WrapClip / EmbeddingMappingLayer / contra_score /
load_latentguard / score_prompt) shared by exp_text_level_latentguard.py and
exp_text_level_latentguard_sae_and_typo.py. No side effects on import.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPTokenizer
from tqdm import tqdm

class WrapClip:
    def __init__(self, device, model_name='openai/clip-vit-large-patch14'):
        self.device = device
        self.clip_model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(model_name)
        # CLIP BPE special-token IDs are stable across versions
        self.bos_id = self.clip_tokenizer.bos_token_id   # 49406
        self.eos_id = self.clip_tokenizer.eos_token_id   # 49407
        self.pad_id = self.clip_tokenizer.pad_token_id or 0

    @torch.no_grad()
    def _encode_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (1, 77) — encode and return (1, 78, 768)."""
        outputs = self.clip_model.text_model(input_ids=token_ids.to(self.device))
        z = outputs.last_hidden_state                         # (1, 77, 768)
        eos_pos = int(token_ids.argmax(dim=-1))               # CLIP convention
        pooled = z[:, eos_pos, :].unsqueeze(1)                # (1, 1, 768)
        res = torch.cat([pooled, z], dim=1)                   # (1, 78, 768)
        assert res.shape == (1, 78, 768)
        return res

    @torch.no_grad()
    def get_emb(self, targetp: str) -> torch.Tensor:
        """Single-pass (truncated to 77 tokens). Backwards-compatible."""
        batch = self.clip_tokenizer(
            [targetp], truncation=True, max_length=77,
            padding="max_length", return_tensors="pt"
        )
        return self._encode_token_ids(batch["input_ids"])

    @torch.no_grad()
    def get_emb_chunked(self, targetp: str,
                        max_inner: int = 75, stride: int = 60) -> list:
        """Sliding-window encoding. Returns a list of (1, 78, 768) tensors,
        one per chunk. Each chunk = [BOS] + ≤max_inner content tokens + [EOS],
        padded to 77."""
        # 1. Raw content tokens (no special tokens, no truncation)
        raw = self.clip_tokenizer(targetp, add_special_tokens=False,
                                   return_tensors="pt", truncation = False)["input_ids"][0]

        embs = []
        if len(raw) == 0:
            empty = torch.tensor([[self.bos_id, self.eos_id] + [self.pad_id]*75])
            return [self._encode_token_ids(empty)]

        # 2. Sliding chunks
        for i in range(0, len(raw), stride):
            content = raw[i : i + max_inner]
            chunk = torch.cat([
                torch.tensor([self.bos_id]),
                content,
                torch.tensor([self.eos_id]),
            ])
            pad_n = 77 - len(chunk)
            chunk = torch.cat([chunk, torch.full((pad_n,), self.pad_id,
                                                 dtype=torch.long)]).unsqueeze(0)
            embs.append(self._encode_token_ids(chunk))
            if i + max_inner >= len(raw):
                break
        return embs

class EmbeddingMappingLayer(nn.Module):
    def __init__(self, num_heads, head_dim, out_dim=512):
        super(EmbeddingMappingLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.key_d = self.head_dim * self.num_heads
        self.out_dim = out_dim

        # Ensure the head dimension is an integer
        assert self.key_d % self.num_heads == 0, "key_d must be divisible by num_heads"

        self.x1_to_key = nn.Linear(768, self.key_d)
        self.x2_to_query = nn.Linear(768, self.key_d)
        self.x1_to_value = nn.Linear(768, self.key_d)
        
        self.final_mlp = nn.Linear(self.key_d, self.out_dim)  # Optional: final layer to combine head outputs
        self.mlp_query1 = nn.Linear(self.key_d, self.out_dim)
        self.tempr = nn.Parameter(torch.tensor(1/0.07), requires_grad=True) #1.0 / 0.07 #

    def forward(self, x1, x2):
        batch_size, seq_len, _ = x1.shape

        # Process x1 to generate keys and values
        # key shape: (batch_size, seq_len, num_heads, head_dim)
        key = self.x1_to_key(x1).view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.transpose(1, 2)  # Reshape to (batch_size, num_heads, seq_len, head_dim)

        # value shape: (batch_size, seq_len, num_heads, head_dim)
        value = self.x1_to_value(x1).view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.transpose(1, 2)  # Reshape to (batch_size, num_heads, seq_len, head_dim)

        # Process x2 to generate queries
        # query shape: (batch_size, 1, num_heads, head_dim)
        query = self.x2_to_query(x2).view(batch_size, 1, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)  # Reshape to (batch_size, num_heads, 1, head_dim)

        # Compute attention scores for each head
        # attention_scores shape: (batch_size, num_heads, 1, seq_len)
        scaling_factor = self.head_dim ** 0.5
        attention_scores = torch.einsum('bnqd,bnkd->bnqk', query, key) / scaling_factor
        attention_weights = F.softmax(attention_scores, dim=-1)

        V = torch.einsum('bnqk,bnkd->bnqd', attention_weights, value)
        V = V.view(batch_size, -1)
        V = self.final_mlp(V)
        V_prime = V  # V' after processing

        query = query.view(batch_size, -1)

        query = self.mlp_query1(query)
        query_prime = query  # query' after processing

        return V_prime, query_prime

def contra_score(model, v_prime, q_prime):
    """Temperature-scaled cosine similarity."""
    v = F.normalize(v_prime, p=2, dim=1)
    q = F.normalize(q_prime, p=2, dim=1)
    return (v * q).sum(dim=1) * model.tempr

def load_latentguard(
    path_model:    str = "./latent_guard_checkpoint/model_parameters.pth",
    path_dataset:  str = "./latent_guard_checkpoint/CoPro_v1.0.json",
    cache_dir:     str = "./latent_guard_checkpoint",
    device:        str = "cuda:0" if torch.cuda.is_available() else "cpu",
    clip_name:     str = "openai/clip-vit-large-patch14",
    use_ood:       bool = True,
    verbose:       bool = True,
):
    
    log = print if verbose else (lambda *a, **k: None)

    log("Loading CLIP-vit-large ...")
    wrap_clip = WrapClip(device, model_name=clip_name)

    log("Loading LatentGuard scoring head ...")
    model = EmbeddingMappingLayer(num_heads=16, head_dim=32, out_dim=128).to(device)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.eval()

    log(f"Loading CoPro concept lists from {path_dataset} ...")
    ds = json.load(open(path_dataset))
    concepts = list(ds["ID_concepts"])
    if use_ood:
        concepts += list(ds["OOD_concepts"])
    log(f"  total concepts: {len(concepts)}")

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "concept_cache.pt")

    if os.path.exists(cache_path):
        log(f"Loading existing concept cache from {cache_path} ...")
        clip_cache = torch.load(cache_path)
        log(f"  loaded cache: {len(clip_cache)} entries")
    else:
        log(f"No cache found at {cache_path}; encoding concepts ...")
        clip_cache = {}
        for c in tqdm(concepts, desc="CLIP encode", disable=not verbose):
            clip_cache[c] = wrap_clip.get_emb(c).cpu()
        log(f"Saving concept cache to {cache_path} ...")
        torch.save(clip_cache, cache_path)

    log("Building concept bank ...")
    embs = []
    for c in concepts:
        if c in clip_cache:
            embs.append(clip_cache[c].to(device))
        else:
            emb = wrap_clip.get_emb(c).to(device)
            clip_cache[c] = emb.cpu()
            embs.append(emb)
    concept_bank = torch.cat(embs, dim=0)[:, 0, :]   # (N, 768)
    log(f"  bank shape: {tuple(concept_bank.shape)}")

    return wrap_clip, model, clip_cache, concept_bank

@torch.no_grad()
def score_prompt(prompt, wrap_clip, model, clip_cache, concept_bank,
                 chunk_threshold: int = 75):
    """LatentGuard score. HIGHER → more unsafe (opposite sign convention to GuardT2I).

    Automatically chooses single-pass vs sliding-window encoding based on
    the prompt's tokenized length:
        - ≤ chunk_threshold tokens : single pass, cache enabled
        - >  chunk_threshold tokens: sliding window via get_emb_chunked(),
                                     cache bypassed (chunk boundaries depend
                                     on stride and don't compose with
                                     prompt-level cache keys)

    For each chunk:
        1. encode → (1, 78, 768) embedding
        2. for every concept c in the bank, compute contra(emb, c) → (N,)
        3. take max over c — closeness to the *most matching* blacklisted concept
    Across chunks, mean-pool to keep the score length-invariant.
    """
    import numpy as np
    device = concept_bank.device
    N = concept_bank.shape[0]

    # 1. Decide single-pass vs chunking by content-token count
    n_tokens = len(wrap_clip.clip_tokenizer(
        prompt, add_special_tokens=False, truncation = False)["input_ids"])

    if n_tokens <= chunk_threshold:
        # Short prompt: single pass + cache
        if prompt in clip_cache:
            prompt_embs = [clip_cache[prompt].to(device)]
        else:
            emb = wrap_clip.get_emb(prompt).to(device)
            clip_cache[prompt] = emb.cpu()
            prompt_embs = [emb]
    else:
        # Long prompt: sliding-window, cache bypassed
        prompt_embs = [e.to(device) for e in wrap_clip.get_emb_chunked(prompt)]

    # 2. Score each chunk: max over blacklisted concepts
    per_chunk_max = []
    for emb in prompt_embs:
        repeated = emb.repeat(N, 1, 1)                     # (N, 78, 768)
        v_prime, q_prime = model(repeated, concept_bank)
        scores = contra_score(model, v_prime, q_prime)     # (N,)
        per_chunk_max.append(scores.max().item())

    # 3. Mean-pool across chunks → length-invariant prompt-level score
    return float(np.mean(per_chunk_max))
