"""
AdverSentinel-LLM: Contrastive Dual-Encoder Transformer (CDET)
for Adversarial Prompt Detection and Zero-Day Jailbreak Defense.

Architecture:
  - Prompt Semantic Encoder (PSE): BERT-base-uncased
  - Threat Pattern Encoder (TPE): BERT-base-uncased + Threat Feature Extraction (TFE)
  - Projection Head: 2-layer MLP (1536 -> 256)
  - Dynamic Threat Memory Bank (DTMB): 64 prototypes, 8 categories, EMA update
  - Anomaly-Aware Gating Mechanism (AAGM): context-sensitive adaptive fusion

Reference: AdverSentinel-LLM paper, CMC 2025.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Threat Feature Extraction (TFE) Module
# ---------------------------------------------------------------------------

class ThreatFeatureExtractor(nn.Module):
    """
    Computes lexical, structural, and statistical threat-indicative features
    from raw input text strings (before transformer encoding).

    Output dimension: 64 (lexical=24, structural=24, statistical=16)
    """

    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        # Learned projection from hand-crafted features to embedding space
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 768),  # maps to hidden_dim for concatenation
        )

    def compute_raw_features(self, input_ids: torch.Tensor,
                              attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable proxy features from token ids.

        Lexical (24 dims):
          - Token entropy (normalized)
          - Fraction of non-ASCII-range tokens (proxy for encoding obfuscation)
          - Unique token ratio
          - High-frequency adversarial token indicators (role-play markers, etc.)

        Structural (24 dims):
          - Prompt length (normalized to 512)
          - Sentence boundary count (proxy: period/exclamation tokens)
          - Instruction hierarchy markers (numbered list proxies)
          - Role-play keyword presence (soft indicator via token id ranges)

        Statistical (16 dims):
          - Unigram frequency deviation (proxy: std of token id distribution)
          - Repetition ratio
          - Average token id (proxy for vocabulary region)
          - N-gram novelty score (proxy: fraction of consecutive unique bigrams)
        """
        B, T = input_ids.shape
        mask = attention_mask.float()
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B,1)

        # --- Lexical features (24) ---
        # Token value distribution statistics
        ids_f = input_ids.float()
        mean_id = (ids_f * mask).sum(1) / lengths.squeeze(1)                  # (B,)
        std_id  = ((ids_f - mean_id.unsqueeze(1)) ** 2 * mask).sum(1) / lengths.squeeze(1)
        std_id  = std_id.sqrt()

        # Unique token ratio
        # Use approximate: std / (max_id) as proxy for diversity
        max_id = ids_f.max(dim=1).values.clamp(min=1)
        unique_ratio = (std_id / max_id).unsqueeze(1)                         # (B,1)

        # Fraction of tokens in high-id range (>20000) — proxy for rare/obfuscated tokens
        high_id_frac = ((ids_f > 20000).float() * mask).sum(1, keepdim=True) / lengths  # (B,1)

        # Fraction of tokens in [1000,3000] range — common adversarial template tokens
        adv_range = ((ids_f >= 1000) & (ids_f <= 3000)).float()
        adv_range_frac = (adv_range * mask).sum(1, keepdim=True) / lengths    # (B,1)

        # Token id normalized mean and std (both scaled to [0,1])
        mean_norm = (mean_id / 30522).unsqueeze(1)                            # (B,1)
        std_norm  = (std_id  / 30522).unsqueeze(1)                            # (B,1)

        # Pad lexical block to 24 dims
        lex_block = torch.cat([
            mean_norm, std_norm, unique_ratio, high_id_frac, adv_range_frac,
            mean_norm * std_norm,          # interaction
            (mean_norm ** 2),
            (std_norm  ** 2),
        ] + [torch.zeros(B, 1, device=input_ids.device)] * 16, dim=1)[:, :24]  # (B,24)

        # --- Structural features (24) ---
        # Prompt length (normalized)
        len_norm = (lengths / 512).squeeze(1).unsqueeze(1)                    # (B,1)

        # Sentence-boundary proxy: tokens 1012 (,), 1008 (?), 999 (.), 999 (!)
        boundary_ids = torch.tensor([1012, 1008, 999, 1029], device=input_ids.device)
        boundary_mask = (input_ids.unsqueeze(-1) == boundary_ids).any(-1).float()
        boundary_frac = (boundary_mask * mask).sum(1, keepdim=True) / lengths # (B,1)

        # Very short prompt (<=20 tokens) — some jailbreaks are terse
        short_flag = (lengths < 20).float()                                   # (B,1)

        # Very long prompt (>300 tokens) — multi-turn crescendo proxy
        long_flag  = (lengths > 300).float()                                  # (B,1)

        # Fraction of sequence that is padding (inverse density)
        pad_frac = 1.0 - (lengths / T)                                        # (B,1)

        struct_block = torch.cat([
            len_norm, boundary_frac, short_flag, long_flag, pad_frac,
            len_norm * boundary_frac,
            len_norm ** 2,
            boundary_frac ** 2,
        ] + [torch.zeros(B, 1, device=input_ids.device)] * 16, dim=1)[:, :24]  # (B,24)

        # --- Statistical features (16) ---
        # Bigram repetition ratio
        ids_shifted = torch.roll(input_ids, 1, dims=1)
        ids_shifted[:, 0] = 0
        bigram_repeat = ((input_ids == ids_shifted).float() * mask).sum(1, keepdim=True) / lengths.clamp(min=2)

        # Token id range spread (max - min, normalized)
        id_range = (max_id - ids_f.min(dim=1).values).unsqueeze(1) / 30522    # (B,1)

        # Median proxy (use 25th percentile via sorted approximation — use mean as proxy)
        # Difference between mean and median proxy
        median_proxy = mean_norm  # simplified
        skew_proxy = (mean_norm - median_proxy).abs()                          # (B,1)

        stat_block = torch.cat([
            bigram_repeat, id_range, skew_proxy, mean_norm * id_range,
            std_norm * bigram_repeat,
        ] + [torch.zeros(B, 1, device=input_ids.device)] * 11, dim=1)[:, :16]  # (B,16)

        # Concatenate all blocks -> (B, 64)
        features = torch.cat([lex_block, struct_block, stat_block], dim=1)
        return features  # (B, 64)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns projected threat features: (B, 768)"""
        raw = self.compute_raw_features(input_ids, attention_mask)
        return self.proj(raw)  # (B, 768)


# ---------------------------------------------------------------------------
# Prompt Semantic Encoder (PSE)
# ---------------------------------------------------------------------------

class PromptSemanticEncoder(nn.Module):
    """
    BERT-base-uncased encoder for deep contextual prompt representations.
    Output: mean-pooled CLS+token representation, shape (B, 768).
    """

    def __init__(self, pretrained: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pool over non-padding tokens (Eq. 5 in paper)
        hidden = outputs.last_hidden_state                  # (B, T, 768)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        h_pse = (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)
        return h_pse  # (B, 768)


# ---------------------------------------------------------------------------
# Threat Pattern Encoder (TPE)
# ---------------------------------------------------------------------------

class ThreatPatternEncoder(nn.Module):
    """
    BERT-base-uncased encoder augmented with explicit threat features.
    The TFE features are projected to 768d and prepended to the token
    embedding sequence as an additional virtual token (Eq. 7).
    Output: attention-pooled representation, shape (B, 768).
    """

    def __init__(self, pretrained: str = "bert-base-uncased",
                 feature_dim: int = 64):
        super().__init__()
        self.tfe = ThreatFeatureExtractor(feature_dim)
        self.bert = BertModel.from_pretrained(pretrained)
        # Learnable query for attention pooling (Eq. 8)
        self.attn_query = nn.Parameter(torch.randn(1, 1, 768))

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape

        # 1. Extract threat features and project to 768d
        z_tfe = self.tfe(input_ids, attention_mask)  # (B, 768)

        # 2. Get BERT token embeddings
        embeddings = self.bert.embeddings(input_ids=input_ids)  # (B, T, 768)

        # 3. Prepend TFE as virtual token (Eq. 7): concat [z_tfe; embeddings]
        z_tfe_token = z_tfe.unsqueeze(1)                        # (B, 1, 768)
        aug_embeddings = torch.cat([z_tfe_token, embeddings], dim=1)  # (B, T+1, 768)

        # 4. Extend attention mask for the prepended token
        extra_mask = torch.ones(B, 1, device=attention_mask.device, dtype=attention_mask.dtype)
        aug_mask = torch.cat([extra_mask, attention_mask], dim=1)  # (B, T+1)

        # 5. Run through BERT encoder layers (skip embedding layer; feed directly)
        encoder_outputs = self.bert.encoder(
            aug_embeddings,
            attention_mask=self.bert.get_extended_attention_mask(
                aug_mask, aug_embeddings.shape[:2]
            ),
        )
        hidden = encoder_outputs.last_hidden_state  # (B, T+1, 768)

        # 6. Attention pooling with learnable query (Eq. 8)
        query = self.attn_query.expand(B, -1, -1)               # (B, 1, 768)
        scores = torch.bmm(query, hidden.transpose(1, 2)) / math.sqrt(768)  # (B, 1, T+1)
        # Mask padding positions
        aug_mask_bool = aug_mask.unsqueeze(1).bool()
        scores = scores.masked_fill(~aug_mask_bool, float('-inf'))
        weights = torch.softmax(scores, dim=-1)                  # (B, 1, T+1)
        h_tpe = torch.bmm(weights, hidden).squeeze(1)            # (B, 768)
        return h_tpe


# ---------------------------------------------------------------------------
# Projection Head
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """
    2-layer MLP: 1536 -> 768 -> 256, with GELU and LayerNorm (Eq. 9).
    """

    def __init__(self, in_dim: int = 1536, hidden_dim: int = 768,
                 out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h_pse: torch.Tensor, h_tpe: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h_pse, h_tpe], dim=-1)  # (B, 1536)
        return self.net(combined)                       # (B, 256)


# ---------------------------------------------------------------------------
# Dynamic Threat Memory Bank (DTMB)
# ---------------------------------------------------------------------------

class DynamicThreatMemoryBank(nn.Module):
    """
    Prototype-based memory bank with:
      - EMA prototype updates (Eq. 12)
      - Staleness-based rejuvenation (stale threshold = 500 steps)
      - Cosine-similarity-based pruning (threshold = 0.98)

    Shape: M prototypes of dimension proj_dim (256).
    """

    def __init__(self, num_prototypes: int = 64, num_categories: int = 8,
                 proj_dim: int = 256, momentum: float = 0.999,
                 stale_threshold: int = 500, cos_prune_threshold: float = 0.98,
                 grad_update_eps: float = 0.01, bandwidth: float = 0.5):
        super().__init__()
        self.M = num_prototypes
        self.C = num_categories
        self.proj_dim = proj_dim
        self.beta = momentum
        self.stale_threshold = stale_threshold
        self.cos_prune_threshold = cos_prune_threshold
        self.grad_eps = grad_update_eps

        # Prototype embeddings (not trainable via gradient — updated via EMA)
        self.register_buffer('prototypes', F.normalize(
            torch.randn(num_prototypes, proj_dim), dim=-1))

        # Learnable attention weights and bandwidths (Eq. 11)
        self.log_alpha = nn.Parameter(torch.zeros(num_prototypes))
        self.log_sigma = nn.Parameter(torch.full((num_prototypes,),
                                                  math.log(bandwidth)))

        # Activation counters for staleness tracking (non-persistent)
        self.register_buffer('activation_counter',
                              torch.zeros(num_prototypes, dtype=torch.long))
        self.register_buffer('step_counter', torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update_prototypes(self, embeddings: torch.Tensor,
                          labels: torch.Tensor) -> None:
        """
        EMA update of prototypes from current batch (Eq. 12).
        Then apply pruning and rejuvenation.
        """
        self.step_counter += 1
        protos = self.prototypes  # (M, D)

        # Assign each sample to nearest prototype
        sim = F.cosine_similarity(
            embeddings.unsqueeze(1),   # (B, 1, D)
            protos.unsqueeze(0),       # (1, M, D)
            dim=-1,
        )  # (B, M)
        assignments = sim.argmax(dim=1)  # (B,)

        # EMA update for assigned prototypes
        updated = torch.zeros_like(protos)
        count   = torch.zeros(self.M, device=protos.device)
        for k in range(self.M):
            mask = (assignments == k)
            if mask.any():
                mean_emb = embeddings[mask].mean(0)
                updated[k] = self.beta * protos[k] + (1 - self.beta) * mean_emb
                count[k] = mask.float().sum()

        # Only update prototypes that received samples
        has_samples = (count > 0)
        updated_norm = F.normalize(updated, dim=-1)
        self.prototypes[has_samples] = updated_norm[has_samples]

        # Update activation counters
        self.activation_counter[has_samples] = 0
        self.activation_counter[~has_samples] += 1

        # --- Rejuvenation: stale prototypes ---
        stale_mask = self.activation_counter > self.stale_threshold
        if stale_mask.any() and embeddings.shape[0] > 0:
            stale_indices = stale_mask.nonzero(as_tuple=True)[0]
            for idx in stale_indices:
                # Reinitialize from mean of adversarial samples in current batch
                adv_mask = (labels == 1)
                if adv_mask.any():
                    new_proto = embeddings[adv_mask].mean(0)
                else:
                    new_proto = embeddings[torch.randint(len(embeddings), (1,))].squeeze(0)
                self.prototypes[idx] = F.normalize(new_proto, dim=-1)
                self.activation_counter[idx] = 0

        # --- Pruning: collapsed prototypes ---
        norms_p = F.normalize(self.prototypes, dim=-1)
        cos_matrix = torch.mm(norms_p, norms_p.t())  # (M, M)
        cos_matrix.fill_diagonal_(-1.0)
        collapse_pairs = (cos_matrix >= self.cos_prune_threshold).nonzero(as_tuple=False)
        pruned = set()
        for pair in collapse_pairs:
            i, j = pair[0].item(), pair[1].item()
            if i in pruned or j in pruned:
                continue
            # Prune the one with lower activation count (higher counter = less active)
            victim = i if self.activation_counter[i] > self.activation_counter[j] else j
            pruned.add(victim)
            # Reinitialize from random adversarial sample
            adv_mask = (labels == 1)
            if adv_mask.any():
                rand_idx = torch.randint(adv_mask.sum(), (1,)).item()
                adv_samples = embeddings[adv_mask]
                new_proto = adv_samples[rand_idx]
            else:
                new_proto = embeddings[torch.randint(len(embeddings), (1,))].squeeze(0)
            self.prototypes[victim] = F.normalize(new_proto, dim=-1)
            self.activation_counter[victim] = 0

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute threat proximity score via attention-weighted prototype distance (Eq. 11).
        v: (B, proj_dim)
        Returns: s_dtmb (B,)
        """
        alpha  = torch.softmax(self.log_alpha, dim=0)           # (M,)
        sigma2 = self.log_sigma.exp().pow(2)                     # (M,)

        # Squared L2 distance: (B, M)
        diff = v.unsqueeze(1) - self.prototypes.unsqueeze(0)    # (B, M, D)
        dist2 = (diff ** 2).sum(-1)                              # (B, M)

        # Gaussian kernel weighted by alpha (Eq. 11)
        scores = alpha * torch.exp(-dist2 / (sigma2 + 1e-8))    # (B, M)
        s_dtmb = scores.sum(dim=1)                               # (B,)
        return s_dtmb


# ---------------------------------------------------------------------------
# Anomaly-Aware Gating Mechanism (AAGM)
# ---------------------------------------------------------------------------

class AnomalyAwareGating(nn.Module):
    """
    Adaptive gating between DTMB score and classifier score (Eq. 13, 14).
    Context features: [prompt_length_norm, turn_position, domain_indicator]
    """

    def __init__(self, proj_dim: int = 256, ctx_dim: int = 3):
        super().__init__()
        gate_in = proj_dim + 1 + ctx_dim  # v + s_dtmb + ctx
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.classifier_head = nn.Linear(proj_dim, 1)

    def forward(self, v: torch.Tensor, s_dtmb: torch.Tensor,
                ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        v:      (B, proj_dim)
        s_dtmb: (B,)
        ctx:    (B, ctx_dim) — optional context features
        Returns: y_hat (B,) in [0,1]
        """
        B = v.size(0)
        if ctx is None:
            ctx = torch.zeros(B, 3, device=v.device)

        gate_input = torch.cat([v, s_dtmb.unsqueeze(1), ctx], dim=-1)  # (B, proj+4)
        gamma = torch.sigmoid(self.gate(gate_input)).squeeze(1)         # (B,)

        s_cls = torch.sigmoid(self.classifier_head(v)).squeeze(1)       # (B,)
        y_hat = gamma * s_dtmb + (1 - gamma) * s_cls                    # (B,)
        return y_hat


# ---------------------------------------------------------------------------
# Full AdverSentinel-LLM Model
# ---------------------------------------------------------------------------

class AdverSentinelLLM(nn.Module):
    """
    Complete AdverSentinel-LLM model combining:
      PSE + TPE -> ProjectionHead -> DTMB -> AAGM -> detection score.
    """

    def __init__(self,
                 pretrained: str = "bert-base-uncased",
                 proj_dim: int = 256,
                 num_prototypes: int = 64,
                 num_categories: int = 8,
                 dtmb_momentum: float = 0.999,
                 stale_threshold: int = 500,
                 cos_prune_threshold: float = 0.98,
                 bandwidth: float = 0.5):
        super().__init__()
        self.pse  = PromptSemanticEncoder(pretrained)
        self.tpe  = ThreatPatternEncoder(pretrained)
        self.proj = ProjectionHead(in_dim=1536, hidden_dim=768, out_dim=proj_dim)
        self.dtmb = DynamicThreatMemoryBank(
            num_prototypes=num_prototypes,
            num_categories=num_categories,
            proj_dim=proj_dim,
            momentum=dtmb_momentum,
            stale_threshold=stale_threshold,
            cos_prune_threshold=cos_prune_threshold,
        )
        self.aagm = AnomalyAwareGating(proj_dim=proj_dim, ctx_dim=3)

    def encode(self, input_ids: torch.Tensor,
               attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns projected embedding v and DTMB score s_dtmb.
        """
        h_pse  = self.pse(input_ids, attention_mask)       # (B, 768)
        h_tpe  = self.tpe(input_ids, attention_mask)       # (B, 768)
        v      = self.proj(h_pse, h_tpe)                   # (B, 256)
        s_dtmb = self.dtmb(v)                              # (B,)
        return v, s_dtmb

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                ctx: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (y_hat, v)
          y_hat: (B,) detection scores in [0,1]
          v:     (B, 256) projected embeddings (for contrastive loss)
        """
        v, s_dtmb = self.encode(input_ids, attention_mask)
        y_hat = self.aagm(v, s_dtmb, ctx)
        return y_hat, v
