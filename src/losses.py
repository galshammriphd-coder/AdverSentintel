"""
Loss functions for AdverSentinel-LLM training.

  L_total = L_con + alpha * L_BCE + mu * L_div       (Eq. 15)

  L_con  : Supervised Contrastive Loss (Eq. 10)
  L_BCE  : Binary Cross-Entropy with class-balanced focal weighting (Eq. 16)
  L_div  : Prototype Diversity Loss (Eq. 17)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised contrastive loss (Eq. 10).

    For each anchor i, positives are all j in the same class (j != i).
    Negatives are all other samples in the batch.

    Loss = mean over anchors of:
        -1/|P(i)| * sum_{j in P(i)} log [
            exp(sim(v_i, v_j) / tau) /
            sum_{k != i} exp(sim(v_i, v_k) / tau)
        ]
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, D) L2-normalized projected embeddings
        labels:     (B,)   binary {0, 1}
        Returns: scalar loss
        """
        B = embeddings.size(0)
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # L2-normalize embeddings
        v = F.normalize(embeddings, dim=-1)                # (B, D)

        # Cosine similarity matrix
        sim_matrix = torch.mm(v, v.t()) / self.tau         # (B, B)

        # Mask out self-similarities on the diagonal
        diag_mask = torch.eye(B, dtype=torch.bool, device=v.device)
        sim_matrix = sim_matrix.masked_fill(diag_mask, float('-inf'))

        # Denominator: sum over all k != i
        log_denom = torch.logsumexp(sim_matrix, dim=1)     # (B,)

        # Positive mask: same class, different index
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        pos_mask  = labels_eq & ~diag_mask                       # (B, B)

        loss = torch.tensor(0.0, device=v.device)
        valid_anchors = 0

        for i in range(B):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_indices) == 0:
                continue
            # log-prob for each positive pair
            log_probs = sim_matrix[i][pos_indices] - log_denom[i]
            loss += -log_probs.mean()
            valid_anchors += 1

        if valid_anchors == 0:
            return torch.tensor(0.0, device=v.device, requires_grad=True)

        return loss / valid_anchors


class FocalBCELoss(nn.Module):
    """
    Binary cross-entropy with focal weighting (gamma=2.0) for class imbalance.
    Used as the classification component L_BCE (Eq. 16).
    """

    def __init__(self, gamma: float = 2.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, y_hat: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        y_hat:  (B,) predicted probability in [0,1]
        labels: (B,) ground truth {0.0, 1.0}
        """
        labels = labels.float()
        bce    = F.binary_cross_entropy(y_hat, labels, reduction='none')
        # Focal weight: (1 - p_t)^gamma
        p_t    = y_hat * labels + (1.0 - y_hat) * (1.0 - labels)
        focal  = (1.0 - p_t) ** self.gamma
        # Apply class weight for the positive class
        weight = labels * self.pos_weight + (1.0 - labels)
        return (focal * weight * bce).mean()


class PrototypeDiversityLoss(nn.Module):
    """
    Prototype diversity loss (Eq. 17) — discourages prototype collapse.

    L_div = -1 / (M*(M-1)) * sum_{j != k} ||m_j - m_k||_2
    """

    def forward(self, prototypes: torch.Tensor) -> torch.Tensor:
        """
        prototypes: (M, D) prototype embeddings
        Returns: scalar loss (negated mean pairwise distance)
        """
        M = prototypes.size(0)
        if M < 2:
            return torch.tensor(0.0, device=prototypes.device)

        # Pairwise L2 distances
        diff = prototypes.unsqueeze(0) - prototypes.unsqueeze(1)  # (M,M,D)
        dist = diff.norm(dim=-1)                                   # (M,M)

        # Mask diagonal
        mask = ~torch.eye(M, dtype=torch.bool, device=prototypes.device)
        avg_dist = dist[mask].mean()

        return -avg_dist   # Maximize spread -> minimize negative


class AdverSentinelLoss(nn.Module):
    """
    Combined training loss (Eq. 15):
        L_total = L_con + alpha * L_BCE + mu * L_div
    """

    def __init__(self, temperature: float = 0.07,
                 alpha: float = 1.0, mu: float = 0.1,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.contrastive_loss  = SupervisedContrastiveLoss(temperature)
        self.bce_loss          = FocalBCELoss(gamma=focal_gamma)
        self.diversity_loss    = PrototypeDiversityLoss()
        self.alpha = alpha
        self.mu    = mu

    def forward(self, y_hat: torch.Tensor,
                embeddings: torch.Tensor,
                labels: torch.Tensor,
                prototypes: torch.Tensor) -> dict:
        """
        y_hat:      (B,) detection scores
        embeddings: (B, D) projected embeddings
        labels:     (B,) ground truth
        prototypes: (M, D) current DTMB prototypes
        Returns: dict with 'total', 'contrastive', 'bce', 'diversity'
        """
        l_con = self.contrastive_loss(embeddings, labels)
        l_bce = self.bce_loss(y_hat, labels)
        l_div = self.diversity_loss(prototypes)

        total = l_con + self.alpha * l_bce + self.mu * l_div

        return {
            'total':       total,
            'contrastive': l_con.detach(),
            'bce':         l_bce.detach(),
            'diversity':   l_div.detach(),
        }
