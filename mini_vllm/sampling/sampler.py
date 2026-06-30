"""Token sampling: greedy, top-k, top-p (nucleus sampling)."""
from __future__ import annotations
import torch
import torch.nn.functional as F


class Sampler:
    """Stateless — call sample() directly."""

    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, vocab_size) — raw pre-softmax scores
            temperature: 0.0 → greedy; >0 → stochastic
            top_p: nucleus probability mass to keep (1.0 = no filter)
            top_k: keep only top-k tokens (-1 = no filter)
        Returns:
            (batch,) sampled token ids
        """
        if temperature == 0.0:
            return logits.argmax(dim=-1)

        logits = logits / temperature

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            threshold, _ = torch.topk(logits, k, dim=-1)
            # zero out anything below the k-th value
            min_threshold = threshold[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < min_threshold, float("-inf"))

        if top_p < 1.0:
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # shift right so the token that *pushes* cumulative over top_p is kept
            remove = (cumulative - sorted_probs) > top_p
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            # scatter back to vocabulary order
            probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
            return torch.multinomial(probs, num_samples=1).squeeze(-1)

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
