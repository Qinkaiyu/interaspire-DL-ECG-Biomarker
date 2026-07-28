from __future__ import annotations

import torch


def cox_partial_likelihood_loss(
    log_risk: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood using the Breslow tie approximation."""
    n_events = event.sum()
    if n_events.item() == 0:
        return log_risk.sum() * 0.0

    order = torch.argsort(time, descending=True)
    ordered_risk = log_risk[order]
    ordered_event = event[order]
    log_risk_set = torch.logcumsumexp(ordered_risk, dim=0)
    partial_log_likelihood = (
        (ordered_risk - log_risk_set) * ordered_event
    ).sum()
    return -partial_log_likelihood / n_events

