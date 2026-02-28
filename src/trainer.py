"""Training and evaluation loops for the Spike Transformer.

Both functions are deliberately stateless (no global state); all required
objects are passed as arguments so they can be reused across K-fold splits.
"""
import numpy as np
import torch

from src.config import LAMBDA_D, LAMBDA_FEAS, N_ZONES


def train_epoch(
    model, loader, optimizer, scheduler,
    criterion_ce, criterion_nll, criterion_d, feas_loss, device,
):
    """One training epoch.

    Returns
    -------
    losses : dict[str, float]  — averaged losses (total, cls, pos, d, feas)
    acc    : float             — zone classification accuracy
    """
    model.train()
    totals = dict(loss=0.0, cls=0.0, pos=0.0, d=0.0, feas=0.0,
                  correct=0, n=0, batches=0)

    for batch in loader:
        wf           = batch["waveforms"].to(device)
        sid          = batch["shank_ids"].to(device)
        mask         = batch["mask"].to(device)
        targets      = batch["targets"].to(device)
        d_targets    = batch["d_targets"].to(device)
        zone_targets = batch["zone_targets"].to(device)

        optimizer.zero_grad()
        cls_logits, mus, sigmas, d_pred = model(wf, sid, mask)

        # Zone classification loss
        loss_cls = criterion_ce(cls_logits, zone_targets)

        # Per-zone Gaussian NLL regression loss
        loss_pos = torch.tensor(0.0, device=device)
        for z in range(N_ZONES):
            zmask = zone_targets == z
            if zmask.any():
                loss_pos = loss_pos + criterion_nll(
                    mus[z][zmask], targets[zmask], sigmas[z][zmask] ** 2
                )

        # Curvilinear distance MSE loss
        loss_d = criterion_d(d_pred.squeeze(-1), d_targets)

        # Feasibility loss (penalise out-of-corridor predictions)
        probs       = torch.softmax(cls_logits, dim=1).unsqueeze(-1)
        mu_combined = (probs * torch.stack(mus, dim=1)).sum(dim=1)
        loss_feas   = feas_loss(mu_combined)

        loss = loss_cls + loss_pos + LAMBDA_D * loss_d + LAMBDA_FEAS * loss_feas
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        totals["loss"]    += loss.item()
        totals["cls"]     += loss_cls.item()
        totals["pos"]     += loss_pos.item()
        totals["d"]       += loss_d.item()
        totals["feas"]    += loss_feas.item()
        totals["correct"] += (cls_logits.argmax(1) == zone_targets).sum().item()
        totals["n"]       += len(zone_targets)
        totals["batches"] += 1

    nb     = totals["batches"]
    losses = {k: totals[k] / nb for k in ("loss", "cls", "pos", "d", "feas")}
    acc    = totals["correct"] / totals["n"]
    return losses, acc


@torch.no_grad()
def eval_epoch(
    model, loader,
    criterion_ce, criterion_nll, criterion_d, feas_loss, device,
):
    """Evaluation epoch (no gradient computation).

    Returns
    -------
    losses  : dict[str, float]
    acc     : float
    arrays  : list of np.ndarray
              [mu, sigma, probs, d_pred, targets, d_targets, zone_targets]
    """
    model.eval()
    totals  = dict(loss=0.0, cls=0.0, pos=0.0, d=0.0, feas=0.0,
                   correct=0, n=0, batches=0)
    buffers = {k: [] for k in
               ("mu", "sigma", "probs", "d", "targets", "d_targets", "zone_targets")}

    for batch in loader:
        wf           = batch["waveforms"].to(device)
        sid          = batch["shank_ids"].to(device)
        mask         = batch["mask"].to(device)
        targets      = batch["targets"].to(device)
        d_targets    = batch["d_targets"].to(device)
        zone_targets = batch["zone_targets"].to(device)

        mu, sigma, probs, d_pred     = model.predict(wf, sid, mask)
        cls_logits, mus, sigmas_z, _ = model(wf, sid, mask)

        loss_cls = criterion_ce(cls_logits, zone_targets)
        loss_pos = torch.tensor(0.0, device=device)
        for z in range(N_ZONES):
            zmask = zone_targets == z
            if zmask.any():
                loss_pos = loss_pos + criterion_nll(
                    mus[z][zmask], targets[zmask], sigmas_z[z][zmask] ** 2
                )
        loss_d    = criterion_d(d_pred.squeeze(-1), d_targets)
        loss_feas = feas_loss(mu)
        loss      = loss_cls + loss_pos + LAMBDA_D * loss_d + LAMBDA_FEAS * loss_feas

        totals["loss"]    += loss.item()
        totals["cls"]     += loss_cls.item()
        totals["pos"]     += loss_pos.item()
        totals["d"]       += loss_d.item()
        totals["feas"]    += loss_feas.item()
        totals["correct"] += (cls_logits.argmax(1) == zone_targets).sum().item()
        totals["n"]       += len(zone_targets)
        totals["batches"] += 1

        buffers["mu"].append(mu.cpu().numpy())
        buffers["sigma"].append(sigma.cpu().numpy())
        buffers["probs"].append(probs.cpu().numpy())
        buffers["d"].append(d_pred.cpu().numpy())
        buffers["targets"].append(targets.cpu().numpy())
        buffers["d_targets"].append(d_targets.cpu().numpy())
        buffers["zone_targets"].append(zone_targets.cpu().numpy())

    nb     = totals["batches"]
    losses = {k: totals[k] / nb for k in ("loss", "cls", "pos", "d", "feas")}
    acc    = totals["correct"] / totals["n"]
    arrays = [np.concatenate(buffers[k])
              for k in ("mu", "sigma", "probs", "d", "targets", "d_targets", "zone_targets")]
    return losses, acc, arrays

