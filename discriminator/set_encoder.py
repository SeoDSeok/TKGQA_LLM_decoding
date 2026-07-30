"""Question-conditioned set-wise temporal scorer (R2).

first/last/before/after are all *relational over the candidate set*: the answer is
argmin/argmax (first/last) or a threshold test against an anchor (before/after).
A pointwise scorer cannot express argmin/argmax, so candidates must attend to each
other. Each layer = candidate↔candidate self-attention (the comparison) +
candidate→question cross-attention (the question sets the direction/anchor). No
positional encoding: the scorer is permutation-equivariant over candidates
(Set-Transformer principle), so the score cannot depend on candidate order.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from discriminator.encode_time import TimeEncoder


class SetLayer(nn.Module):
    def __init__(self, h, heads=4, p=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(h, heads, dropout=p, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(h, heads, dropout=p, batch_first=True)
        self.n1, self.n2, self.n3 = nn.LayerNorm(h), nn.LayerNorm(h), nn.LayerNorm(h)
        self.ff = nn.Sequential(nn.Linear(h, 2 * h), nn.GELU(), nn.Dropout(p), nn.Linear(2 * h, h))

    def forward(self, x, zq, key_padding_mask):
        s, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.n1(x + s)
        c, _ = self.cross_attn(x, zq, zq, need_weights=False)  # zq: (B,1,H)
        x = self.n2(x + c)
        x = self.n3(x + self.ff(x))
        return x


class TemporalSetScorer(nn.Module):
    def __init__(self, edim=384, h=256, time_dim=64, layers=2, heads=4, p=0.1,
                 time_mode="bochner", pointwise=False, use_leak=False, cond="text"):
        super().__init__()
        self.time_mode = time_mode          # 'bochner' | 'zero'  (Φ ablation)
        self.pointwise = pointwise          # True -> no candidate self-attn (R2 ablation)
        self.use_leak = use_leak            # True -> feed rank/is_min/is_max (leakage probe)
        self.cond = cond                    # 'text' (question emb) | 'polarity' (LLM-supplied, operator-agnostic)
        self.time = TimeEncoder(time_dim)
        tdim = self.time.out_dim
        cand_in = 2 * edim + tdim + (3 if use_leak else 0)
        if cond == "polarity_signed":
            # direction (wants_later) is folded into the candidate features as a
            # SIGN, so after signing every operator reduces to the same problem
            # ("prefer high signed time / positive signed anchor-distance"). This
            # is what makes it generalize to held-out operators (breaks the 2x2
            # factorial XOR). Extra per-candidate dims: signed anchor-dist + thr.
            cand_in += 2
        elif cond == "polarity_interval":
            # unify EVERY operator as an interval predicate + superlative direction
            # (Allen-style): valid iff signed-lower-dist > 0 AND signed-upper-dist > 0;
            # direction dir∈{-1 min, 0 none, +1 max} sets which valid t to prefer.
            # first/last = superlative no-bound; before/after = one bound; during =
            # both bounds; equal = tight interval. Extra dims: sdl, sdu, hl, hu.
            cand_in += 4
        self.cand_proj = nn.Sequential(nn.Linear(cand_in, h), nn.GELU(), nn.LayerNorm(h))
        if cond == "polarity":
            self.q_proj = nn.Sequential(nn.Linear(2 + tdim, h), nn.GELU(), nn.LayerNorm(h))
        elif cond == "polarity_signed":
            self.q_proj = nn.Sequential(nn.Linear(1 + tdim, h), nn.GELU(), nn.LayerNorm(h))  # thr + anchorΦ
        elif cond == "polarity_interval":
            self.q_proj = nn.Sequential(nn.Linear(3 + 2 * tdim, h), nn.GELU(), nn.LayerNorm(h))
        else:
            self.q_proj = nn.Sequential(nn.Linear(edim + tdim, h), nn.GELU(), nn.LayerNorm(h))
        if pointwise:
            self.layers = None
            self.pw = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, h))
        else:
            self.layers = nn.ModuleList([SetLayer(h, heads, p) for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, 1))

    def _phi(self, t_rel, t_abs):
        if self.time_mode == "zero":
            return torch.zeros(*t_rel.shape, self.time.out_dim, device=t_rel.device)
        return self.time(t_rel, t_abs)

    def forward(self, batch):
        t_rel, t_abs = batch["t_rel"], batch["t_abs"]              # (B,K)
        if self.cond == "polarity_signed":
            pol = batch["polarity"]                                # [wl, thr, has, a_rel, a_abs]
            sign = (2 * pol[:, 0:1] - 1)                           # (B,1) wants_later -> +1 / earlier -> -1
            thr = pol[:, 1:2]; has = pol[:, 2:3]; a_abs = pol[:, 4:5]
            st = sign * (t_rel - 0.5)                              # (B,K) signed relative time
            sd = sign * (t_abs - a_abs) * has                     # (B,K) signed anchor distance (>0 = satisfies)
            phi = self._phi(st, sd)                                # Φ over signed features
            thr_b = thr.expand(-1, t_rel.shape[1]).unsqueeze(-1)   # (B,K,1)
            feats = [batch["o_emb"], batch["r_emb"], phi, sd.unsqueeze(-1), thr_b]
        elif self.cond == "polarity_interval":
            pol = batch["polarity_iv"]                             # [dir,hl,hu,Lr,La,Ur,Ua]
            dr = pol[:, 0:1]; hl = pol[:, 1:2]; hu = pol[:, 2:3]
            La = pol[:, 4:5]; Ua = pol[:, 6:7]
            st = dr * (t_rel - 0.5)                                # superlative direction term
            sdl = (t_abs - La) * hl                                # >0 satisfies lower bound
            sdu = (Ua - t_abs) * hu                                # >0 satisfies upper bound
            phi = self._phi(st, t_abs)
            K = t_rel.shape[1]
            feats = [batch["o_emb"], batch["r_emb"], phi, sdl.unsqueeze(-1), sdu.unsqueeze(-1),
                     hl.expand(-1, K).unsqueeze(-1), hu.expand(-1, K).unsqueeze(-1)]
        else:
            phi = self._phi(t_rel, t_abs)                          # (B,K,tdim)
            feats = [batch["o_emb"], batch["r_emb"], phi]
        if self.use_leak:
            feats.append(batch["leak"])
        x = self.cand_proj(torch.cat(feats, dim=-1))               # (B,K,H)
        # conditioning rep + anchor Φ
        if self.cond == "polarity":
            pol = batch["polarity"]                                # [wl, thr, has, a_rel, a_abs]
            a_phi = self._phi(pol[:, 3:4], pol[:, 4:5]).squeeze(1) * pol[:, 2:3]
            zq = self.q_proj(torch.cat([pol[:, 0:2], a_phi], dim=-1)).unsqueeze(1)
        elif self.cond == "polarity_signed":
            pol = batch["polarity"]
            a_phi = self._phi(pol[:, 3:4], pol[:, 4:5]).squeeze(1) * pol[:, 2:3]
            zq = self.q_proj(torch.cat([pol[:, 1:2], a_phi], dim=-1)).unsqueeze(1)  # thr + anchorΦ (no raw wl)
        elif self.cond == "polarity_interval":
            pol = batch["polarity_iv"]                             # [dir,hl,hu,Lr,La,Ur,Ua]
            lphi = self._phi(pol[:, 3:4], pol[:, 4:5]).squeeze(1) * pol[:, 1:2]   # lower Φ * hl
            uphi = self._phi(pol[:, 5:6], pol[:, 6:7]).squeeze(1) * pol[:, 2:3]   # upper Φ * hu
            zq = self.q_proj(torch.cat([pol[:, 0:3], lphi, uphi], dim=-1)).unsqueeze(1)
        else:
            a = batch["anchor"]                                    # [has, a_rel, a_abs]
            a_phi = self._phi(a[:, 1:2], a[:, 2:3]).squeeze(1) * a[:, 0:1]   # (B,tdim)
            zq = self.q_proj(torch.cat([batch["q_emb"], a_phi], dim=-1)).unsqueeze(1)  # (B,1,H)
        mask = batch["mask"]                                       # (B,K) True=pad
        if self.pointwise:
            B, K, H = x.shape
            h = self.pw(torch.cat([x, zq.expand(-1, K, -1)], dim=-1))
        else:
            h = x
            for layer in self.layers:
                h = layer(h, zq, key_padding_mask=mask)
        zq_e = zq.expand(-1, h.shape[1], -1)
        s = self.head(torch.cat([h, zq_e, h * zq_e], dim=-1)).squeeze(-1)  # (B,K)
        # raw scores; masking is applied by the caller with a finite constant
        # (baking -inf here makes logsumexp backward produce NaN).
        return s

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
