"""Absolute-Dominance v6 reference bank fitter & portable compressor for ModelTrace.

Fits the full v6 unified model (Windowed 325-token cdf225/lex1065/PL + Monotonic Student/Cauchy
Dual-Debiased GLS Pair Views + 6-Channel Heteroscedastic Guard-Ring vMF + Landmark Multimodal KRR
+ Sparse ET & Block-4 Unpaired-Covariance Sibling Experts + 2-Fold Inner OOF Contrastive Heads)
directly on the JS-identical portable feature extractors, and exports compact Base64 tensors.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, gammaln, ive, logsumexp

import v6_portable as vp

PL_HAZARD_ALPHA = 20.0
PL_HAZARD_ITERS = 40
PL_RANK_L2 = 1.0 / 355.0

REG_SOURCE = 0.20
REG_PROMPT = 0.50
REG_PAIR_PROMPT = 1.00
REG_EFFORT = 0.25
SIBLING_COS_THRESHOLD = 0.35
SIBLING_MIN_SHARED_PROMPTS = 4

NONLOW_EFFORTS = ("medium", "high", "xhigh")
VALID_EFFORTS = ("low", "medium", "high", "xhigh", "unknown")

_PURE_CACHE: dict[tuple[int, ...], tuple[Any, ...]] = {}
_AR_STATE_CACHE: dict[bytes, tuple[Any, ...]] = {}
_SEQ_CACHE: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}


def get_cached_pure(tup: tuple[int, ...]):
    hit = _PURE_CACHE.get(tup)
    if hit is not None:
        return hit
    out = vp.extract_pure_portable_tuple(tup)
    if len(_PURE_CACHE) < 8192:
        _PURE_CACHE[tup] = out
    return out


def get_cached_seq(tup: tuple[int, ...], arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hit = _SEQ_CACHE.get(tup)
    if hit is not None:
        return hit
    out = (vp.extract_taboo_fast(arr), vp.extract_seq_grammar(arr))
    if len(_SEQ_CACHE) < 8192:
        _SEQ_CACHE[tup] = out
    return out


def get_cached_ar_state(arr: np.ndarray):
    key = arr.astype(np.int16).tobytes()
    hit = _AR_STATE_CACHE.get(key)
    if hit is not None:
        z0_i16, packed_u8, ap_lo, ap_hi, ap_r2, ap_c2 = hit
        return (z0_i16, (packed_u8 & 7).astype(np.int8), (packed_u8 >> 3).astype(np.int8), ap_lo, ap_hi, ap_r2, ap_c2)
    st = vp.precompute_ar_state(arr)
    if len(_AR_STATE_CACHE) < 4096:
        z0_i16, lag_b, occ_c, ap_lo, ap_hi, ap_r2, ap_c2 = st
        packed_u8 = (occ_c.astype(np.uint8) << 3) | lag_b.astype(np.uint8)
        _AR_STATE_CACHE[key] = (z0_i16, packed_u8, ap_lo, ap_hi, ap_r2, ap_c2)
    return st


def psi_vmf_exact(x: np.ndarray | float, d: int) -> np.ndarray | float:
    nu = 0.5 * d - 1.0
    xa = np.asarray(x, dtype=np.float64)
    small = xa < 1e-7
    xs = np.where(small, 1.0, xa)
    val = np.log(np.maximum(ive(nu, xs), 1e-300)) + xs - nu * np.log(0.5 * xs)
    zero_val = -gammaln(nu + 1.0) + (xa * xa) / (2.0 * d)
    out = np.where(small, zero_val, val)
    return float(out) if np.ndim(x) == 0 else out


def build_psi_spline_dict(d: int, x_max: float = 1400.0, n_pts: int = 513) -> dict[str, Any]:
    nu = 0.5 * d - 1.0
    xs = np.linspace(0.0, x_max, n_pts, dtype=np.float64)
    pz = float(psi_vmf_exact(0.0, d))
    ys = np.asarray(psi_vmf_exact(xs, d), dtype=np.float64) - pz
    ys[0] = 0.0
    dys = np.zeros(n_pts, dtype=np.float64)
    nz = xs[1:]
    dys[1:] = ive(nu + 1.0, nz) / np.maximum(ive(nu, nz), 1e-300)
    return {
        "x_max": float(x_max),
        "y": vp.pack_f64(ys),
        "dy": vp.pack_f64(dys),
    }


def banerjee_kappa(r_bar: float, d: int) -> float:
    rb = float(np.clip(r_bar, 0.05, 0.96))
    return float(rb * (d - rb * rb) / max(1.0 - rb * rb, 1e-4))


def _fast_right_svd(B: np.ndarray) -> tuple[None, np.ndarray, np.ndarray]:
    if B.size == 0 or B.shape[0] == 0:
        return None, np.zeros(0, dtype=np.float64), np.zeros((0, B.shape[1]), dtype=np.float64)
    if B.shape[1] <= B.shape[0]:
        ev, V = np.linalg.eigh(B.T @ B)
        keep = ev > 1e-12
        if not np.any(keep):
            return None, np.zeros(0, dtype=np.float64), np.zeros((0, B.shape[1]), dtype=np.float64)
        idx = np.flatnonzero(keep)[::-1]
        s = np.sqrt(ev[idx])
        return None, s, V[:, idx].T
    ev, U = np.linalg.eigh(B @ B.T)
    keep = ev > 1e-12
    if not np.any(keep):
        return None, np.zeros(0, dtype=np.float64), np.zeros((0, B.shape[1]), dtype=np.float64)
    idx = np.flatnonzero(keep)[::-1]
    s = np.sqrt(ev[idx])
    return None, s, (U[:, idx].T @ B) / s[:, None]


def _factorize_joint_pair_matrices(
    c1: np.ndarray,
    c3: np.ndarray,
    max_rank: int = 96,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Jointly factorizes [c1; c3] into a single shared row-scaled int8 basis Q [R, D]
    plus least-squares re-projected int16 weights W1, W3 [P, R] to cancel Q quantization error."""
    P, D = c1.shape
    c_both = np.vstack([c1, c3])
    if P == 0 or D == 0 or float(np.max(np.abs(c_both))) < 1e-14:
        r0 = min(1, D)
        W1_p = vp.pack_mat_i16(np.zeros((P, r0), dtype=np.float64))
        W3_p = vp.pack_mat_i16(np.zeros((P, r0), dtype=np.float64))
        Q_p = vp.pack_mat_i8(np.zeros((r0, D), dtype=np.float64))
        return W1_p, W3_p, Q_p, vp.unpack_mat_i16(W1_p), vp.unpack_mat_i16(W3_p), vp.unpack_mat_i16(Q_p)
    _, sv, Vt = _fast_right_svd(c_both)
    R = min(max_rank, len(sv))
    if R == 0:
        R = 1
        Q = np.zeros((1, D), dtype=np.float64)
    else:
        Q = Vt[:R]
    Q_p = vp.pack_mat_i8(Q)
    Q_q = vp.unpack_mat_i16(Q_p)
    # Exact least-squares projection onto quantized Q_q cancels int8 basis non-orthogonality
    G_qq = Q_q @ Q_q.T + 1e-10 * np.eye(R, dtype=np.float64)
    W1_ls = np.linalg.solve(G_qq, Q_q @ c1.T).T
    W3_ls = np.linalg.solve(G_qq, Q_q @ c3.T).T
    W1_p = vp.pack_mat_i16(W1_ls)
    W3_p = vp.pack_mat_i16(W3_ls)
    return W1_p, W3_p, Q_p, vp.unpack_mat_i16(W1_p), vp.unpack_mat_i16(W3_p), Q_q


def _factorize_pair_matrix(c_mat: np.ndarray, max_rank: int = 96) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    """Factorizes c_mat [P, D] into quantized W [P, R], Q [R, D] and returns both packed dicts & unpacked float64s."""
    P, D = c_mat.shape
    if P == 0 or D == 0 or float(np.max(np.abs(c_mat))) < 1e-14:
        r0 = min(1, D)
        W_p = vp.pack_mat_i16(np.zeros((P, r0), dtype=np.float64))
        Q_p = vp.pack_mat_i16(np.zeros((r0, D), dtype=np.float64))
        return W_p, Q_p, vp.unpack_mat_i16(W_p), vp.unpack_mat_i16(Q_p)
    _, sv, Vt = _fast_right_svd(c_mat)
    R = min(max_rank, len(sv))
    if R == 0:
        R = 1
        Q = np.zeros((1, D), dtype=np.float64)
        W = np.zeros((P, 1), dtype=np.float64)
    else:
        Q = Vt[:R]
        W = c_mat @ Q.T
    Q_p = vp.pack_mat_i16(Q)
    Q_q = vp.unpack_mat_i16(Q_p)
    # Re-project W onto quantized Q_q via least squares / orthonormal projection to cancel Q quantization error!
    W_p = vp.pack_mat_i16(c_mat @ Q_q.T)
    W_q = vp.unpack_mat_i16(W_p)
    return W_p, Q_p, W_q, Q_q


def fit_pooled_hazard(orders: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / max(float(w.sum()), 1e-12)
    grouped: dict[bytes, tuple[np.ndarray, float]] = {}
    for order, weight in zip(orders, w):
        key = order.astype(np.int16).tobytes()
        if key in grouped:
            prev_o, prev_w = grouped[key]
            grouped[key] = (prev_o, prev_w + float(weight))
        else:
            grouped[key] = (order, float(weight))
    canon_items = [grouped[k] for k in sorted(grouped)]
    n_eff_total = float(len(orders))

    count = np.zeros(vp.DIMENSION, dtype=np.float64)
    for order, weight in canon_items:
        count[order] += weight * n_eff_total
    count += PL_HAZARD_ALPHA

    hazard = np.ones(vp.DIMENSION, dtype=np.float64)
    for _ in range(PL_HAZARD_ITERS):
        exposure = np.zeros(vp.DIMENSION, dtype=np.float64)
        h_sum = float(hazard.sum())
        for order, weight in canon_items:
            total = h_sum - np.r_[0.0, np.cumsum(hazard[order])[:-1]]
            cumulative = np.cumsum(1.0 / np.maximum(total, 1e-9))
            row_exp = np.full(vp.DIMENSION, cumulative[-1], dtype=np.float64)
            row_exp[order] = cumulative
            exposure += (weight * n_eff_total) * row_exp
        updated = count / (exposure + PL_HAZARD_ALPHA / max(float(hazard.mean()), 1e-12))
        hazard = updated / float(updated.mean())
    return hazard


def fit_pooled_ar_law(arr_list: Sequence[np.ndarray], src_idx: np.ndarray, M: int, per_class_cap: int = 2):
    sel_ix, sel_w = [], []
    present = 0
    for c in range(M):
        c_ix = np.flatnonzero(src_idx == c)
        if len(c_ix) == 0:
            continue
        present += 1
        pick = c_ix if len(c_ix) <= per_class_cap else c_ix[np.linspace(0, len(c_ix) - 1, per_class_cap, dtype=np.int64)]
        for i in pick:
            sel_ix.append(int(i))
            sel_w.append(1.0 / len(pick))
    sub_w = np.asarray(sel_w, dtype=np.float64) / max(present, 1)

    s_cnt = np.bincount(src_idx, minlength=M).astype(np.float64)
    w_all = np.where(s_cnt[src_idx] > 0, (1.0 / np.maximum(s_cnt[src_idx], 1.0)) / max(present, 1), 0.0)
    full_tok = np.zeros(355, dtype=np.float64)
    for arr_i, wi in zip(arr_list, w_all):
        if wi > 0.0:
            z0 = arr_i - 1
            full_tok += wi * (np.bincount(z0, minlength=355) / float(len(z0)))
    b_init = np.log(full_tok + 1e-4)
    b_init -= float(np.mean(b_init))

    obs_tok = np.zeros(355, dtype=np.float64)
    obs_mk = np.zeros(11, dtype=np.float64)
    obs_hi = np.zeros(9, dtype=np.float64)
    sel_states = []
    for idx_i, wi in zip(sel_ix, sub_w):
        st = get_cached_ar_state(arr_list[idx_i])
        z0, lag_b, occ_c, ap_lo, ap_hi, ap_r2, ap_c2 = st
        L = len(z0)
        sel_states.append(st)
        obs_tok += wi * (np.bincount(z0, minlength=355) / float(L))
        if L > 1:
            obs_mk += (wi / float(L)) * vp._MARKOV_B[z0[:-1], z0[1:]].sum(axis=0)
        r_ix = np.arange(L)
        lb_ch = lag_b[r_ix, z0]
        oc_ch = occ_c[r_ix, z0]
        for b_id in range(1, 7):
            obs_hi[b_id - 1] += (wi / float(L)) * float(np.sum(lb_ch == b_id))
        obs_hi[6] += (wi / float(L)) * float(np.sum(oc_ch >= 1))
        obs_hi[7] += (wi / float(L)) * float(np.sum(oc_ch >= 2))
        if L >= 3:
            z_tail = z0[2:]
            obs_hi[8] += (wi / float(L)) * float(np.sum((z_tail >= ap_lo) & (z_tail < ap_hi)))
    obs_dyn = np.r_[obs_mk, obs_hi]

    def _obj_grad(theta: np.ndarray):
        beta = theta[:355] - float(np.mean(theta[:355]))
        a_mk = theta[355:366]
        a_hi = theta[366:375]
        M_base = (vp._MARKOV_B @ a_mk) + beta[None, :]
        lag_w_lut = np.array([0.0, a_hi[0], a_hi[1], a_hi[2], a_hi[3], a_hi[4], a_hi[5]], dtype=np.float64)
        occ_w_lut = np.array([0.0, a_hi[6], a_hi[6] + a_hi[7]], dtype=np.float64)
        a_ap = float(a_hi[8])

        db = beta - b_init
        loss = 0.5 * 5e-3 * float(db @ db) + 0.5 * 1e-3 * float(a_mk @ a_mk + a_hi @ a_hi)
        exp_tok = np.zeros(355, dtype=np.float64)
        exp_hi = np.zeros(9, dtype=np.float64)
        P_by_prev = np.zeros((355, 355), dtype=np.float64)

        for (z0, lag_b, occ_c, ap_lo, ap_hi, ap_r2, ap_c2), wi in zip(sel_states, sub_w):
            L = len(z0)
            logits = lag_w_lut[lag_b] + occ_w_lut[occ_c]
            logits[0] += beta
            if L > 1:
                logits[1:] += M_base[z0[:-1]]
            if len(ap_r2) and a_ap != 0.0:
                logits[ap_r2, ap_c2] += a_ap

            lse = logsumexp(logits, axis=1)
            loss += wi * float(np.mean(lse - logits[np.arange(L), z0]))
            P = np.exp(logits - lse[:, None])
            w_step = wi / float(L)
            exp_tok += w_step * P.sum(axis=0)
            if L > 1:
                np.add.at(P_by_prev, z0[:-1], w_step * P[1:])
            bc_lag = np.bincount(lag_b.ravel(), weights=P.ravel(), minlength=7)
            exp_hi[:6] += w_step * bc_lag[1:7]
            bc_occ = np.bincount(occ_c.ravel(), weights=P.ravel(), minlength=3)
            exp_hi[6] += w_step * (bc_occ[1] + bc_occ[2])
            exp_hi[7] += w_step * bc_occ[2]
            if L >= 3:
                cp = np.hstack([np.zeros((L - 2, 1)), np.cumsum(P[2:], axis=1)])
                r_ap = np.arange(L - 2)
                exp_hi[8] += w_step * float(np.sum(cp[r_ap, ap_hi] - cp[r_ap, ap_lo]))

        exp_mk = np.einsum("uv,uvk->k", P_by_prev, vp._MARKOV_B)
        grad_b = (exp_tok - obs_tok) + 5e-3 * db
        grad_b -= float(np.mean(grad_b))
        grad_a = (np.r_[exp_mk, exp_hi] - obs_dyn) + 1e-3 * theta[355:]
        return float(loss), np.r_[grad_b, grad_a]

    x0 = np.r_[b_init, np.zeros(20, dtype=np.float64)]
    res = minimize(_obj_grad, x0, method="L-BFGS-B", jac=True, options=dict(maxiter=16, ftol=1e-7, gtol=1e-4))
    th = res.x
    b0 = th[:355] - float(np.mean(th[:355]))
    a0 = th[355:]
    M_base = (vp._MARKOV_B @ a0[:11]) + b0[None, :]
    return b0, a0, M_base


def _pack_orders_batch(orders_list: Sequence[np.ndarray], weights: np.ndarray):
    N = len(orders_list)
    lengths = np.asarray([len(o) for o in orders_list], dtype=np.int64)
    L_max = int(lengths.max())
    O_pad = np.zeros((N, L_max), dtype=np.int64)
    valid = np.zeros((N, L_max), dtype=bool)
    unobs = np.ones((N, 355), dtype=bool)
    pos_in_seq = np.full((N, 355), -1, dtype=np.int64)
    for i, o in enumerate(orders_list):
        L = len(o)
        O_pad[i, :L] = o
        valid[i, :L] = True
        unobs[i, o] = False
        pos_in_seq[i, o] = np.arange(L)
    w_step = weights / lengths.astype(np.float64)
    obs_weighted = np.sum(w_step[:, None] * (~unobs), axis=0)
    return {
        "N": N,
        "L_max": L_max,
        "O_pad": O_pad,
        "valid": valid,
        "unobs": unobs,
        "pos_in_seq": pos_in_seq,
        "w_step": w_step,
        "obs_weighted": obs_weighted,
    }


def _pl_loss_and_grad(beta: np.ndarray, packed: dict[str, Any], prior_mean: np.ndarray, lam_per_obs: float):
    b = beta - float(np.mean(beta))
    N, L_max = packed["N"], packed["L_max"]
    O_pad, valid, unobs = packed["O_pad"], packed["valid"], packed["unobs"]
    pos_in_seq, w_step = packed["pos_in_seq"], packed["w_step"]

    b_mat = np.where(unobs, b[None, :], -np.inf)
    has_unobs = unobs.any(axis=1)
    log_out = np.full(N, -np.inf, dtype=np.float64)
    if has_unobs.any():
        log_out[has_unobs] = logsumexp(b_mat[has_unobs], axis=1)

    b_chosen = np.where(valid, b[O_pad], -np.inf)
    suf = np.logaddexp.accumulate(b_chosen[:, ::-1], axis=1)[:, ::-1]
    log_Z = np.logaddexp(suf, log_out[:, None])

    step_nll = np.where(valid, log_Z - b_chosen, 0.0)
    data_nll = float(np.sum(w_step * np.sum(step_nll, axis=1)))
    diff = b - prior_mean
    reg = 0.5 * lam_per_obs * float(np.dot(diff, diff))

    inv_Z_w = np.where(valid, w_step[:, None] * np.exp(-log_Z), 0.0)
    C = np.cumsum(inv_Z_w, axis=1)
    tot_C = C[:, -1]
    idx_clamped = np.maximum(pos_in_seq, 0)
    C_at_pos = C[np.arange(N)[:, None], idx_clamped]
    S = np.sum(np.where(pos_in_seq >= 0, C_at_pos, tot_C[:, None]), axis=0)

    grad = np.exp(b) * S - packed["obs_weighted"] + lam_per_obs * diff
    grad -= float(np.mean(grad))
    return data_nll + reg, grad


def fit_pl_rank_view(orders: Sequence[np.ndarray], y_tr: np.ndarray, M: int, hazard_init: np.ndarray | None = None, maxiter: int = 80):
    s_cnt = np.bincount(y_tr, minlength=M).astype(np.float64)
    present = int(np.sum(s_cnt > 0))
    w_bal = np.where(s_cnt[y_tr] > 0, (1.0 / np.maximum(s_cnt[y_tr], 1.0)) / max(present, 1), 0.0)
    if hazard_init is not None:
        b_init = np.log(np.maximum(hazard_init, 1e-6))
        b_init -= float(np.mean(b_init))
    else:
        b_init = np.zeros(355, dtype=np.float64)

    packed_all = _pack_orders_batch(orders, w_bal)
    res_sh = minimize(
        _pl_loss_and_grad,
        b_init,
        args=(packed_all, np.zeros(355), PL_RANK_L2),
        method="L-BFGS-B",
        jac=True,
        options=dict(maxiter=maxiter, ftol=1e-9, gtol=1e-5),
    )
    shared = res_sh.x - float(np.mean(res_sh.x))

    coefs = np.empty((M, 355), dtype=np.float64)
    for c in range(M):
        ix = np.flatnonzero(y_tr == c)
        if len(ix) == 0:
            coefs[c] = shared
            continue
        sub_o = [orders[i] for i in ix]
        sub_w = np.full(len(ix), 1.0 / len(ix), dtype=np.float64)
        pk = _pack_orders_batch(sub_o, sub_w)
        res_c = minimize(
            _pl_loss_and_grad,
            shared.copy(),
            args=(pk, shared, PL_RANK_L2),
            method="L-BFGS-B",
            jac=True,
            options=dict(maxiter=maxiter, ftol=1e-9, gtol=1e-5),
        )
        coefs[c] = res_c.x - float(np.mean(res_c.x))
    return {"shared": shared, "coefs": coefs}


def fit_portable_krr(
    Xtr: np.ndarray,
    y_tr: np.ndarray,
    M: int,
    prompt_keys: Sequence[str],
    efforts_tr: Sequence[str],
    pca_dim: int = 144,
    landmarks_per_class: int = 7,
) -> tuple[dict[str, Any], dict[str, Any]]:
    s_cnt = np.bincount(y_tr, minlength=M).astype(np.float64)
    present_mask = s_cnt > 0
    n_pres = max(int(np.sum(present_mask)), 1)
    c_means = np.stack([Xtr[y_tr == c].mean(axis=0) if s_cnt[c] > 0 else np.zeros(Xtr.shape[1]) for c in range(M)])
    mu = c_means[present_mask].mean(axis=0)
    w_r = np.where(s_cnt[y_tr] > 0, (1.0 / np.maximum(s_cnt[y_tr], 1.0)) / float(n_pres), 0.0)
    sd = np.sqrt(np.sum(((Xtr - mu) ** 2) * w_r[:, None], axis=0)) + 1e-8
    Z = (Xtr - mu) / sd

    srcs_by_pk = defaultdict(set)
    for ci, pk in zip(y_tr.tolist(), prompt_keys):
        srcs_by_pk[pk].add(ci)
    act_p = sorted([pk for pk, sset in srcs_by_pk.items() if len(sset) >= 4])

    nuis_vecs = []
    if len(act_p) >= 4:
        p_map = {pk: i for i, pk in enumerate(act_p)}
        P = len(act_p)
        p_means = np.zeros((P, Z.shape[1]), dtype=np.float64)
        p_n = np.zeros(P, dtype=np.float64)
        z_cls = np.stack([Z[y_tr == c].mean(axis=0) if s_cnt[c] > 0 else np.zeros(Z.shape[1]) for c in range(M)])
        res_c = Z - z_cls[y_tr]
        for i, pk in enumerate(prompt_keys):
            if pk in p_map:
                pid = p_map[pk]
                p_means[pid] += res_c[i]
                p_n[pid] += 1.0
        p_means /= np.maximum(p_n[:, None], 1.0)
        p_means -= p_means.mean(axis=0, keepdims=True)
        _, sv, Vt = _fast_right_svd(p_means)
        r_e = min(4, int(np.sum(sv > sv[0] * 0.18))) if len(sv) > 0 else 0
        if r_e > 0:
            nuis_vecs.append(Vt[:r_e])

    low_ix = [i for i, ef in enumerate(efforts_tr) if ef == "low"]
    hi_ix = [i for i, ef in enumerate(efforts_tr) if ef in NONLOW_EFFORTS]
    if len(low_ix) >= 4 and len(hi_ix) >= 4:
        cot_d = []
        for c in range(M):
            li = [i for i in low_ix if y_tr[i] == c]
            hi = [i for i in hi_ix if y_tr[i] == c]
            if li and hi:
                cot_d.append(Z[hi].mean(axis=0) - Z[li].mean(axis=0))
        if cot_d:
            dc = np.mean(cot_d, axis=0)
            dc /= max(float(np.linalg.norm(dc)), 1e-9)
            nuis_vecs.append(dc[None, :])

    if nuis_vecs:
        Q_n, _ = np.linalg.qr(np.vstack(nuis_vecs).T)
        V_n = Q_n.T
        Z_dn = Z - (Z @ V_n.T) @ V_n
    else:
        V_n = np.zeros((0, Z.shape[1]), dtype=np.float64)
        Z_dn = Z

    # Class-balanced SVD subspace projection for compact Landmark KRR
    Zw_pca = Z_dn * np.sqrt(w_r)[:, None]
    _, sv_p, Vt_p = _fast_right_svd(Zw_pca)
    d_k = min(pca_dim, len(sv_p))
    V_pca = Vt_p[:d_k] if d_k > 0 else np.eye(min(pca_dim, Z.shape[1]), Z.shape[1])
    P_krr = V_pca - (V_pca @ V_n.T) @ V_n if len(V_n) else V_pca
    P_krr_p = vp.pack_mat_i8(P_krr)
    P_krr_q = vp.unpack_mat_i16(P_krr_p)

    Z_sub = Z @ P_krr_q.T
    lm_idx = []
    for c in range(M):
        c_ix = np.flatnonzero(y_tr == c)
        if len(c_ix) == 0:
            continue
        if len(c_ix) <= landmarks_per_class:
            lm_idx.extend(c_ix.tolist())
        else:
            pick = c_ix[np.linspace(0, len(c_ix) - 1, landmarks_per_class, dtype=np.int64)]
            lm_idx.extend(pick.tolist())
    Z_lm_raw = Z_sub[lm_idx]
    Z_lm_p = vp.pack_mat_i16(Z_lm_raw)
    Z_lm_q = vp.unpack_mat_i16(Z_lm_p)
    n_lm = len(Z_lm_q)

    # Bandwidth from landmark within-class distances
    y_lm = y_tr[lm_idx]
    lm_n2 = np.sum(Z_lm_q * Z_lm_q, axis=1)
    D2_ll = np.maximum(lm_n2[:, None] + lm_n2[None, :] - 2.0 * (Z_lm_q @ Z_lm_q.T), 0.0)
    upper = np.triu(np.ones((n_lm, n_lm), dtype=bool), 1)
    same = y_lm[:, None] == y_lm[None, :]
    pos_w = D2_ll[upper & same & (D2_ll > 1e-12)]
    pos_all = D2_ll[upper & (D2_ll > 1e-12)]
    bw = float(np.median(pos_w)) if len(pos_w) >= 6 else (float(np.median(pos_all)) if len(pos_all) else float(Z_sub.shape[1]))
    bw = max(bw, 1.0)

    # Full Nyström KRR fit over all N training rows!
    tr_n2 = np.sum(Z_sub * Z_sub, axis=1, keepdims=True)
    D2_nl = np.maximum(tr_n2 + lm_n2[None, :] - 2.0 * (Z_sub @ Z_lm_q.T), 0.0)
    K_nl_rbf = np.exp(-D2_nl / bw)
    K_nl_lap = np.exp(-np.sqrt(D2_nl / bw))
    K_ll_rbf = np.exp(-D2_ll / bw)
    K_ll_lap = np.exp(-np.sqrt(D2_ll / bw))

    wts_n = len(Z_sub) / (float(n_pres) * np.maximum(s_cnt[y_tr], 1.0))
    Y_oh = np.eye(M, dtype=np.float64)[y_tr]

    A_rbf = (K_nl_rbf.T * wts_n[None, :]) @ K_nl_rbf + 0.14 * K_ll_rbf + 1e-4 * np.eye(n_lm)
    B_rbf = (K_nl_rbf.T * wts_n[None, :]) @ Y_oh
    alpha_rbf = np.linalg.solve(A_rbf, B_rbf)

    A_lap = (K_nl_lap.T * wts_n[None, :]) @ K_nl_lap + 0.14 * K_ll_lap + 1e-4 * np.eye(n_lm)
    B_lap = (K_nl_lap.T * wts_n[None, :]) @ Y_oh
    alpha_lap = np.linalg.solve(A_lap, B_lap)

    H_rbf_raw = np.linalg.inv(K_ll_rbf + 0.14 * np.eye(n_lm))
    H_lap_raw = np.linalg.inv(K_ll_lap + 0.14 * np.eye(n_lm))
    H_rbf_p = vp.pack_mat_i16(H_rbf_raw)
    H_lap_p = vp.pack_mat_i16(H_lap_raw)

    mu_p = vp.pack_f32(mu)
    sd_p = vp.pack_f32(sd)
    alpha_rbf_p = vp.pack_f32(alpha_rbf)
    alpha_lap_p = vp.pack_f32(alpha_lap)
    packed = {
        "n_lm": int(n_lm),
        "bw": float(bw),
        "mu": mu_p,
        "sd": sd_p,
        "P_krr": P_krr_p,
        "Z_lm": Z_lm_p,
        "alpha_rbf": alpha_rbf_p,
        "alpha_lap": alpha_lap_p,
        "H_rbf": H_rbf_p,
        "H_lap": H_lap_p,
    }
    decoded = {
        "mu": vp.unpack_f64(mu_p),
        "sd": vp.unpack_f64(sd_p),
        "P_krr": P_krr_q,
        "Z_lm": Z_lm_q,
        "bw": float(bw),
        "alpha_rbf": vp.unpack_f64(alpha_rbf_p, (int(n_lm), M)),
        "alpha_lap": vp.unpack_f64(alpha_lap_p, (int(n_lm), M)),
        "H_rbf": vp.unpack_mat_i16(H_rbf_p),
        "H_lap": vp.unpack_mat_i16(H_lap_p),
    }
    return packed, decoded


def _solve_debiased_woodbury_pairs(
    mu_gls: np.ndarray,
    residuals: np.ndarray,
    indices: list[np.ndarray],
    pair_cells: list[dict],
    shared_nuis: np.ndarray | None = None,
    studentize: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    M, D = mu_gls.shape
    counts = np.asarray([len(ix) for ix in indices], dtype=np.int64)
    class_traces = np.asarray(
        [
            float(np.einsum("nd,nd->", residuals[ix], residuals[ix])) / max(len(ix) - 1, 1) if len(ix) >= 2 else 0.0
            for ix in indices
        ]
    )
    pos_tr = class_traces[counts >= 2]
    fb_tr = float(np.median(pos_tr)) if len(pos_tr) else float(D)
    eff_traces = np.where(counts >= 2, class_traces, fb_tr)

    if shared_nuis is not None and len(shared_nuis) > 0:
        n_sh = len(shared_nuis)
        all_rows = np.vstack([residuals, shared_nuis])
        sh_ids = np.arange(len(residuals), len(residuals) + n_sh, dtype=np.int64)
    else:
        n_sh = 0
        all_rows = residuals
        sh_ids = np.zeros(0, dtype=np.int64)

    N_all = len(all_rows)
    row_scale_global = np.ones(N_all, dtype=np.float64)
    for ix in indices:
        if len(ix) > 0:
            row_scale_global[ix] = 1.0 / (2.0 * math.sqrt(max(len(ix) - 1, 1)))
    U_scaled = all_rows * row_scale_global[:, None]
    G_scaled = U_scaled @ U_scaled.T
    U_dot_mu = U_scaled @ mu_gls.T

    pairs = np.asarray(list(itertools.combinations(range(M), 2)), dtype=np.int64)
    P_pairs = len(pairs)
    Alpha = np.zeros((P_pairs, N_all), dtype=np.float64)
    Delta_mat = np.zeros((P_pairs, D), dtype=np.float64)
    Mid_mat = np.zeros((P_pairs, D), dtype=np.float64)
    lams = np.ones(P_pairs, dtype=np.float64)
    loo_shifts = np.zeros(P_pairs, dtype=np.float64)
    valid_pair = np.zeros(P_pairs, dtype=bool)
    eye_cache: dict[int, np.ndarray] = {}

    for pi, (a, b) in enumerate(pairs):
        na, nb = int(counts[a]), int(counts[b])
        if na == 0 or nb == 0:
            continue
        valid_pair[pi] = True
        lam = 0.25 * float(eff_traces[a] + eff_traces[b]) / D + 1e-5
        lams[pi] = lam
        ids = np.r_[indices[a], indices[b], sh_ids] if n_sh else np.r_[indices[a], indices[b]]
        gram = G_scaled[np.ix_(ids, ids)].copy()
        gram.flat[:: len(gram) + 1] += lam

        shared_p2 = sorted(set(pair_cells[a]).intersection(pair_cells[b])) if (pair_cells[a] and pair_cells[b]) else []
        delta_gls = mu_gls[a] - mu_gls[b]
        mid_gls = 0.5 * (mu_gls[a] + mu_gls[b])
        if len(shared_p2) >= 4:
            d_p2 = np.mean([pair_cells[a][ck] - pair_cells[b][ck] for ck in shared_p2], axis=0)
            m_p2 = 0.5 * np.mean([pair_cells[a][ck] + pair_cells[b][ck] for ck in shared_p2], axis=0)
            wf = len(shared_p2) / (len(shared_p2) + 6.0)
            delta = (1.0 - wf) * delta_gls + wf * d_p2
            mid = (1.0 - wf) * mid_gls + wf * m_p2
            rhs = U_scaled[ids] @ delta
        else:
            delta = delta_gls
            mid = mid_gls
            rhs = U_dot_mu[ids, a] - U_dot_mu[ids, b]

        Delta_mat[pi] = delta
        Mid_mat[pi] = mid
        L_ch, _ = cho_factor(gram, lower=True, check_finite=False)
        corr = cho_solve((L_ch, True), rhs, check_finite=False)
        Alpha[pi, ids] = corr

        if na >= 2 and nb >= 2 and na != nb:
            m_sz = len(gram)
            I_m = eye_cache.get(m_sz)
            if I_m is None:
                I_m = np.eye(m_sz, dtype=np.float64)
                eye_cache[m_sz] = I_m
            L_inv = solve_triangular(L_ch, I_m, lower=True, check_finite=False)
            d_vec = np.clip(1.0 - lam * np.sum(L_inv * L_inv, axis=0), 0.0, 0.92)
            gam_a = float(na) / float(na - 1)
            gam_b = float(nb) / float(nb - 1)
            da, pa = d_vec[:na], corr[:na]
            db_v, pb_v = d_vec[na : na + nb], corr[na : na + nb]
            w_ra = (2.0 * math.sqrt(na - 1) * pa - 4.0 * da) / np.maximum(1.0 - gam_a * da, 0.08)
            w_rb = (2.0 * math.sqrt(nb - 1) * pb_v + 4.0 * db_v) / np.maximum(1.0 - gam_b * db_v, 0.08)
            loo_shifts[pi] = 0.5 * (-float(np.mean(w_ra)) - float(np.mean(w_rb)))

    coefs = Alpha @ U_scaled
    del Alpha
    np.subtract(Delta_mat, coefs, out=coefs)
    coefs /= lams[:, None]
    coefs[~valid_pair] = 0.0
    intercepts = -np.einsum("pd,pd->p", coefs, Mid_mat) + loo_shifts
    intercepts[~valid_pair] = 0.0

    if studentize:
        mahal_sq = np.maximum(np.einsum("pd,pd->p", coefs, Delta_mat), 1e-8)
        na_v = counts[pairs[:, 0]].astype(np.float64)
        nb_v = counts[pairs[:, 1]].astype(np.float64)
        n_hm = 2.0 * na_v * nb_v / np.maximum(na_v + nb_v, 1.0)
        scale_w = np.where(valid_pair, np.sqrt(n_hm / (n_hm + 2.0)) / np.sqrt(mahal_sq), 0.0)
        coefs *= scale_w[:, None]
        intercepts *= scale_w
    return pairs, coefs, intercepts


def fit_portable_pair_view(
    Xtr: np.ndarray,
    src_idx: np.ndarray,
    M: int,
    prompt_keys: Sequence[str],
    efforts: Sequence[str],
    cot_pen: float = 12.0,
    studentize: bool = False,
    cov_cap_per_class: int | None = 64,
    max_rank: int = 96,
) -> tuple[dict[str, Any], dict[str, Any]]:
    D = Xtr.shape[1]
    s_cnt = np.bincount(src_idx, minlength=M).astype(np.float64)
    present = s_cnt > 0
    n_pres = max(int(np.sum(present)), 1)

    c_means = np.stack([Xtr[src_idx == c].mean(axis=0) if s_cnt[c] > 0 else np.zeros(D) for c in range(M)])
    mu0 = c_means[present].mean(axis=0)
    w_r = np.where(s_cnt[src_idx] > 0, (1.0 / np.maximum(s_cnt[src_idx], 1.0)) / float(n_pres), 0.0)
    sc = np.sqrt(np.sum(((Xtr - c_means[src_idx]) ** 2) * w_r[:, None], axis=0))
    pos_sc = sc[sc > 1e-8]
    med_sc = float(np.median(pos_sc)) if len(pos_sc) else 1.0
    sc = np.maximum(sc, 0.15 * med_sc + 1e-6)
    Ztr = (Xtr - mu0) / sc

    by_sp = defaultdict(lambda: defaultdict(list))
    for i, (ci, pk, ef) in enumerate(zip(src_idx.tolist(), prompt_keys, efforts)):
        by_sp[(ci, pk.split("::")[0])][ef].append(i)
    cot_diffs = []
    for (_, _), em in sorted(by_sp.items(), key=lambda kv: (kv[0][1], min(kv[1].get("low", [0])))):
        if "low" in em:
            zl = Ztr[em["low"]].mean(axis=0)
            for ef in NONLOW_EFFORTS:
                if ef in em:
                    cot_diffs.append(Ztr[em[ef]].mean(axis=0) - zl)
    if cot_diffs:
        u_cot = np.mean(cot_diffs, axis=0) * (len(cot_diffs) / (len(cot_diffs) + 2.0))
        un2 = float(u_cot @ u_cot)
        if un2 > 1e-9:
            damp_c = 1.0 - 1.0 / math.sqrt(1.0 + cot_pen * un2)
            u_cot = u_cot / math.sqrt(un2)
        else:
            u_cot, damp_c = np.zeros(D, dtype=np.float64), 0.0
    else:
        u_cot, damp_c = np.zeros(D, dtype=np.float64), 0.0

    Ztr_d = Ztr - damp_c * ((Ztr @ u_cot)[:, None] * u_cot[None, :]) if damp_c > 0.0 else Ztr

    srcs_by_pk = defaultdict(set)
    for ci, pk in zip(src_idx.tolist(), prompt_keys):
        srcs_by_pk[pk].add(ci)
    act_prompts = sorted([pk for pk, sset in srcs_by_pk.items() if len(sset) >= 2])
    p_map = {pk: i for i, pk in enumerate(act_prompts)}
    reg_p_list = [REG_PAIR_PROMPT if len(srcs_by_pk[pk]) == 2 else REG_PROMPT for pk in act_prompts]
    P = len(act_prompts)
    e_map = {ef: i for i, ef in enumerate(NONLOW_EFFORTS)}
    E = len(NONLOW_EFFORTS)

    N = len(Xtr)
    A = np.zeros((N, M + P + E), dtype=np.float64)
    for i, (ci, pk, ef) in enumerate(zip(src_idx.tolist(), prompt_keys, efforts)):
        A[i, ci] = 1.0
        if pk in p_map:
            A[i, M + p_map[pk]] = 1.0
        if ef in e_map:
            A[i, M + P + e_map[ef]] = 1.0
    G_gls = A.T @ A + np.diag([REG_SOURCE] * M + reg_p_list + [REG_EFFORT] * E)
    theta = np.linalg.solve(G_gls, A.T @ Ztr_d)
    mu_gls = theta[:M]
    nu_hat = theta[M : M + P]
    eta_hat = theta[M + P :]

    Z_clean = Ztr_d.copy()
    Z_env = Ztr_d.copy()
    for i, (pk, ef) in enumerate(zip(prompt_keys, efforts)):
        if pk in p_map:
            Z_clean[i] -= nu_hat[p_map[pk]]
        if ef in e_map:
            Z_clean[i] -= eta_hat[e_map[ef]]
            Z_env[i] -= eta_hat[e_map[ef]]

    pair_cells = [dict() for _ in range(M)]
    by_cpe = defaultdict(list)
    for i, (ci, pk, ef) in enumerate(zip(src_idx.tolist(), prompt_keys, efforts)):
        if len(srcs_by_pk[pk]) >= 2:
            by_cpe[(ci, pk, ef)].append(i)
    for (ci, pk, ef), ix in by_cpe.items():
        pair_cells[ci][(pk, ef)] = Z_env[ix].mean(axis=0)

    full_indices = [np.flatnonzero(src_idx == c) for c in range(M)]
    if cov_cap_per_class is not None:
        cov_indices = [
            ix if len(ix) <= cov_cap_per_class else ix[np.linspace(0, len(ix) - 1, cov_cap_per_class, dtype=np.int64)]
            for ix in full_indices
        ]
    else:
        cov_indices = full_indices

    emp_clean = np.stack([Z_clean[ix].mean(axis=0) if len(ix) > 0 else np.zeros(D) for ix in cov_indices])
    res_clean = Z_clean - emp_clean[src_idx]
    pairs, c1, b1 = _solve_debiased_woodbury_pairs(mu_gls, res_clean, cov_indices, pair_cells, shared_nuis=None, studentize=studentize)

    emp_env = np.stack([Z_env[ix].mean(axis=0) if len(ix) > 0 else np.zeros(D) for ix in cov_indices])
    res_env = Z_env - emp_env[src_idx]
    multi_p_idx = [p_map[pk] for pk in act_prompts if len(srcs_by_pk[pk]) >= 4]
    U_sh = np.zeros((0, D), dtype=np.float64)
    if len(multi_p_idx) >= 4:
        nu_m = nu_hat[multi_p_idx]
        nu_m = nu_m - nu_m.mean(axis=0, keepdims=True)
        _, sv_p, Vt_p = _fast_right_svd(nu_m)
        r_p = min(6, int(np.sum(sv_p > sv_p[0] * 0.15))) if len(sv_p) > 0 else 0
        if r_p > 0:
            U_sh = math.sqrt(0.85 / len(multi_p_idx)) * (sv_p[:r_p, None] * Vt_p[:r_p])
    _, c3, b3 = _solve_debiased_woodbury_pairs(mu_gls, res_env, cov_indices, pair_cells, shared_nuis=U_sh, studentize=studentize)

    W1_p, W3_p, Q_p, W1_q, W3_q, Q_q = _factorize_joint_pair_matrices(c1, c3, max_rank=max_rank)

    h_tr = Ztr_d @ Q_q.T
    raw1 = h_tr @ W1_q.T + b1[None, :]
    rms1 = np.sqrt(np.mean(raw1 * raw1, axis=1))
    r1 = max(float(np.median([float(np.median(rms1[ix])) for ix in full_indices if len(ix) > 0])) * (1.45 if studentize else 1.0), 0.5)

    raw3 = h_tr @ W3_q.T + b3[None, :]
    rms3 = np.sqrt(np.mean(raw3 * raw3, axis=1))
    r3 = max(float(np.median([float(np.median(rms3[ix])) for ix in full_indices if len(ix) > 0])) * (1.50 if studentize else 1.05), 0.5)

    mu0_p = vp.pack_f32(mu0)
    sc_p = vp.pack_f32(sc)
    u_cot_p = vp.pack_f32(u_cot)
    packed = {
        "mu0": mu0_p,
        "sc": sc_p,
        "u_cot": u_cot_p,
        "damp_c": float(damp_c),
        "Q": Q_p,
        "W1": W1_p,
        "b1": vp.pack_f64(b1),
        "r1": float(r1),
        "W3": W3_p,
        "b3": vp.pack_f64(b3),
        "r3": float(r3),
    }
    decoded = {
        "mu0": vp.unpack_f64(mu0_p),
        "sc": vp.unpack_f64(sc_p),
        "u_cot": vp.unpack_f64(u_cot_p),
        "damp_c": float(damp_c),
        "W1": W1_q,
        "Q1": Q_q,
        "b1": b1,
        "r1": float(r1),
        "W3": W3_q,
        "Q3": Q_q,
        "b3": b3,
        "r3": float(r3),
        "pairs": pairs,
    }
    return packed, decoded


def fit_portable_vmf_channel(
    X_raw: np.ndarray,
    src_idx: np.ndarray,
    M: int,
    prompt_keys: Sequence[str],
    efforts: Sequence[str],
    splines_cache: dict[int, tuple[dict, dict]],
    cot_pen: float = 12.0,
    env_keys: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    N, D = X_raw.shape
    s_cnt = np.bincount(src_idx, minlength=M).astype(np.float64)
    present_mask = s_cnt > 0
    n_pres = max(int(np.sum(present_mask)), 1)

    c_means = np.stack([X_raw[src_idx == c].mean(axis=0) if s_cnt[c] > 0 else np.zeros(D) for c in range(M)])
    mu0 = c_means[present_mask].mean(axis=0)
    w_r = np.where(s_cnt[src_idx] > 0, (1.0 / np.maximum(s_cnt[src_idx], 1.0)) / float(n_pres), 0.0)
    res0 = X_raw - c_means[src_idx]
    pos_std = np.sqrt(np.sum((res0 ** 2) * w_r[:, None], axis=0))
    med_std = float(np.median(pos_std[pos_std > 1e-8])) if np.any(pos_std > 1e-8) else 1.0
    sc = np.maximum(pos_std, 0.15 * med_std + 1e-6)
    Z = (X_raw - mu0) / sc

    srcs_by_pk = defaultdict(set)
    for ci, pk in zip(src_idx.tolist(), prompt_keys):
        srcs_by_pk[pk].add(ci)
    act_prompts = sorted([pk for pk, sset in srcs_by_pk.items() if len(sset) >= 2])
    p_map = {pk: i for i, pk in enumerate(act_prompts)}
    reg_p_list = [REG_PAIR_PROMPT if len(srcs_by_pk[pk]) == 2 else REG_PROMPT for pk in act_prompts]
    P = len(act_prompts)
    e_map = {ef: i for i, ef in enumerate(NONLOW_EFFORTS)}
    E = len(NONLOW_EFFORTS)

    by_sp = defaultdict(lambda: defaultdict(list))
    for i, (ci, pk, ef) in enumerate(zip(src_idx.tolist(), prompt_keys, efforts)):
        by_sp[(ci, pk.split("::")[0])][ef].append(i)
    cot_diffs = []
    cot_by_ef: dict[str, list[np.ndarray]] = defaultdict(list)
    for (_, _), em in sorted(by_sp.items(), key=lambda kv: (kv[0][1], min(kv[1].get("low", [0])))):
        if "low" in em:
            zl = Z[em["low"]].mean(axis=0)
            for ef in NONLOW_EFFORTS:
                if ef in em:
                    diff_e = Z[em[ef]].mean(axis=0) - zl
                    cot_diffs.append(diff_e)
                    cot_by_ef[ef].append(diff_e)

    A = np.zeros((N, M + P + E), dtype=np.float64)
    for i, (ci, pk, ef) in enumerate(zip(src_idx.tolist(), prompt_keys, efforts)):
        A[i, ci] = 1.0
        if pk in p_map:
            A[i, M + p_map[pk]] = 1.0
        if ef in e_map:
            A[i, M + P + e_map[ef]] = 1.0

    G = A.T @ A + np.diag([REG_SOURCE] * M + reg_p_list + [REG_EFFORT] * E)
    G_inv = np.linalg.inv(G)
    theta = G_inv @ (A.T @ Z)
    mu_hat = theta[:M]
    nu_hat = theta[M : M + P]
    eta_hat = theta[M + P :]
    eff_n = np.where(present_mask, 1.0 / np.maximum(np.diag(G_inv)[:M], 1e-9), 0.0)

    Z_adj = Z.copy()
    Z_env = Z.copy()
    for i, (pk, ef) in enumerate(zip(prompt_keys, efforts)):
        if pk in p_map:
            Z_adj[i] -= nu_hat[p_map[pk]]
        if ef in e_map:
            Z_adj[i] -= eta_hat[e_map[ef]]
            Z_env[i] -= eta_hat[e_map[ef]]

    res_rows = Z - A @ theta
    diag_eps = np.sum((res_rows * res_rows) * w_r[:, None], axis=0)
    multi4_idx = [p_map[pk] for pk in act_prompts if len(srcs_by_pk[pk]) >= 4]
    if len(multi4_idx) >= 2:
        nu_m4 = nu_hat[multi4_idx]
        diag_nu = np.sum(nu_m4 * nu_m4, axis=0) / len(multi4_idx)
    elif P > 0:
        diag_nu = np.sum(nu_hat * nu_hat, axis=0) / P
    else:
        diag_nu = np.zeros(D, dtype=np.float64)
    d_vec = 0.60 * (diag_eps + diag_nu) + 0.35
    inv_sqrt_d = 1.0 / np.sqrt(d_vec)

    U_parts = [np.sqrt(0.40 * w_r)[:, None] * res_rows]
    if len(multi4_idx) >= 2:
        U_parts.append(math.sqrt(0.40 / len(multi4_idx)) * nu_hat[multi4_idx])
    elif P > 0:
        U_parts.append(math.sqrt(0.40 / P) * nu_hat)
    n_cot = len(cot_diffs)
    delta_cot_raw = np.zeros(D, dtype=np.float64)
    if n_cot > 0:
        delta_cot_raw = np.mean(cot_diffs, axis=0) * (n_cot / (n_cot + 2.0))
        if cot_pen > 0.0:
            for ef in NONLOW_EFFORTS:
                ef_list = cot_by_ef[ef]
                if ef_list:
                    d_ef = np.mean(ef_list, axis=0) * (len(ef_list) / (len(ef_list) + 2.0))
                    U_parts.append(math.sqrt(cot_pen) * d_ef[None, :])
    U_lr = np.vstack(U_parts)
    _, s_vals, Wt = _fast_right_svd(U_lr * inv_sqrt_d[None, :])
    damp = 1.0 - 1.0 / np.sqrt(1.0 + s_vals * s_vals)

    def _whiten(X_in: np.ndarray) -> np.ndarray:
        Y = X_in * inv_sqrt_d[None, :]
        return Y - ((Y @ Wt.T) * damp[None, :]) @ Wt if len(Wt) else Y

    M_w = _whiten(mu_hat)
    mu_bar = M_w[present_mask].mean(axis=0)
    Mc = M_w - mu_bar[None, :]
    delta_cot_w = _whiten(delta_cot_raw[None, :])[0]

    basis_seed = np.vstack([Mc[present_mask], delta_cot_w[None, :]]) if n_cot > 0 else Mc[present_mask]
    _, sb_seed, vt_seed = _fast_right_svd(basis_seed)
    r = max(min(min(n_pres, D), int(np.sum(sb_seed > 1e-10))), 2)
    V_sub = vt_seed[:r].copy()
    for k_i in range(r):
        idx_max = int(np.argmax(np.abs(V_sub[k_i])))
        if V_sub[k_i, idx_max] < 0.0:
            V_sub[k_i] = -V_sub[k_i]

    M_sub = Mc @ V_sub.T
    emp_var = np.sum(M_sub[present_mask] ** 2, axis=0) / float(n_pres)
    lam_B = np.maximum(emp_var - 0.5 * float(np.mean(1.0 / np.maximum(eff_n[present_mask], 1.0))), 0.05)

    P_proj_raw = (V_sub - ((V_sub @ Wt.T) * damp[None, :]) @ Wt) * inv_sqrt_d[None, :]
    P_proj_p = vp.pack_mat_i8(P_proj_raw)
    P_proj_q = vp.unpack_mat_i16(P_proj_p)
    offset_sub = mu_bar @ V_sub.T

    Z_sub = Z_adj @ P_proj_q.T - offset_sub[None, :]
    Utr = Z_sub / np.maximum(np.linalg.norm(Z_sub, axis=1, keepdims=True), 1e-12)

    raw_r_by_c = np.full(M, np.nan, dtype=np.float64)
    n_base_by_c = np.zeros(M, dtype=np.float64)
    for c in range(M):
        if not present_mask[c]:
            continue
        ix = [i for i in np.flatnonzero(src_idx == c).tolist() if efforts[i] in ("low", "unknown")]
        if not ix:
            ix = np.flatnonzero(src_idx == c).tolist()
        n_base_by_c[c] = len(ix)
        if len(ix) >= 2:
            us = Utr[ix]
            su = us.sum(axis=0)
            vals = [float(us[j] @ ((su - us[j]) / max(float(np.linalg.norm(su - us[j])), 1e-9))) for j in range(len(ix))]
            raw_r_by_c[c] = max(0.12, float(np.mean(vals)))

    R_pool = float(np.nanmean(raw_r_by_c)) if np.any(np.isfinite(raw_r_by_c)) else 0.55
    shrink = (eff_n[:, None] * lam_B[None, :]) / (eff_n[:, None] * lam_B[None, :] + 1.0)
    M_eb = shrink * M_sub
    M_unit_base = M_eb / np.maximum(np.linalg.norm(M_eb, axis=1, keepdims=True), 1e-12)

    w_eb = np.maximum(n_base_by_c - 1.0, 0.0) / (np.maximum(n_base_by_c - 1.0, 0.0) + 5.0)
    R_c = np.where(np.isfinite(raw_r_by_c), w_eb * raw_r_by_c + (1.0 - w_eb) * R_pool, R_pool) * 0.80
    kappa_c = np.asarray([banerjee_kappa(float(rc), r) for rc in R_c], dtype=np.float64)
    n_eff_capped = (eff_n * 10.0) / (eff_n + 10.0)
    rho_c = kappa_c * np.maximum(n_eff_capped, 1.10)

    cos_mat = M_unit_base @ M_unit_base.T
    np.fill_diagonal(cos_mat, -1.0)
    if M >= 3:
        s_cos = np.sort(cos_mat, axis=1)
        top_cos = 0.72 * s_cos[:, -1] + 0.28 * s_cos[:, -2]
    elif M == 2:
        top_cos = np.max(cos_mat, axis=1)
    else:
        top_cos = np.full(M, 0.5)
    rho_sib_raw = np.asarray(
        [banerjee_kappa(max(0.12, 0.84 * float(tc) * float(rc / 0.80)), r) for tc, rc in zip(top_cos, R_c)],
        dtype=np.float64,
    )
    rho_sib = np.minimum(rho_sib_raw, 0.72 * rho_c)

    Zw_env_c = _whiten(Z_env) - mu_bar[None, :]
    pres_ids = np.flatnonzero(s_cnt > 0)
    loo_r1 = np.full(M, np.nan, dtype=np.float64)
    if len(pres_ids) >= 2:
        P_pr = len(pres_ids)
        sums_e = np.stack([Zw_env_c[src_idx == c].sum(axis=0) for c in pres_ids])
        cnts_e = s_cnt[pres_ids]
        means_e = sums_e / cnts_e[:, None]
        G_e = means_e @ means_e.T
        G_reg_e = G_e + 1e-8 * np.eye(P_pr)
        Z_dot_e = Zw_env_c @ means_e.T
        z_n2_e = np.sum(Zw_env_c * Zw_env_c, axis=1)
        for loc_c, c in enumerate(pres_ids.tolist()):
            nc = int(s_cnt[c])
            if nc < 2:
                continue
            ix_c = np.flatnonzero(src_idx == c)
            oth = [j for j in range(P_pr) if j != loc_c]
            G_oth_inv = np.linalg.inv(G_reg_e[np.ix_(oth, oth)])
            sum_c_n2 = float(sums_e[loc_c] @ sums_e[loc_c])
            sum_dot_oth = G_e[loc_c, oth] * nc
            inv_nm1 = 1.0 / float(nc - 1)
            cs_list = []
            for i_row in ix_c:
                zi_n2 = float(z_n2_e[i_row])
                zi_dot_sum = float(Z_dot_e[i_row, loc_c]) * nc
                mu_n2 = max((sum_c_n2 - 2.0 * zi_dot_sum + zi_n2) * (inv_nm1 ** 2), 1e-12)
                zi_dot_mu = (zi_dot_sum - zi_n2) * inv_nm1
                p_z_oth = Z_dot_e[i_row, oth]
                p_mu_oth = (sum_dot_oth - p_z_oth) * inv_nm1
                w_mu = G_oth_inv @ p_mu_oth
                w_z = G_oth_inv @ p_z_oth
                z_in_oth2 = max(float(p_z_oth @ w_z), 0.0)
                mu_in_oth2 = max(float(p_mu_oth @ w_mu), 0.0)
                dot_in_oth = float(p_z_oth @ w_mu)
                rem_n2 = max(mu_n2 - mu_in_oth2, 1e-12)
                z_along_rem = (zi_dot_mu - dot_in_oth) / math.sqrt(rem_n2)
                z_sub_n2 = max(z_in_oth2 + z_along_rem * z_along_rem, 1e-12)
                mu_sub_n2 = max(mu_in_oth2 + rem_n2, 1e-12)
                cs_list.append(zi_dot_mu / math.sqrt(z_sub_n2 * mu_sub_n2))
            loo_r1[c] = float(np.clip(np.mean(cs_list), 0.15, 0.70))

    R1_pool_oos = float(np.nanmean(loo_r1)) if np.any(np.isfinite(loo_r1)) else 0.42
    w_eb_oos = np.maximum(s_cnt - 1.0, 0.0) / (np.maximum(s_cnt - 1.0, 0.0) + 8.0)
    R1_oos = np.clip(np.where(np.isfinite(loo_r1), w_eb_oos * loo_r1 + (1.0 - w_eb_oos) * R1_pool_oos, R1_pool_oos), 0.18, 0.66)
    mu2_rat = (R1_oos ** 2) / np.maximum(1.0 - R1_oos ** 2, 0.10)
    R3_oos = np.clip(np.sqrt(mu2_rat / (mu2_rat + 0.46)), 0.25, 0.76)

    max_cos = np.max(cos_mat, axis=1) if M > 1 else np.full(M, 0.45)
    Rh1 = np.clip(0.68 * R1_oos * np.clip(max_cos, 0.35, 0.92), 0.10, 0.44)
    Rh3 = np.clip(0.68 * R3_oos * np.clip(max_cos, 0.35, 0.92), 0.12, 0.54)
    kap1_o = np.asarray([banerjee_kappa(float(x), r) for x in R1_oos], dtype=np.float64)
    kap3_o = np.asarray([banerjee_kappa(float(x), r) for x in R3_oos], dtype=np.float64)
    kh1_o = np.asarray([banerjee_kappa(float(x), r) for x in Rh1], dtype=np.float64)
    kh3_o = np.asarray([banerjee_kappa(float(x), r) for x in Rh3], dtype=np.float64)

    if r not in splines_cache:
        sp_p = build_psi_spline_dict(r)
        sp_d = {"x_max": float(sp_p["x_max"]), "y": vp.unpack_f64(sp_p["y"]), "dy": vp.unpack_f64(sp_p["dy"])}
        splines_cache[r] = (sp_p, sp_d)
    _, sp_dec = splines_cache[r]

    if env_keys is not None:
        n_envs_c = np.zeros(M, dtype=np.float64)
        for c in range(M):
            if present_mask[c]:
                ix_c = np.flatnonzero(src_idx == c)
                n_envs_c[c] = float(len({env_keys[i] for i in ix_c.tolist()}))
        if np.any((n_envs_c > 0.0) & (n_envs_c < 6.0)):
            well_mask = n_envs_c >= 9.0
            ref_mask = well_mask if np.any(well_mask) else present_mask
            kap_pool = float(np.median(kappa_c[ref_mask]))
            kap1_pool = float(np.median(kap1_o[ref_mask]))
            kap3_pool = float(np.median(kap3_o[ref_mask]))
            for c in range(M):
                ne = float(n_envs_c[c])
                if 0.0 < ne < 6.0:
                    eff_n[c] = min(float(eff_n[c]), 2.2 * ne)
                    w_k = max(ne - 3.0, 0.0) / (max(ne - 3.0, 0.0) + 5.0)
                    kappa_c[c] = w_k * min(float(kappa_c[c]), 1.05 * kap_pool) + (1.0 - w_k) * kap_pool
                    rho_c[c] = kappa_c[c] * max(0.45, (ne * ne) / (ne + 12.0))
                    rho_sib[c] = min(float(rho_sib[c]), 0.72 * rho_c[c])
                    kap1_o[c] = w_k * min(float(kap1_o[c]), 1.05 * kap1_pool) + (1.0 - w_k) * kap1_pool
                    kap3_o[c] = w_k * min(float(kap3_o[c]), 1.05 * kap3_pool) + (1.0 - w_k) * kap3_pool

    open_est_dec = {
        "kap1": kap1_o,
        "ln1": vp.eval_psi_spline(sp_dec, kap1_o),
        "kap3": kap3_o,
        "ln3": vp.eval_psi_spline(sp_dec, kap3_o),
        "kh1": kh1_o,
        "lnh1": vp.eval_psi_spline(sp_dec, kh1_o),
        "kh3": kh3_o,
        "lnh3": vp.eval_psi_spline(sp_dec, kh3_o),
    }
    mu0_p = vp.pack_f32(mu0)
    sc_p = vp.pack_f32(sc)
    packed = {
        "r": int(r),
        "mu0": mu0_p,
        "sc": sc_p,
        "P_proj": P_proj_p,
        "offset_sub": vp.pack_f64(offset_sub),
        "M_unit_base": vp.pack_f64(M_unit_base.ravel()),
        "kappa_c": vp.pack_f64(kappa_c),
        "rho_c": vp.pack_f64(rho_c),
        "rho_sib": vp.pack_f64(rho_sib),
        "eff_n": vp.pack_f64(eff_n),
        "open_est": {k: vp.pack_f64(v) for k, v in open_est_dec.items()},
    }
    decoded = {
        "r": int(r),
        "spline": sp_dec,
        "mu0": vp.unpack_f64(mu0_p),
        "sc": vp.unpack_f64(sc_p),
        "P_proj": P_proj_q,
        "offset_sub": offset_sub,
        "M_unit_base": M_unit_base,
        "kappa_c": kappa_c,
        "rho_c": rho_c,
        "rho_sib": rho_sib,
        "eff_n": eff_n,
        "open_est": open_est_dec,
    }
    return packed, decoded


def fit_portable_clean_refiner(
    Xcdf: np.ndarray,
    Xgeom: np.ndarray,
    et_rows: Sequence[tuple[np.ndarray, np.ndarray]],
    src_idx: np.ndarray,
    M: int,
    prompt_keys: Sequence[str],
    efforts: Sequence[str],
    M_unit_ref: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    s_cnt = np.bincount(src_idx, minlength=M).astype(np.float64)
    present_mask = s_cnt > 0
    n_pres = max(int(np.sum(present_mask)), 1)
    w_r = np.where(s_cnt[src_idx] > 0, (1.0 / np.maximum(s_cnt[src_idx], 1.0)) / float(n_pres), 0.0)
    sub_cols = list(vp.SUBLIMINAL_COLS)
    mac_cols = list(vp.MACRO_COLS)

    c_means = np.stack([Xcdf[src_idx == c].mean(axis=0) if s_cnt[c] > 0 else np.zeros(Xcdf.shape[1]) for c in range(M)])
    mu_c = c_means[present_mask].mean(axis=0)
    sc_c = np.sqrt(np.sum(((Xcdf - mu_c) ** 2) * w_r[:, None], axis=0)) + 1e-6

    g_means = np.stack([Xgeom[src_idx == c].mean(axis=0) if s_cnt[c] > 0 else np.zeros(Xgeom.shape[1]) for c in range(M)])
    mu_g = g_means[present_mask].mean(axis=0)
    sc_g = np.sqrt(np.sum(((Xgeom - mu_g) ** 2) * w_r[:, None], axis=0)) + 1e-6

    Zcdf = (Xcdf - mu_c) / sc_c
    Zgeom = (Xgeom - mu_g) / sc_g

    by_sp = defaultdict(lambda: defaultdict(list))
    by_pr = defaultdict(lambda: defaultdict(list))
    for i, (ci, pk, ef) in enumerate(zip(src_idx.tolist(), prompt_keys, efforts)):
        by_sp[(ci, pk)][ef].append(i)
        if ef in ("low", "unknown"):
            by_pr[pk][ci].append(i)

    cot_sub, cot_mg = [], []
    for (_, _), em in sorted(by_sp.items(), key=lambda kv: (kv[0][1], min(kv[1].get("low", [0])))):
        if "low" in em:
            zs0 = Zcdf[em["low"]][:, sub_cols].mean(axis=0)
            zmg0 = np.r_[Zcdf[em["low"]][:, mac_cols].mean(axis=0), Zgeom[em["low"]].mean(axis=0)]
            for ef in NONLOW_EFFORTS:
                if ef in em:
                    cot_sub.append(Zcdf[em[ef]][:, sub_cols].mean(axis=0) - zs0)
                    cot_mg.append(np.r_[Zcdf[em[ef]][:, mac_cols].mean(axis=0), Zgeom[em[ef]].mean(axis=0)] - zmg0)
    d_sub = np.mean(cot_sub, axis=0) if cot_sub else np.zeros(len(sub_cols), dtype=np.float64)
    d_mg = np.mean(cot_mg, axis=0) if cot_mg else np.zeros(len(mac_cols) + Xgeom.shape[1], dtype=np.float64)

    total_et_dim = vp.ET_ORIG_DIM
    et_orig_rows: list[tuple[np.ndarray, np.ndarray]] = []
    for ic_r, vc_r in et_rows:
        if len(ic_r) and int(ic_r[-1]) >= total_et_dim:
            m_orig = ic_r < total_et_dim
            et_orig_rows.append((ic_r[m_orig], vc_r[m_orig]))
        else:
            et_orig_rows.append((ic_r, vc_r))

    cell_et = {}
    for pk, sm in by_pr.items():
        for ci, ix in sm.items():
            if len(ix) == 1:
                cell_et[(pk, ci)] = et_orig_rows[ix[0]]
            else:
                non_empty_i = [et_orig_rows[i][0] for i in ix if len(et_orig_rows[i][0])]
                if not non_empty_i:
                    cell_et[(pk, ci)] = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64))
                else:
                    ic = np.concatenate(non_empty_i)
                    vc = np.concatenate([et_orig_rows[i][1] for i in ix if len(et_orig_rows[i][0])]) / float(len(ix))
                    oi = np.argsort(ic)
                    ui, fp = np.unique(ic[oi], return_index=True)
                    cell_et[(pk, ci)] = (ui, np.add.reduceat(vc[oi], fp))

    buf_d = np.zeros(total_et_dim, dtype=np.float64)
    buf_s = np.zeros(total_et_dim, dtype=np.float64)
    wet_buf = np.zeros(total_et_dim, dtype=np.float64)
    cos_m = M_unit_ref @ M_unit_ref.T

    pairs_packed = []
    pairs_dec = []
    for a in range(M):
        if not present_mask[a]:
            continue
        for b in range(a + 1, M):
            if not present_mask[b] or cos_m[a, b] <= SIBLING_COS_THRESHOLD:
                continue
            shared_p = sorted([pk for pk, sm in by_pr.items() if a in sm and b in sm])
            if len(shared_p) < SIBLING_MIN_SHARED_PROMPTS:
                continue

            Za_sub = np.stack([Zcdf[by_pr[pk][a]][:, sub_cols].mean(axis=0) for pk in shared_p])
            Zb_sub = np.stack([Zcdf[by_pr[pk][b]][:, sub_cols].mean(axis=0) for pk in shared_p])
            PD1 = Za_sub - Zb_sub
            Sw1 = (
                0.30 * np.cov(PD1, rowvar=False)
                + 0.70 * np.diag(np.var(PD1, axis=0))
                + 0.60 * np.eye(len(sub_cols))
                + 6.0 * np.outer(d_sub, d_sub)
            )
            w1 = np.linalg.solve(Sw1, PD1.mean(axis=0))
            w1 /= float(np.std(PD1 @ w1)) + 1e-6
            m1 = 0.5 * (Za_sub.mean(axis=0) + Zb_sub.mean(axis=0))

            inv_p = 1.0 / float(len(shared_p))
            touched = []
            for pk in shared_p:
                ia, va = cell_et[(pk, a)]
                ib, vb = cell_et[(pk, b)]
                if len(ia):
                    buf_d[ia] += va * inv_p
                    buf_s[ia] += va * inv_p
                    touched.append(ia)
                if len(ib):
                    buf_d[ib] -= vb * inv_p
                    buf_s[ib] += vb * inv_p
                    touched.append(ib)
            ut = np.unique(np.concatenate(touched)) if touched else np.zeros(0, dtype=np.int64)
            wv = np.zeros(len(ut), dtype=np.float64)
            if len(ut):
                wv = buf_d[ut] / (buf_s[ut] + 0.015)
                imp = np.abs(buf_d[ut] * wv)
                buf_d[ut] = 0.0
                buf_s[ut] = 0.0
                if len(ut) > 512:
                    top_ix = np.sort(np.argpartition(imp, -512)[-512:])
                    ut = ut[top_ix]
                    wv = wv[top_ix]
                wet_buf[ut] = wv

            sa_e = np.asarray([float(cell_et[(pk, a)][1] @ wet_buf[cell_et[(pk, a)][0]]) for pk in shared_p], dtype=np.float64)
            sb_e = np.asarray([float(cell_et[(pk, b)][1] @ wet_buf[cell_et[(pk, b)][0]]) for pk in shared_p], dtype=np.float64)
            emid = 0.5 * float(np.mean(sa_e) + np.mean(sb_e))
            estd = float(np.std(sa_e - sb_e)) + 1e-6
            if len(ut):
                wv /= estd
                wet_buf[ut] = wv
            emid /= estd

            # Keep non-negligible sparse ET weights for compact JSON
            keep_et = np.abs(wv) > 2e-4 if len(ut) else np.zeros(0, dtype=bool)
            wi_keep = ut[keep_et] if len(ut) else np.zeros(0, dtype=np.int64)
            wv_keep = wv[keep_et] if len(ut) else np.zeros(0, dtype=np.float64)
            if len(ut):
                wet_buf[ut] = 0.0
            if len(wi_keep):
                wet_buf[wi_keep] = wv_keep

            Za_mg = np.stack([np.r_[Zcdf[by_pr[pk][a]][:, mac_cols].mean(axis=0), Zgeom[by_pr[pk][a]].mean(axis=0)] for pk in shared_p])
            Zb_mg = np.stack([np.r_[Zcdf[by_pr[pk][b]][:, mac_cols].mean(axis=0), Zgeom[by_pr[pk][b]].mean(axis=0)] for pk in shared_p])
            PD2 = Za_mg - Zb_mg
            d_var = 0.80 * np.var(PD2, axis=0) + 0.70
            inv_d = 1.0 / d_var
            md2 = PD2.mean(axis=0)
            u_dmg = d_mg * inv_d
            w2 = md2 * inv_d - (16.0 * float(u_dmg @ md2) / (1.0 + 16.0 * float(d_mg @ u_dmg))) * u_dmg
            w2 /= float(np.std(PD2 @ w2)) + 1e-6
            m2 = 0.5 * (Za_mg.mean(axis=0) + Zb_mg.mean(axis=0))

            ix_a = [i for pk in shared_p for i in by_pr[pk][a]]
            ix_b = [i for pk in shared_p for i in by_pr[pk][b]]

            def _c2(ix_list: list[int]) -> np.ndarray:
                zc = Zcdf[ix_list]
                zg = Zgeom[ix_list]
                et_sc = np.asarray([float(et_orig_rows[i][1] @ wet_buf[et_orig_rows[i][0]]) - emid for i in ix_list], dtype=np.float64)
                s1 = 0.55 * ((zc[:, sub_cols] - m1[None, :]) @ w1) + 0.45 * et_sc
                s2 = (np.c_[zc[:, mac_cols], zg] - m2[None, :]) @ w2
                return np.stack([s1, s2], axis=1)

            Ca, Cb = _c2(ix_a), _c2(ix_b)
            if len(wi_keep):
                wet_buf[wi_keep] = 0.0
            mu_a, mu_b = Ca.mean(axis=0), Cb.mean(axis=0)
            var2 = 0.5 * (Ca.var(axis=0) + Cb.var(axis=0)) + 0.30
            rel = len(shared_p) / (len(shared_p) + 6.0)

            w1_p = vp.pack_f32(w1)
            m1_p = vp.pack_f32(m1)
            wv_p = vp.pack_f32(wv_keep) if len(wv_keep) else ""
            w2_p = vp.pack_f32(w2)
            m2_p = vp.pack_f32(m2)
            wv_d = vp.unpack_f64(wv_p) if len(wv_keep) else np.zeros(0, dtype=np.float64)
            pairs_packed.append(
                {
                    "a": int(a),
                    "b": int(b),
                    "w1": w1_p,
                    "m1": m1_p,
                    "wi": vp.pack_i32(wi_keep) if len(wi_keep) else "",
                    "wv": wv_p,
                    "emid": float(emid),
                    "w2": w2_p,
                    "m2": m2_p,
                    "mu_a": vp.pack_f64(mu_a),
                    "mu_b": vp.pack_f64(mu_b),
                    "var2": vp.pack_f64(var2),
                    "rel": float(rel),
                }
            )
            pairs_dec.append(
                {
                    "a": int(a),
                    "b": int(b),
                    "w1": vp.unpack_f64(w1_p),
                    "m1": vp.unpack_f64(m1_p),
                    "wi": wi_keep,
                    "wv": wv_d,
                    "w_map": {int(ik): float(vk) for ik, vk in zip(wi_keep.tolist(), wv_d.tolist())} if len(wi_keep) else {},
                    "emid": float(emid),
                    "w2": vp.unpack_f64(w2_p),
                    "m2": vp.unpack_f64(m2_p),
                    "mu_a": mu_a,
                    "mu_b": mu_b,
                    "var2": var2,
                    "rel": float(rel),
                }
            )

    mu_c_p = vp.pack_f32(mu_c)
    sc_c_p = vp.pack_f32(sc_c)
    mu_g_p = vp.pack_f32(mu_g)
    sc_g_p = vp.pack_f32(sc_g)
    return (
        {
            "mu_c": mu_c_p,
            "sc_c": sc_c_p,
            "mu_g": mu_g_p,
            "sc_g": sc_g_p,
            "pairs": pairs_packed,
        },
        {
            "mu_c": vp.unpack_f64(mu_c_p),
            "sc_c": vp.unpack_f64(sc_c_p),
            "mu_g": vp.unpack_f64(mu_g_p),
            "sc_g": vp.unpack_f64(sc_g_p),
            "pairs": pairs_dec,
        },
    )


def fit_portable_block4_sibling_expert(
    blocks: Sequence[np.ndarray],
    et_rows: Sequence[tuple[np.ndarray, np.ndarray]],
    src_idx: np.ndarray,
    M: int,
    prompt_keys: Sequence[str],
    efforts: Sequence[str],
    M_unit_ref: np.ndarray | None = None,
    *,
    env_keys: Sequence[str] | None = None,
    lam_cs: Sequence[float] = (0.08, 0.10, 0.08, 0.08, 0.08, 0.40),
    gamma_ef: float = 0.0,
    mid_nl_cap: float = 0.0,
    alpha_me: float = 0.0,
    calibrate_oof_llr: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """v6.2 6-Block Sequential + Dual-Regime Effort-Debiased + OOF-LLR Calibrated `sref_b4` Expert."""
    n_B = len(blocks)
    evs_list = list(env_keys) if env_keys is not None else list(prompt_keys)
    regimes = [pk.split("::")[-1] for pk in prompt_keys]
    s_cnt = np.bincount(src_idx, minlength=M)
    present = s_cnt > 0
    n_pres = max(int(np.sum(present)), 1)
    all_ix = [np.flatnonzero(src_idx == c) for c in range(M)]
    wt_all = np.where(s_cnt[src_idx] > 0, 1.0 / (float(n_pres) * np.maximum(s_cnt[src_idx], 1)), 0.0)

    block_scalers = []
    Z_blocks = []
    for X_blk in blocks:
        Db = X_blk.shape[1]
        cm = np.stack([X_blk[all_ix[c]].mean(axis=0) if present[c] else np.zeros(Db) for c in range(M)])
        mu_b = cm[present].mean(axis=0)
        sc_b = np.sqrt(np.sum(((X_blk - mu_b[None, :]) ** 2) * wt_all[:, None], axis=0)) + 1e-6
        Z_b = (X_blk - mu_b[None, :]) / sc_b[None, :]
        Z_blocks.append(Z_b)
        block_scalers.append((mu_b, sc_b, np.zeros((0, Db), dtype=np.float64), np.zeros(0, dtype=np.float64)))

    Z0 = Z_blocks[0]
    cm0 = np.stack([Z0[all_ix[c]].mean(axis=0) if present[c] else np.zeros(Z0.shape[1]) for c in range(M)])
    nrm0 = np.maximum(np.linalg.norm(cm0, axis=1, keepdims=True), 1e-9)
    cos_m = (cm0 / nrm0) @ (cm0 / nrm0).T

    by_pe: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    by_spk: dict[tuple[int, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for gi in range(len(src_idx)):
        c_i = int(src_idx[gi])
        by_pe[(prompt_keys[gi], efforts[gi])][c_i].append(gi)
        if efforts[gi] != "unknown":
            by_spk[(c_i, prompt_keys[gi])][efforts[gi]].append(gi)

    ef_groups = []
    for (sc_i, _), ef_map in sorted(by_spk.items()):
        if len(ef_map) >= 2:
            any_gi = next(iter(ef_map.values()))[0]
            ef_groups.append((evs_list[any_gi], sc_i, [np.asarray(ixs, dtype=np.int64) for _, ixs in sorted(ef_map.items())]))

    wet1 = np.zeros(vp.ET_EXT_DIM + 8, dtype=np.float64)
    wet2 = np.zeros(vp.ET_EXT_DIM + 8, dtype=np.float64)
    pairs_packed, pairs_dec = [], []

    for a_c in range(M):
        if not present[a_c]:
            continue
        for b_c in range(a_c + 1, M):
            if not present[b_c]:
                continue
            cos_ab = float(cos_m[a_c, b_c])
            min_n_all = min(len(all_ix[a_c]), len(all_ix[b_c]))
            sh_cells = sorted([ck for ck, sm in by_pe.items() if a_c in sm and b_c in sm])
            Mp = len({ck[0] for ck in sh_cells})
            is_multi_panel_sib = (min_n_all >= 42) and (Mp >= 15) and (cos_ab > 0.40)
            is_ultra_close_twin = (cos_ab >= 0.72) and (Mp >= 12)
            if not (is_multi_panel_sib or is_ultra_close_twin):
                continue

            use_nB = n_B if is_multi_panel_sib else min(4, n_B)
            all_ab = np.r_[all_ix[a_c], all_ix[b_c]]
            n_a_all, n_b_all = len(all_ix[a_c]), len(all_ix[b_c])
            u_evs = sorted({evs_list[i] for i in all_ab.tolist()})
            if len(u_evs) >= 4:
                n_blks = min(12, len(u_evs))
                ev_to_blk = {ev: idx % n_blks for idx, ev in enumerate(u_evs)}
                ab_blk = np.asarray([ev_to_blk[evs_list[i]] for i in all_ab.tolist()], dtype=np.int64)
                cell_blk = np.asarray([ev_to_blk[evs_list[by_pe[ck][a_c][0]]] for ck in sh_cells], dtype=np.int64)
            else:
                all_pks = sorted({prompt_keys[i] for i in all_ab.tolist()})
                if len(all_pks) < 4:
                    continue
                n_blks = min(8, len(all_pks))
                ev_to_blk = {pk: idx % n_blks for idx, pk in enumerate(all_pks)}
                ab_blk = np.asarray([ev_to_blk[prompt_keys[i]] for i in all_ab.tolist()], dtype=np.int64)
                cell_blk = np.asarray([ev_to_blk[ck[0]] for ck in sh_cells], dtype=np.int64)

            is_nl = np.asarray([ck[1] not in ("low", "unknown") for ck in sh_cells], dtype=bool)
            y_ab = np.r_[np.ones(n_a_all), -np.ones(n_b_all)]

            oof_valid = np.zeros(len(all_ab), dtype=bool)
            for blk in range(n_blks):
                if np.sum(ab_blk[:n_a_all] != blk) >= 3 and np.sum(ab_blk[n_a_all:] != blk) >= 3:
                    oof_valid[ab_blk == blk] = True
            if int(np.sum(oof_valid & (y_ab > 0))) < 6 or int(np.sum(oof_valid & (y_ab < 0))) < 6:
                continue

            oof_g = np.zeros((len(all_ab), use_nB + 2), dtype=np.float64)
            ws_g_full, mids_g_full = [], []

            for bi in range(use_nB):
                Z1 = Z_blocks[bi]
                Db = Z1.shape[1]
                Z_ab = Z1[all_ab]
                cd = np.stack([Z1[by_pe[ck][a_c]].mean(axis=0) - Z1[by_pe[ck][b_c]].mean(axis=0) for ck in sh_cells])
                cs = 0.5 * np.stack([Z1[by_pe[ck][a_c]].mean(axis=0) + Z1[by_pe[ck][b_c]].mean(axis=0) for ck in sh_cells])
                ef_vs, ef_bs = [], []
                if gamma_ef > 0.0 and is_multi_panel_sib:
                    for ev_g, sc_i, ix_l in ef_groups:
                        ms = np.stack([Z1[ixs].mean(axis=0) for ixs in ix_l])
                        gm = ms.mean(axis=0)
                        wm_e = 1.4 if sc_i in (a_c, b_c) else 0.7
                        for mv in ms:
                            ef_vs.append((mv - gm) * wm_e)
                            ef_bs.append(ev_to_blk.get(ev_g, -1))
                E_mat = np.stack(ef_vs) if ef_vs else np.zeros((0, Db), dtype=np.float64)
                E_blk = np.asarray(ef_bs, dtype=np.int64)
                M_all = np.vstack([Z_ab, E_mat]) if len(E_mat) else Z_ab
                G_all = M_all @ M_all.T
                lam_c = float(lam_cs[bi]) if bi < len(lam_cs) else 0.08

                def _fit_gb(blk_ex: int | None):
                    mr = (ab_blk != blk_ex) if blk_ex is not None else np.ones(len(all_ab), dtype=bool)
                    la = np.flatnonzero(mr[:n_a_all])
                    lb = n_a_all + np.flatnonzero(mr[n_a_all:])
                    cm_c = (cell_blk != blk_ex) if blk_ex is not None else np.ones(len(sh_cells), dtype=bool)
                    na, nb = len(la), len(lb)
                    sa, sb = 1.0 / (2.0 * math.sqrt(na)), 1.0 / (2.0 * math.sqrt(nb))
                    ma, mb = Z_ab[la].mean(axis=0), Z_ab[lb].mean(axis=0)

                    G_aa_r = G_all[np.ix_(la, la)]
                    ra_m = G_aa_r.mean(axis=1)
                    G_aa_c = G_aa_r - ra_m[:, None] - ra_m[None, :] + float(ra_m.mean())
                    G_bb_r = G_all[np.ix_(lb, lb)]
                    rb_m = G_bb_r.mean(axis=1)
                    G_bb_c = G_bb_r - rb_m[:, None] - rb_m[None, :] + float(rb_m.mean())
                    G_ab_r = G_all[np.ix_(la, lb)]
                    rab_r, rab_c = G_ab_r.mean(axis=1), G_ab_r.mean(axis=0)
                    G_ab_c = G_ab_r - rab_r[:, None] - rab_c[None, :] + float(rab_r.mean())

                    tra = max(float(np.trace(G_aa_c)) / na, 0.0)
                    trb = max(float(np.trace(G_bb_c)) / nb, 0.0)
                    n_c_sub = int(np.sum(cm_c))
                    if n_c_sub >= 4:
                        if mid_nl_cap > 0.0 and np.any(cm_c & is_nl) and np.any(cm_c & (~is_nl)):
                            c_nl, c_lo = cm_c & is_nl, cm_c & (~is_nl)
                            n_nl = int(np.sum(c_nl))
                            wd = min(0.12, n_nl / (n_nl + 25.0))
                            wm = min(mid_nl_cap, n_nl / (n_nl + 15.0))
                            dm = (1.0 - wd) * cd[c_lo].mean(axis=0) + wd * cd[c_nl].mean(axis=0)
                            mm = (1.0 - wm) * cs[c_lo].mean(axis=0) + wm * cs[c_nl].mean(axis=0)
                        else:
                            dm, mm = cd[cm_c].mean(axis=0), cs[cm_c].mean(axis=0)
                        if n_c_sub >= 12:
                            delta, mid = dm, mm
                        else:
                            wf = n_c_sub / (n_c_sub + 6.0)
                            delta = (1.0 - wf) * (ma - mb) + wf * dm
                            mid = (1.0 - wf) * 0.5 * (ma + mb) + wf * mm
                    else:
                        delta, mid = ma - mb, 0.5 * (ma + mb)

                    em_ix = np.flatnonzero((E_blk != blk_ex) if blk_ex is not None else np.ones(len(E_mat), dtype=bool)) if len(E_mat) else np.zeros(0, dtype=np.int64)
                    ne = len(em_ix)
                    lam = lam_c * (tra + trb) / Db + 0.03
                    G = np.empty((na + nb + ne, na + nb + ne), dtype=np.float64)
                    G[:na, :na] = (sa * sa) * G_aa_c
                    G[:na, na : na + nb] = (sa * sb) * G_ab_c
                    G[na : na + nb, :na] = G[:na, na : na + nb].T
                    G[na : na + nb, na : na + nb] = (sb * sb) * G_bb_c
                    if ne > 0:
                        se = math.sqrt(gamma_ef / float(ne))
                        ge_g = len(all_ab) + em_ix
                        G_ea_r, G_eb_r = G_all[np.ix_(ge_g, la)], G_all[np.ix_(ge_g, lb)]
                        G[na + nb :, :na] = (se * sa) * (G_ea_r - G_ea_r.mean(axis=1, keepdims=True))
                        G[:na, na + nb :] = G[na + nb :, :na].T
                        G[na + nb :, na : na + nb] = (se * sb) * (G_eb_r - G_eb_r.mean(axis=1, keepdims=True))
                        G[na : na + nb, na + nb :] = G[na + nb :, na : na + nb].T
                        G[na + nb :, na + nb :] = (se * se) * G_all[np.ix_(ge_g, ge_g)]
                    G.flat[:: len(G) + 1] += lam

                    rhs = [sa * (Z_ab[la] @ delta - float(ma @ delta)), sb * (Z_ab[lb] @ delta - float(mb @ delta))]
                    if ne > 0:
                        rhs.append(se * (E_mat[em_ix] @ delta))
                    corr = cho_solve(cho_factor(G, lower=True, check_finite=False), np.concatenate(rhs), check_finite=False)
                    ca, cb = corr[:na], corr[na : na + nb]
                    utc = sa * (ca @ Z_ab[la] - float(np.sum(ca)) * ma) + sb * (cb @ Z_ab[lb] - float(np.sum(cb)) * mb)
                    if ne > 0:
                        utc = utc + se * (corr[na + nb :] @ E_mat[em_ix])
                    return (delta - utc) / lam, mid

                for blk in range(n_blks):
                    if np.sum(ab_blk[:n_a_all] != blk) >= 3 and np.sum(ab_blk[n_a_all:] != blk) >= 3:
                        wf, mf = _fit_gb(blk)
                        vl = np.flatnonzero(ab_blk == blk)
                        if len(vl):
                            oof_g[vl, bi] = (Z_ab[vl] - mf) @ wf
                wf_full, mf_full = _fit_gb(None)
                ws_g_full.append(wf_full)
                mids_g_full.append(mf_full)

            def _fit_et2(blk_ex: int | None):
                cm = (cell_blk != blk_ex) if blk_ex is not None else np.ones(len(sh_cells), dtype=bool)
                c_ixs = np.flatnonzero(cm).tolist()
                if not c_ixs:
                    return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64), 0.0, 0.0
                inv_c = 1.0 / len(c_ixs)
                buf_d = np.zeros(vp.ET_ORIG_DIM, dtype=np.float64)
                buf_s = np.zeros(vp.ET_ORIG_DIM, dtype=np.float64)
                touched = []
                for ci in c_ixs:
                    ck = sh_cells[ci]
                    for r_i in by_pe[ck][a_c]:
                        ic, vc = et_rows[r_i]
                        m_o = ic < vp.ET_ORIG_DIM
                        if np.any(m_o):
                            ico, vco = ic[m_o], vc[m_o]
                            buf_d[ico] += vco * inv_c / len(by_pe[ck][a_c])
                            buf_s[ico] += vco * inv_c / len(by_pe[ck][a_c])
                            touched.append(ico)
                    for r_i in by_pe[ck][b_c]:
                        ic, vc = et_rows[r_i]
                        m_o = ic < vp.ET_ORIG_DIM
                        if np.any(m_o):
                            ico, vco = ic[m_o], vc[m_o]
                            buf_d[ico] -= vco * inv_c / len(by_pe[ck][b_c])
                            buf_s[ico] += vco * inv_c / len(by_pe[ck][b_c])
                            touched.append(ico)
                ut = np.unique(np.concatenate(touched)) if touched else np.zeros(0, dtype=np.int64)
                if not len(ut):
                    return ut, np.zeros(0, dtype=np.float64), 0.0, 0.0
                wv = buf_d[ut] / (buf_s[ut] + 0.018)
                imp = np.abs(buf_d[ut] * wv)
                if len(ut) > 1024:
                    tix = np.sort(np.argpartition(imp, -1024)[-1024:])
                    ut, wv = ut[tix], wv[tix]
                wet1[ut] = wv

                def _em_pair(mask_sub):
                    s1a, s1b = [], []
                    for ci in np.flatnonzero(mask_sub).tolist():
                        ck = sh_cells[ci]
                        s1a.append(float(np.mean([float(et_rows[gi][1][et_rows[gi][0] < vp.ET_ORIG_DIM] @ wet1[et_rows[gi][0][et_rows[gi][0] < vp.ET_ORIG_DIM]]) for gi in by_pe[ck][a_c]])))
                        s1b.append(float(np.mean([float(et_rows[gi][1][et_rows[gi][0] < vp.ET_ORIG_DIM] @ wet1[et_rows[gi][0][et_rows[gi][0] < vp.ET_ORIG_DIM]]) for gi in by_pe[ck][b_c]])))
                    return 0.5 * (float(np.mean(s1a)) + float(np.mean(s1b))), 0.0

                if mid_nl_cap > 0.0 and np.any(cm & is_nl) and np.any(cm & (~is_nl)):
                    e1_lo, _ = _em_pair(cm & (~is_nl))
                    e1_nl, _ = _em_pair(cm & is_nl)
                    n_nl = int(np.sum(cm & is_nl))
                    wm = min(mid_nl_cap, n_nl / (n_nl + 15.0))
                    em1, em2 = (1.0 - wm) * e1_lo + wm * e1_nl, 0.0
                else:
                    em1, em2 = _em_pair(cm)
                return ut, wv, em1, em2

            for blk in range(n_blks):
                if np.sum(ab_blk[:n_a_all] != blk) >= 3 and np.sum(ab_blk[n_a_all:] != blk) >= 3:
                    ut_f, _, em1_f, em2_f = _fit_et2(blk)
                    for lj in np.flatnonzero(ab_blk == blk).tolist():
                        ic, vc = et_rows[int(all_ab[lj])]
                        oof_g[lj, use_nB] = float(vc @ wet1[ic]) - em1_f
                        oof_g[lj, use_nB + 1] = float(vc @ wet2[ic]) - em2_f
                    m1_f = ut_f < vp.ET_ORIG_DIM
                    wet1[ut_f[m1_f]] = 0.0
                    wet2[ut_f[~m1_f]] = 0.0

            ut_full, wv_full, em1_full, em2_full = _fit_et2(None)
            m1_full = ut_full < vp.ET_ORIG_DIM
            wet1[ut_full[m1_full]] = 0.0
            wet2[ut_full[~m1_full]] = 0.0

            yv = y_ab[oof_valid]
            pos_m, neg_m = yv > 0, yv < 0
            Xv = oof_g[oof_valid]
            sds = np.maximum(np.sqrt(0.5 * (np.var(Xv[pos_m], axis=0) + np.var(Xv[neg_m], axis=0))), 1e-6)
            Xn = Xv / sds[None, :]
            dps = np.mean(Xn[pos_m], axis=0) - np.mean(Xn[neg_m], axis=0)
            if float(np.max(dps)) <= (0.42 if is_multi_panel_sib else 0.72):
                continue

            w_samp = np.where(pos_m, 0.5 / np.sum(pos_m), 0.5 / np.sum(neg_m))
            C_dim = use_nB + 2
            l2_reg = 0.25 if is_multi_panel_sib else 0.35

            def _lobj(th: np.ndarray):
                z = Xn @ th[:C_dim] + th[C_dim]
                yz = yv * z
                loss = float(w_samp @ np.logaddexp(0.0, -yz)) + 0.5 * l2_reg * float(th[:C_dim] @ th[:C_dim]) + 0.12 * (th[C_dim] ** 2)
                pe = -yv * expit(-yz) * w_samp
                grad = np.r_[pe @ Xn + l2_reg * th[:C_dim], float(np.sum(pe)) + 0.24 * th[C_dim]]
                return loss, grad

            b_lim = 0.35 if is_multi_panel_sib else 0.0
            res_l = minimize(_lobj, np.r_[np.maximum(dps * 0.20, 0.0), 0.0], method="L-BFGS-B", jac=True, bounds=[(0.0, 3.5)] * C_dim + [(-b_lim, b_lim)])
            if not np.isfinite(res_l.x).all() or float(np.sum(res_l.x[:C_dim])) < 0.15:
                continue

            comb_oof = Xn @ res_l.x[:C_dim]
            sd_oof = max(math.sqrt(0.5 * (float(np.var(comb_oof[pos_m])) + float(np.var(comb_oof[neg_m])))), 1e-6)
            dp_oof = float(np.mean(comb_oof[pos_m]) - np.mean(comb_oof[neg_m])) / sd_oof
            if dp_oof <= (0.45 if is_multi_panel_sib else 0.85):
                continue

            cal_mult = (dp_oof / sd_oof) if calibrate_oof_llr else 1.0
            ws_out = [np.zeros(Z_blocks[bi].shape[1], dtype=np.float64) for bi in range(n_B)]
            mids_out = [np.zeros(Z_blocks[bi].shape[1], dtype=np.float64) for bi in range(n_B)]
            bias_val = 0.0
            for bi in range(use_nB):
                sc_bi = (float(res_l.x[bi]) / float(sds[bi])) * cal_mult
                ws_out[bi] = ws_g_full[bi] * sc_bi
                mids_out[bi] = mids_g_full[bi]

            s_et1 = (float(res_l.x[use_nB]) / float(sds[use_nB])) * cal_mult
            s_et2 = (float(res_l.x[use_nB + 1]) / float(sds[use_nB + 1])) * cal_mult

            if alpha_me > 0.0 and is_multi_panel_sib:
                reg_sh: dict[str, int] = defaultdict(int)
                reg_ef: dict[str, set[str]] = defaultdict(set)
                for ck in sh_cells:
                    rg = ck[0].split("::")[-1]
                    reg_sh[rg] += 1
                    if ck[1] != "unknown":
                        reg_ef[rg].add(ck[1])
                me_regs = {rg for rg, cnt in reg_sh.items() if cnt >= 12 and len(reg_ef[rg]) >= 2}
                ix_a_me = np.asarray([gi for gi in all_ix[a_c].tolist() if regimes[gi] in me_regs], dtype=np.int64)
                ix_b_me = np.asarray([gi for gi in all_ix[b_c].tolist() if regimes[gi] in me_regs], dtype=np.int64)
                if len(ix_a_me) >= 8 and len(ix_b_me) >= 8:
                    me_bis = [bi for bi in (0, 1, 2, 5) if bi < use_nB]
                    all_me = np.r_[ix_a_me, ix_b_me]
                    na_m, nb_m = len(ix_a_me), len(ix_b_me)
                    sh_me = [ck for ck in sh_cells if ck[0].split("::")[-1] in me_regs]
                    me_evs = sorted({evs_list[i] for i in all_me.tolist()})
                    if len(me_evs) >= 2:
                        n_mb = min(12, len(me_evs))
                        me_map = {ev: idx % n_mb for idx, ev in enumerate(me_evs)}
                        me_blk = np.asarray([me_map[evs_list[i]] for i in all_me.tolist()], dtype=np.int64)
                        me_cblk = np.asarray([me_map[evs_list[by_pe[ck][a_c][0]]] for ck in sh_me], dtype=np.int64)
                        oof_m = np.zeros((len(all_me), len(me_bis)), dtype=np.float64)
                        me_fits = []
                        for m_col, bi in enumerate(me_bis):
                            Xb = blocks[bi]
                            Db = Xb.shape[1]
                            lam_c = float(lam_cs[bi]) if bi < len(lam_cs) else 0.08

                            def _fit_mb(b_ex: int | None):
                                mr = (me_blk != b_ex) if b_ex is not None else np.ones(len(all_me), dtype=bool)
                                la, lb = np.flatnonzero(mr[:na_m]), na_m + np.flatnonzero(mr[na_m:])
                                if len(la) < 3 or len(lb) < 3:
                                    return np.zeros(Db), np.zeros(Db), np.zeros(Db), np.ones(Db)
                                ga, gb = all_me[la], all_me[lb]
                                mu2 = 0.5 * (Xb[ga].mean(axis=0) + Xb[gb].mean(axis=0))
                                sc2 = np.sqrt(0.5 * (np.mean((Xb[ga] - mu2) ** 2, axis=0) + np.mean((Xb[gb] - mu2) ** 2, axis=0))) + 1e-6
                                Za, Zbm = (Xb[ga] - mu2) / sc2, (Xb[gb] - mu2) / sc2
                                ma, mb = Za.mean(axis=0), Zbm.mean(axis=0)
                                cm_m = (me_cblk != b_ex) if b_ex is not None else np.ones(len(sh_me), dtype=bool)
                                c_ixs = np.flatnonzero(cm_m).tolist()
                                if len(c_ixs) >= 4:
                                    dm = np.mean([((Xb[by_pe[sh_me[ci]][a_c]].mean(axis=0) - mu2) / sc2) - ((Xb[by_pe[sh_me[ci]][b_c]].mean(axis=0) - mu2) / sc2) for ci in c_ixs], axis=0)
                                    mm = 0.5 * np.mean([((Xb[by_pe[sh_me[ci]][a_c]].mean(axis=0) - mu2) / sc2) + ((Xb[by_pe[sh_me[ci]][b_c]].mean(axis=0) - mu2) / sc2) for ci in c_ixs], axis=0)
                                    wf = 1.0 if len(c_ixs) >= 12 else len(c_ixs) / (len(c_ixs) + 6.0)
                                    delta, mid = (1.0 - wf) * (ma - mb) + wf * dm, (1.0 - wf) * 0.5 * (ma + mb) + wf * mm
                                else:
                                    delta, mid = ma - mb, 0.5 * (ma + mb)
                                ra, rb = Za - ma, Zbm - mb
                                lam = lam_c * (float(np.einsum("nd,nd->", ra, ra)) / len(la) + float(np.einsum("nd,nd->", rb, rb)) / len(lb)) / Db + 0.03
                                u = np.vstack([ra / (2.0 * math.sqrt(len(la))), rb / (2.0 * math.sqrt(len(lb)))])
                                G = u @ u.T
                                G.flat[:: len(G) + 1] += lam
                                corr = cho_solve(cho_factor(G, lower=True, check_finite=False), u @ delta, check_finite=False)
                                return (delta - u.T @ corr) / lam, mid, mu2, sc2

                            for blk in range(n_mb):
                                wf_m, mf_m, mu2_f, sc2_f = _fit_mb(blk)
                                vl = np.flatnonzero(me_blk == blk)
                                if len(vl):
                                    oof_m[vl, m_col] = (((Xb[all_me[vl]] - mu2_f) / sc2_f) - mf_m) @ wf_m
                            me_fits.append((bi, *_fit_mb(None)))

                        ym = np.r_[np.ones(na_m), -np.ones(nb_m)]
                        pm_m, nm_m = ym > 0, ym < 0
                        sds_m = np.maximum(np.sqrt(0.5 * (np.var(oof_m[pm_m], axis=0) + np.var(oof_m[nm_m], axis=0))), 1e-6)
                        Xn_m = oof_m / sds_m[None, :]
                        ws_m = np.where(pm_m, 0.5 / np.sum(pm_m), 0.5 / np.sum(nm_m))
                        dps_m = np.mean(Xn_m[pm_m], axis=0) - np.mean(Xn_m[nm_m], axis=0)
                        Cm = len(me_bis)
                        res_m = minimize(
                            lambda th: (
                                float(ws_m @ np.logaddexp(0.0, -ym * (Xn_m @ th[:Cm] + th[Cm]))) + 0.5 * 0.45 * float(th[:Cm] @ th[:Cm]) + 0.12 * (th[Cm] ** 2),
                                np.r_[(-ym * expit(-ym * (Xn_m @ th[:Cm] + th[Cm])) * ws_m) @ Xn_m + 0.45 * th[:Cm], float(np.sum(-ym * expit(-ym * (Xn_m @ th[:Cm] + th[Cm])) * ws_m)) + 0.24 * th[Cm]],
                            ),
                            np.r_[np.maximum(dps_m * 0.20, 0.0), 0.0],
                            method="L-BFGS-B",
                            jac=True,
                            bounds=[(0.0, 3.5)] * Cm + [(-0.35, 0.35)],
                        )
                        cm_oof = Xn_m @ res_m.x[:Cm]
                        sd_m = max(math.sqrt(0.5 * (float(np.var(cm_oof[pm_m])) + float(np.var(cm_oof[nm_m])))), 1e-6)
                        dp_m = float(np.mean(cm_oof[pm_m]) - np.mean(cm_oof[nm_m])) / sd_m
                        if dp_m > 0.50:
                            for bi in range(n_B):
                                bias_val -= (1.0 - alpha_me) * float(mids_out[bi] @ ws_out[bi])
                                ws_out[bi] = (1.0 - alpha_me) * ws_out[bi]
                                mids_out[bi] = np.zeros_like(mids_out[bi])
                            s_et1 *= (1.0 - alpha_me)
                            s_et2 *= (1.0 - alpha_me)
                            cal_m = (dp_m / sd_m) if calibrate_oof_llr else (1.0 / sd_m)
                            for m_col, (bi, w2, m2, mu2, sc2) in enumerate(me_fits):
                                mu1, sc1, _, _ = block_scalers[bi]
                                c_me = alpha_me * (float(res_m.x[m_col]) / float(sds_m[m_col])) * cal_m
                                ws_out[bi] = ws_out[bi] + c_me * (sc1 / sc2) * w2
                                bias_val += c_me * float((((mu1 - mu2) / sc2) - m2) @ w2)

            wv_merged = np.where(m1_full, wv_full * s_et1, wv_full * s_et2)
            emid_s = s_et1 * em1_full + s_et2 * em2_full
            keep_et = np.abs(wv_merged) > 1.5e-4 if len(ut_full) else np.zeros(0, dtype=bool)
            wi_k = ut_full[keep_et] if len(ut_full) else np.zeros(0, dtype=np.int64)
            wv_k = wv_merged[keep_et] if len(ut_full) else np.zeros(0, dtype=np.float64)

            rel = (Mp / (Mp + 6.0)) if is_multi_panel_sib else (Mp / (Mp + 18.0))
            ws_p = [vp.pack_f32(w) for w in ws_out]
            mids_p = [vp.pack_f32(m) for m in mids_out]
            wv_p = vp.pack_f32(wv_k) if len(wv_k) else ""
            wv_d = vp.unpack_f64(wv_p) if len(wv_k) else np.zeros(0, dtype=np.float64)
            pairs_packed.append(
                {
                    "a": int(a_c),
                    "b": int(b_c),
                    "ws": ws_p,
                    "mids": mids_p,
                    "wi": vp.pack_i32(wi_k) if len(wi_k) else "",
                    "wv": wv_p,
                    "emid_s": float(emid_s),
                    "bias": float(bias_val),
                    "rel": float(rel),
                    "vmf_mid": 0.0,
                }
            )
            pairs_dec.append(
                {
                    "a": int(a_c),
                    "b": int(b_c),
                    "ws": [vp.unpack_f64(w) for w in ws_p],
                    "mids": [vp.unpack_f64(m) for m in mids_p],
                    "wi": wi_k,
                    "wv": wv_d,
                    "w_map": {int(ik): float(vk) for ik, vk in zip(wi_k.tolist(), wv_d.tolist())} if len(wi_k) else {},
                    "emid_s": float(emid_s),
                    "bias": float(bias_val),
                    "rel": float(rel),
                    "vmf_mid": 0.0,
                }
            )

    scalers_packed = []
    scalers_dec = []
    for (mu_b, sc_b, U_cot, dmps) in block_scalers:
        mu_bp = vp.pack_f32(mu_b)
        sc_bp = vp.pack_f32(sc_b)
        uc_bp = vp.pack_f32(U_cot.ravel())
        rc = int(len(dmps))
        scalers_packed.append(
            {
                "mu": mu_bp,
                "sc": sc_bp,
                "uc": uc_bp,
                "dmp": vp.pack_f64(dmps),
                "rc": rc,
            }
        )
        scalers_dec.append(
            (
                vp.unpack_f64(mu_bp),
                vp.unpack_f64(sc_bp),
                vp.unpack_f64(uc_bp, (rc, len(mu_b))) if rc > 0 else np.zeros((0, len(mu_b)), dtype=np.float64),
                dmps,
            )
        )
    return (
        {"scalers": scalers_packed, "pairs": pairs_packed},
        {"scalers": scalers_dec, "pairs": pairs_dec},
    )


def _make_stratified_prompt_folds(
    raw_tuples: Sequence[tuple[int, ...]],
    src_idx: np.ndarray,
    prompt_keys: Sequence[str],
    M: int,
    n_folds: int = 2,
) -> list[tuple[list[int], list[int]]]:
    rows_by_pk = defaultdict(list)
    for i, pk in enumerate(prompt_keys):
        rows_by_pk[pk].append(i)
    cls_totals = np.bincount(src_idx, minlength=M)
    sorted_pks = sorted(rows_by_pk, key=lambda pk: (int(np.min(cls_totals[src_idx[rows_by_pk[pk]]])), -len(rows_by_pk[pk]), pk))
    fold_cls = np.zeros((n_folds, M), dtype=np.int64)
    fold_tot = np.zeros(n_folds, dtype=np.int64)
    pk_fold = {}
    for pk in sorted_pks:
        c_part = np.unique(src_idx[rows_by_pk[pk]])
        best_f = min(range(n_folds), key=lambda f: (int(np.sum(fold_cls[f, c_part])), int(fold_tot[f]), f))
        pk_fold[pk] = best_f
        for i in rows_by_pk[pk]:
            fold_cls[best_f, src_idx[i]] += 1
            fold_tot[best_f] += 1

    folds = []
    for f in range(n_folds):
        val_ix = [i for i, pk in enumerate(prompt_keys) if pk_fold[pk] == f]
        val_seqs = {raw_tuples[i] for i in val_ix}
        sup_ix = [i for i, pk in enumerate(prompt_keys) if pk_fold[pk] != f and raw_tuples[i] not in val_seqs]
        if len(sup_ix) >= M and len(val_ix) >= M:
            folds.append((sup_ix, val_ix))
    return folds


def _fit_inner_contrastive_heads(
    raw_tuples: Sequence[tuple[int, ...]],
    orders_w_list: Sequence[list[np.ndarray]],
    src_idx: np.ndarray,
    M: int,
    prompt_keys: Sequence[str],
    efforts: Sequence[str],
    env_keys: Sequence[str],
    seq_lens: np.ndarray,
    Xcdf: np.ndarray,
    Xlex: np.ndarray,
    Xar: np.ndarray,
    X1497: np.ndarray,
    Xchunk: np.ndarray,
    Xdyn: np.ndarray,
    Xgeom: np.ndarray,
    Xfreq: np.ndarray,
    Xpl: np.ndarray,
    Xh: np.ndarray,
    F_taboo: np.ndarray,
    F_gram: np.ndarray,
    et_list: Sequence[tuple[np.ndarray, np.ndarray]],
    hazard: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, dict[str, Any]]:
    u_envs = sorted(set(env_keys))
    splits: list[tuple[list[int], list[int]]] = []
    if len(u_envs) >= 8:
        ev_fold = {ev: idx % 4 for idx, ev in enumerate(u_envs)}
        for f in range(4):
            val_ix = [i for i, ev in enumerate(env_keys) if ev_fold[ev] == f]
            val_seqs = {raw_tuples[i] for i in val_ix}
            sup_ix = [i for i, ev in enumerate(env_keys) if ev_fold[ev] != f and raw_tuples[i] not in val_seqs]
            if len(sup_ix) >= M and len(val_ix) >= M:
                splits.append((sup_ix, val_ix))
    if not splits:
        splits = _make_stratified_prompt_folds(raw_tuples, src_idx, prompt_keys, M, n_folds=2)

    default_th1 = np.array([1.1, 1.1, 0.7, 0.7, 0.6, 0.5, 0.4, 0.4, 0.8, 0.8, 1.1, 2.4, 0.9], dtype=np.float64)
    default_th3 = np.array([1.4, 1.4, 0.7, 0.6, 0.7, 0.6, 0.4, 0.4, 0.9, 0.9, 1.2, 2.8, 1.1], dtype=np.float64)
    if not splits or M < 2:
        return default_th1, default_th3, 1.06, 1.09, 1.12, {}

    pairs_global = np.asarray(list(itertools.combinations(range(M), 2)), dtype=np.int64)
    p_first, p_second = pairs_global[:, 0], pairs_global[:, 1]
    inc_idx = np.stack([np.flatnonzero((p_first == c) | (p_second == c)) for c in range(M)])
    inc_sgn = np.stack([np.where(p_first[inc_idx[c]] == c, 1.0, -1.0) for c in range(M)])
    stable_cols = list(vp.STABLE_CDF_COLS)
    sub_cols_l = list(vp.SUBLIMINAL_COLS)
    mac_cols_l = list(vp.MACRO_COLS)

    oof_k1, oof_k3_nat, oof_k3_aug = [], [], []
    oof_F1_full, oof_b4_1, oof_y1_full = [], [], []
    oof_F2_nat_full, oof_b4_2, oof_y2_nat_full = [], [], []
    oof_F3_nat_full, oof_b4_3, oof_y3_nat_full = [], [], []
    sp_cache: dict[int, tuple[dict, dict]] = {}

    for f_idx, (sup_f, val_f) in enumerate(splits):
        y_s = src_idx[sup_f]
        s_cnt = np.bincount(y_s, minlength=M)
        present_set = set(np.flatnonzero(s_cnt > 0).tolist())
        if len(present_set) < max(2, M // 2):
            continue
        pks_s = [prompt_keys[i] for i in sup_f]
        efs_s = [efforts[i] for i in sup_f]
        evs_s = [env_keys[i] for i in sup_f]

        _, pv1 = fit_portable_pair_view(Xcdf[sup_f, :195], y_s, M, pks_s, efs_s, cot_pen=12.0, studentize=False, cov_cap_per_class=64, max_rank=96)
        _, pv2 = fit_portable_pair_view(np.c_[Xlex[sup_f], Xcdf[sup_f, 96:195], Xfreq[sup_f]], y_s, M, pks_s, efs_s, cot_pen=16.0, studentize=False, cov_cap_per_class=64, max_rank=48)
        _, pv3 = fit_portable_pair_view(Xar[sup_f], y_s, M, pks_s, efs_s, cot_pen=14.0, studentize=True, cov_cap_per_class=64, max_rank=96)
        _, pv4 = fit_portable_pair_view(np.c_[X1497[sup_f], Xchunk[sup_f]], y_s, M, pks_s, efs_s, cot_pen=12.0, studentize=True, cov_cap_per_class=64, max_rank=48)
        _, pv5 = fit_portable_pair_view(np.c_[Xcdf[sup_f], Xdyn[sup_f], Xgeom[sup_f], Xpl[sup_f]], y_s, M, pks_s, efs_s, cot_pen=12.0, studentize=True, cov_cap_per_class=64, max_rank=96)

        rv_pl = fit_pl_rank_view([orders_w_list[i][0] for i in sup_f], y_s, M, hazard_init=hazard, maxiter=40)
        _, kr_mm = fit_portable_krr(np.c_[Xar[sup_f, :1088], X1497[sup_f], Xcdf[sup_f], Xgeom[sup_f]], y_s, M, pks_s, efs_s)

        _, vAR = fit_portable_vmf_channel(Xar[sup_f], y_s, M, pks_s, efs_s, sp_cache, cot_pen=14.0, env_keys=evs_s)
        _, vA = fit_portable_vmf_channel(Xcdf[sup_f][:, stable_cols], y_s, M, pks_s, efs_s, sp_cache, cot_pen=8.0, env_keys=evs_s)
        _, vB = fit_portable_vmf_channel(np.c_[Xdyn[sup_f], Xgeom[sup_f]], y_s, M, pks_s, efs_s, sp_cache, cot_pen=12.0, env_keys=evs_s)
        _, vC = fit_portable_vmf_channel(np.c_[Xcdf[sup_f], Xpl[sup_f], Xh[sup_f]], y_s, M, pks_s, efs_s, sp_cache, cot_pen=16.0, env_keys=evs_s)
        _, vL = fit_portable_vmf_channel(Xlex[sup_f], y_s, M, pks_s, efs_s, sp_cache, cot_pen=16.0, env_keys=evs_s)
        _, vM = fit_portable_vmf_channel(np.c_[X1497[sup_f], Xchunk[sup_f, 355:]], y_s, M, pks_s, efs_s, sp_cache, cot_pen=12.0, env_keys=evs_s)
        vmf_decs_f = (vAR, vA, vB, vC, vL, vM)

        _, ref_et = fit_portable_clean_refiner(Xcdf[sup_f], Xgeom[sup_f], [et_list[i] for i in sup_f], y_s, M, pks_s, efs_s, vC["M_unit_base"])
        _, sb4_dec = fit_portable_block4_sibling_expert(
            [Xcdf[sup_f, :96], Xcdf[sup_f][:, stable_cols], X1497[sup_f, 355:], np.c_[X1497[sup_f], Xfreq[sup_f]], F_taboo[sup_f], F_gram[sup_f]],
            [et_list[i] for i in sup_f],
            y_s,
            M,
            pks_s,
            efs_s,
            vC["M_unit_base"],
            env_keys=evs_s,
            lam_cs=(0.08, 0.10, 0.08, 0.08, 0.08, 0.40),
            gamma_ef=0.0,
            mid_nl_cap=0.0,
            alpha_me=0.0,
            calibrate_oof_llr=True,
        )

        V1_v = Xcdf[val_f, :195]
        V2_v = np.c_[Xlex[val_f], Xcdf[val_f, 96:195], Xfreq[val_f]]
        V3_v = Xar[val_f]
        V4_v = np.c_[X1497[val_f], Xchunk[val_f]]
        V5_v = np.c_[Xcdf[val_f], Xdyn[val_f], Xgeom[val_f], Xpl[val_f]]
        Xkr_v = np.c_[Xar[val_f, :1088], X1497[val_f], Xcdf[val_f], Xgeom[val_f]]
        Xcdf_v = Xcdf[val_f]
        Xgeom_v = Xgeom[val_f]
        vmf_val_inputs = (
            V3_v,
            Xcdf_v[:, stable_cols],
            np.c_[Xdyn[val_f], Xgeom_v],
            np.c_[Xcdf_v, Xpl[val_f], Xh[val_f]],
            Xlex[val_f],
            np.c_[X1497[val_f], Xchunk[val_f, 355:]],
        )

        pl_cls_rows, pl_pop_rows = [], []
        for gi in val_f:
            cs_g, ps_g = zip(*(vp.score_pl_rank_one_portable(rv_pl["coefs"], rv_pl["shared"], o) for o in orders_w_list[gi]))
            pl_cls_rows.append(np.mean(cs_g, axis=0))
            pl_pop_rows.append(float(np.mean(ps_g)))
        pl_cls_v = np.stack(pl_cls_rows)
        pl_pop_v = np.asarray(pl_pop_rows, dtype=np.float64)

        val_valid_loc = [loc_i for loc_i, gi in enumerate(val_f) if int(src_idx[gi]) in present_set]
        if not val_valid_loc:
            continue

        by_cls_all = defaultdict(list)
        by_cls_short = defaultdict(list)
        for loc_i in val_valid_loc:
            gi = val_f[loc_i]
            c_i = int(src_idx[gi])
            by_cls_all[c_i].append(loc_i)
            if seq_lens[gi] < 600:
                by_cls_short[c_i].append(loc_i)

        use_short_cls = {c_i for c_i, locs in by_cls_short.items() if len(locs) >= 3}
        val_eval_loc = [
            loc_i for loc_i in val_valid_loc
            if (int(src_idx[val_f[loc_i]]) not in use_short_cls) or (seq_lens[val_f[loc_i]] < 600)
        ]

        pkt1 = [[loc_i] for loc_i in val_eval_loc]
        by_cls_v = defaultdict(list)
        by_env_grp_v = defaultdict(list)
        by_ef_grp_v = defaultdict(list)
        for loc_i in val_eval_loc:
            gi = val_f[loc_i]
            c_i = int(src_idx[gi])
            by_cls_v[c_i].append(loc_i)
            prov_cli = prompt_keys[gi].split("::")[-1]
            by_env_grp_v[(c_i, env_keys[gi], prov_cli, efforts[gi])].append(loc_i)
            by_ef_grp_v[(c_i, efforts[gi])].append(loc_i)

        nat2_specs = []
        nat3_specs = []
        for (c_i, _, _, _), locs in sorted(by_env_grp_v.items()):
            locs_s = sorted(locs, key=lambda l_i: (prompt_keys[val_f[l_i]], l_i))
            for st_i in range(0, len(locs_s) - 1, 2):
                nat2_specs.append((c_i, locs_s[st_i : st_i + 2]))
            for st_i in range(0, len(locs_s) - 2, 3):
                nat3_specs.append((c_i, locs_s[st_i : st_i + 3]))

        if len(nat2_specs) < M:
            nat2_specs = []
            for (c_i, _), locs in sorted(by_ef_grp_v.items()):
                for st_i in range(0, len(locs) - 1, 2):
                    nat2_specs.append((c_i, locs[st_i : st_i + 2]))
        if len(nat3_specs) < M:
            nat3_specs = []
            for (c_i, _), locs in sorted(by_ef_grp_v.items()):
                for st_i in range(0, len(locs) - 2, 3):
                    nat3_specs.append((c_i, locs[st_i : st_i + 3]))

        nat3_cls_set = {c_i for c_i, _ in nat3_specs}
        aug3_specs = []
        if len(nat3_cls_set) < len(by_cls_v):
            seed_txt = f"v6_port_oof|{f_idx}|{len(sup_f)}|{len(val_f)}"
            seed_int = int.from_bytes(hashlib.sha256(seed_txt.encode("utf-8")).digest()[:8], "big")
            rng = np.random.Generator(np.random.PCG64(seed_int))
            for c_i, pool in sorted(by_cls_v.items()):
                if c_i not in nat3_cls_set:
                    for _ in range(4):
                        sel = rng.choice(pool, size=3, replace=len(pool) < 3).tolist()
                        aug3_specs.append((c_i, sel))

        def _pre_pv_rows(pv: dict[str, Any], X_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            Z = (X_rows - pv["mu0"][None, :]) / pv["sc"][None, :]
            if pv["damp_c"] > 0.0:
                uc = pv["u_cot"]
                Z = Z - pv["damp_c"] * ((Z @ uc)[:, None] * uc[None, :])
            t1 = vp._cauchy_radial((Z @ pv["Q1"].T) @ pv["W1"].T + pv["b1"][None, :], pv["r1"])
            t3 = vp._cauchy_radial((Z @ pv["Q3"].T) @ pv["W3"].T + pv["b3"][None, :], pv["r3"])
            return t1, t3

        pv_rows = [
            _pre_pv_rows(pv1, V1_v),
            _pre_pv_rows(pv2, V2_v),
            _pre_pv_rows(pv3, V3_v),
            _pre_pv_rows(pv4, V4_v),
            _pre_pv_rows(pv5, V5_v),
        ]

        Zq_kr = ((Xkr_v - kr_mm["mu"][None, :]) / kr_mm["sd"][None, :]) @ kr_mm["P_krr"].T
        Dq_kr = np.maximum(np.sum(Zq_kr * Zq_kr, axis=1, keepdims=True) + np.sum(kr_mm["Z_lm"] * kr_mm["Z_lm"], axis=1)[None, :] - 2.0 * (Zq_kr @ kr_mm["Z_lm"].T), 0.0)
        Kq_rbf = np.exp(-Dq_kr / kr_mm["bw"])
        Kq_lap = np.exp(-np.sqrt(Dq_kr / kr_mm["bw"]))
        vr_rbf = np.maximum(1.0 - np.sum(Kq_rbf * (Kq_rbf @ kr_mm["H_rbf"]), axis=1), 0.0)
        vr_lap = np.maximum(1.0 - np.sum(Kq_lap * (Kq_lap @ kr_mm["H_lap"]), axis=1), 0.0)
        kr_row_sup = 0.45 * ((Kq_rbf @ kr_mm["alpha_rbf"]) / np.sqrt(vr_rbf[:, None] + 0.08)) + 0.55 * ((Kq_lap @ kr_mm["alpha_lap"]) / np.sqrt(vr_lap[:, None] + 0.08))

        vmf_pre = []
        for vm_ch, X_v_in in zip(vmf_decs_f, vmf_val_inputs):
            sp_v = vm_ch["spline"]
            Mu_v = vm_ch["M_unit_base"]
            kap_c_v = vm_ch["kappa_c"]
            s_kc_v = np.sort(kap_c_v)
            med_kc_v = float(s_kc_v[M // 2]) if (M % 2 == 1) else 0.5 * float(s_kc_v[M // 2 - 1] + s_kc_v[M // 2])
            kap_u0_v = np.minimum(kap_c_v, med_kc_v * 1.25)
            Rb_v = Mu_v * vm_ch["rho_c"][:, None]
            pb0_v = vp.eval_psi_spline(sp_v, vm_ch["rho_c"])
            Rsb_v = Mu_v * vm_ch["rho_sib"][:, None]
            ps0_v = vp.eval_psi_spline(sp_v, vm_ch["rho_sib"])
            ln_c_v = -vp.eval_psi_spline(sp_v, kap_c_v)
            ln_u0_v = -vp.eval_psi_spline(sp_v, kap_u0_v)
            Z_vm = (X_v_in - vm_ch["mu0"][None, :]) / vm_ch["sc"][None, :]
            Zw_vm = Z_vm @ vm_ch["P_proj"].T - vm_ch["offset_sub"][None, :]
            U_vm = Zw_vm / np.maximum(np.linalg.norm(Zw_vm, axis=1, keepdims=True), 1e-12)
            vmf_pre.append((sp_v, kap_c_v, kap_u0_v, Rb_v, pb0_v, Rsb_v, ps0_v, ln_c_v, ln_u0_v, U_vm))

        b4_covered = {(pm["a"], pm["b"]) for pm in sb4_dec["pairs"]}
        Zcdf_ref_v = (Xcdf_v - ref_et["mu_c"][None, :]) / ref_et["sc_c"][None, :]
        Zgeom_ref_v = (Xgeom_v - ref_et["mu_g"][None, :]) / ref_et["sc_g"][None, :]
        n_ref_p = len(ref_et["pairs"])
        row_et_mat = np.zeros((len(val_f), n_ref_p), dtype=np.float64)
        for pi_r, pm in enumerate(ref_et["pairs"]):
            if (pm["a"], pm["b"]) in b4_covered:
                continue
            w_map = pm.get("w_map")
            if w_map:
                for r_loc, gi in enumerate(val_f):
                    ci_arr, cv_arr = et_list[gi]
                    s_et = 0.0
                    for ik, vk in zip(ci_arr.tolist(), cv_arr.tolist()):
                        w_val = w_map.get(ik)
                        if w_val is not None:
                            s_et += vk * w_val
                    row_et_mat[r_loc, pi_r] = s_et

        n_b4_p = len(sb4_dec["pairs"])
        row_b4_tot1 = np.zeros((len(val_f), n_b4_p), dtype=np.float64)
        if n_b4_p > 0:
            raw_b4_v = (
                Xcdf_v[:, :96],
                Xcdf_v[:, stable_cols],
                X1497[val_f, 355:],
                np.c_[X1497[val_f], Xfreq[val_f]],
                F_taboo[val_f],
                F_gram[val_f],
            )
            Z_b4_v = []
            for X_bv, (mu_b, sc_b, uc_mat, dmps) in zip(raw_b4_v, sb4_dec["scalers"]):
                Zb_v = (X_bv - mu_b[None, :]) / sc_b[None, :]
                if len(dmps) > 0:
                    Zb_v = Zb_v - ((Zb_v @ uc_mat.T) * dmps[None, :]) @ uc_mat
                Z_b4_v.append(Zb_v)
            for pi_b, pm in enumerate(sb4_dec["pairs"]):
                t1_vec = np.full(len(val_f), float(pm["bias"]) - float(pm["emid_s"]), dtype=np.float64)
                w_map = pm.get("w_map")
                if w_map:
                    for r_loc, gi in enumerate(val_f):
                        ci_arr, cv_arr = et_list[gi]
                        s_et = 0.0
                        for ik, vk in zip(ci_arr.tolist(), cv_arr.tolist()):
                            w_val = w_map.get(ik)
                            if w_val is not None:
                                s_et += vk * w_val
                        t1_vec[r_loc] += s_et
                else:
                    t1_vec += float(pm["emid_s"])
                for bi in range(len(Z_b4_v)):
                    t1_vec += (Z_b4_v[bi] - pm["mids"][bi][None, :]) @ pm["ws"][bi]
                row_b4_tot1[:, pi_b] = t1_vec

        def _agg_pv(t1_r: np.ndarray, t3_r: np.ndarray, ix: list[int]):
            K_l = len(ix)
            if K_l == 1:
                tb = 0.72 * t1_r[ix[0]] + 0.28 * t3_r[ix[0]]
                return tb, np.tanh(tb)
            if K_l == 2:
                tb = 0.50 * t1_r[ix] + 0.50 * t3_r[ix]
                m_out = np.mean(tb, axis=0)
                th = np.tanh(tb)
                c_out = np.mean(th, axis=0) * np.maximum(0.25, 1.0 - 0.60 * np.var(th, axis=0))
                return m_out, c_out
            tb = 0.30 * t1_r[ix] + 0.70 * t3_r[ix]
            s_tb = np.sort(tb, axis=0)
            m_out = 0.65 * np.mean(tb, axis=0) + 0.35 * s_tb[K_l // 2]
            th = np.tanh(tb)
            c_out = np.mean(th, axis=0) * np.maximum(0.25, 1.0 - 0.85 * np.var(th, axis=0))
            return m_out, c_out

        def _eval_pkts(pkt_locs: list[list[int]]) -> tuple[np.ndarray, list[list[tuple[int, int, float, float]]]]:
            out_rows = []
            out_b4 = []
            for ix in pkt_locs:
                K_l = len(ix)
                m1, c1 = _agg_pv(*pv_rows[0], ix)
                m2, c2 = _agg_pv(*pv_rows[1], ix)
                m3, c3_p = _agg_pv(*pv_rows[2], ix)
                m4, c4 = _agg_pv(*pv_rows[3], ix)
                m5, c5 = _agg_pv(*pv_rows[4], ix)
                pl_g = np.sum(pl_cls_v[ix], axis=0) - float(np.sum(pl_pop_v[ix]))
                m_pl = pl_g[p_first] - pl_g[p_second]

                kr_s = np.mean(kr_row_sup[ix], axis=0)
                s_kr = np.sort(kr_s)
                med_kr = float(s_kr[M // 2]) if (M % 2 == 1) else 0.5 * float(s_kr[M // 2 - 1] + s_kr[M // 2])
                kr_z = (kr_s - med_kr) / max(float(np.std(kr_s)), 0.12)
                m_kr = kr_z[p_first] - kr_z[p_second]

                coh_tot = np.zeros(M, dtype=np.float64)
                for w_v, (sp_v, kap_c_v, kap_u0_v, Rb_v, pb0_v, Rsb_v, ps0_v, ln_c_v, ln_u0_v, U_vm) in zip(vp.VMF_WEIGHTS_6, vmf_pre):
                    Su = np.sum(U_vm[ix], axis=0)
                    su_norm = float(np.linalg.norm(Su))
                    Uq_c = kap_c_v[:, None] * Su[None, :]
                    llc = float(K_l) * ln_c_v + (vp.eval_psi_spline(sp_v, np.linalg.norm(Rb_v + Uq_c, axis=1)) - pb0_v)
                    lls = float(K_l) * ln_c_v + (vp.eval_psi_spline(sp_v, np.linalg.norm(Rsb_v + Uq_c, axis=1)) - ps0_v)
                    llu0 = float(K_l) * ln_u0_v + vp.eval_psi_spline(sp_v, su_norm * kap_u0_v)
                    a_u = math.log(vp.U_UNIFORM_WEIGHT / M) + vp.logsumexp_1d(llu0)
                    b_u = math.log(vp.U_SIBLING_WEIGHT / M) + vp.logsumexp_1d(lls)
                    mx_u = max(a_u, b_u)
                    llu = mx_u + math.log(math.exp(a_u - mx_u) + math.exp(b_u - mx_u))
                    coh_tot += w_v * (llc - llu)

                r_vmf0 = vp.GAMMA_COND_VMF * coh_tot
                r_vmf = r_vmf0.copy()
                if n_ref_p:
                    q0 = np.exp(r_vmf0 - vp.logsumexp_1d(r_vmf0))
                    adj = np.zeros(M, dtype=np.float64)
                    zc = np.mean(Zcdf_ref_v[ix], axis=0)
                    zg = np.mean(Zgeom_ref_v[ix], axis=0)
                    z_sub = zc[sub_cols_l]
                    z_mg = np.r_[zc[mac_cols_l], zg]
                    et_means = np.mean(row_et_mat[ix], axis=0)
                    for pi_r, pm in enumerate(ref_et["pairs"]):
                        a_p, b_p = pm["a"], pm["b"]
                        if (a_p, b_p) in b4_covered or q0[a_p] + q0[b_p] < 0.01:
                            continue
                        et_sc = float(et_means[pi_r]) - pm["emid"]
                        s1 = 0.55 * float(np.dot(z_sub - pm["m1"], pm["w1"])) + 0.45 * et_sc
                        s2 = float(np.dot(z_mg - pm["m2"], pm["w2"]))
                        q2 = np.array([s1, s2], dtype=np.float64)
                        vk = pm["var2"] / float(K_l)
                        lla = -0.5 * float(np.sum(((q2 - pm["mu_a"]) ** 2) / vk))
                        llb = -0.5 * float(np.sum(((q2 - pm["mu_b"]) ** 2) / vk))
                        lam_ab = float(np.clip(vp.ALPHA_SIBLING * pm["rel"] * (lla - llb), -vp.SIBLING_CLIP, vp.SIBLING_CLIP))
                        adj[a_p] += q0[b_p] * lam_ab
                        adj[b_p] -= q0[a_p] * lam_ab
                    r_vmf = r_vmf0 + adj

                b4_pkt = []
                if n_b4_p > 0:
                    k_gain = float(K_l) / (1.0 + float(K_l - 1) * 0.12)
                    t1_means = np.mean(row_b4_tot1[ix], axis=0)
                    for pi_b, pm in enumerate(sb4_dec["pairs"]):
                        llr_k = float(np.clip(pm["rel"] * k_gain * float(t1_means[pi_b]), -9.5, 9.5))
                        b4_pkt.append((int(pm["a"]), int(pm["b"]), float(pm["rel"]), llr_k))

                m_vmf = r_vmf[p_first] - r_vmf[p_second]
                out_rows.append(np.stack([m1, c1, m2, c2, m3, c3_p, m4, c4, m5, c5, m_pl, m_vmf, m_kr], axis=1))
                out_b4.append(b4_pkt)
            return np.stack(out_rows), out_b4

        F1, b4_1 = _eval_pkts(pkt1)
        oof_F1_full.append(F1)
        oof_b4_1.extend(b4_1)
        for qi, loc_i in enumerate(val_eval_loc):
            c_t = int(src_idx[val_f[loc_i]])
            oof_y1_full.append(c_t)
            oof_k1.append((c_t, F1[qi, inc_idx[c_t]] * inc_sgn[c_t, :, None]))

        if nat2_specs:
            F2_nat, b4_2 = _eval_pkts([ix for _, ix in nat2_specs])
            oof_F2_nat_full.append(F2_nat)
            oof_b4_2.extend(b4_2)
            for c_t, _ in nat2_specs:
                oof_y2_nat_full.append(c_t)

        all3_locs = [ix for _, ix in nat3_specs] + [ix for _, ix in aug3_specs]
        if all3_locs:
            F3_all, b4_3_all = _eval_pkts(all3_locs)
            n_nat = len(nat3_specs)
            if n_nat > 0:
                oof_F3_nat_full.append(F3_all[:n_nat])
                oof_b4_3.extend(b4_3_all[:n_nat])
            for qi, (c_t, _) in enumerate(nat3_specs):
                oof_y3_nat_full.append(c_t)
                oof_k3_nat.append((c_t, F3_all[qi, inc_idx[c_t]] * inc_sgn[c_t, :, None]))
            for qi, (c_t, _) in enumerate(aug3_specs):
                oof_k3_aug.append((c_t, F3_all[n_nat + qi, inc_idx[c_t]] * inc_sgn[c_t, :, None]))

    def _solve_nonneg(nat_list, aug_list, l2_val, fallback):
        if len(nat_list) < M and not aug_list:
            return fallback
        cnt_nat = Counter(c for c, _ in nat_list)
        cnt_aug = Counter(c for c, _ in aug_list) if aug_list else Counter()
        all_cls = set(cnt_nat) | set(cnt_aug)
        if len(all_cls) < 2:
            return fallback
        n_cls = float(len(all_cls))
        des_list, wt_list = [], []
        for c, feat in nat_list:
            om = 0.5 if cnt_aug[c] > 0 else 1.0
            wt_list.append(om / (n_cls * cnt_nat[c]))
            des_list.append(feat)
        for c, feat in (aug_list or []):
            om = 0.5 if cnt_nat[c] > 0 else 1.0
            wt_list.append(om / (n_cls * cnt_aug[c]))
            des_list.append(feat)
        wts = np.asarray(wt_list, dtype=np.float64)
        oriented = np.stack(des_list)
        flat = oriented.reshape(-1, 13)
        pw = np.repeat(wts / max(M - 1, 1), M - 1)
        scale = np.maximum(np.sqrt(pw @ (flat * flat)), 1e-6)
        des = oriented / scale[None, None, :]
        n_des, m_des, _ = des.shape
        flat_des = des.reshape(n_des * m_des, 13)

        def _obj(th: np.ndarray):
            logits = (flat_des @ th).reshape(n_des, m_des)
            losses = logsumexp(np.c_[np.zeros(n_des), -logits], axis=1)
            probs = np.exp(-logits - losses[:, None])
            val = float(wts @ losses) + 0.5 * l2_val * float(th @ th)
            grad = -((wts[:, None] * probs).ravel() @ flat_des) + l2_val * th
            return val, grad

        res = minimize(_obj, np.full(13, 0.2, dtype=np.float64), method="L-BFGS-B", jac=True, bounds=[(0.0, None)] * 13, options=dict(maxiter=200))
        if not np.isfinite(res.x).all() or float(np.sum(res.x)) < 1e-3:
            return fallback
        return res.x / scale

    th1 = _solve_nonneg(oof_k1, None, 0.012, default_th1)
    th3 = _solve_nonneg(oof_k3_nat, oof_k3_aug, 0.009, default_th3)
    th2 = 0.45 * th1 + 0.55 * th3

    def _fit_oof_tau_and_acc(F_blocks, b4_list, y_list, th_vec, default_tau):
        if not F_blocks or len(y_list) < M:
            return default_tau, None, 0, 0, None
        F_all = np.concatenate(F_blocks, axis=0)
        th64 = th_vec.astype(np.float64)
        rc_rows = []
        for qi in range(len(y_list)):
            F_pkt = F_all[qi]
            pl = F_pkt @ th64
            Lm = np.zeros((M, M), dtype=np.float64)
            Lm[p_first, p_second] = pl
            Lm[p_second, p_first] = -pl
            for a_p, b_p, rel_p, llr_p in b4_list[qi]:
                if rel_p >= 0.70:
                    if a_p < b_p:
                        p_idx = a_p * M - (a_p * (a_p + 1)) // 2 + (b_p - a_p - 1)
                        l_deb = float((F_pkt[p_idx] * vp.DEBIASED_SIB_MASK) @ th64)
                    else:
                        p_idx = b_p * M - (b_p * (b_p + 1)) // 2 + (a_p - b_p - 1)
                        l_deb = -float((F_pkt[p_idx] * vp.DEBIASED_SIB_MASK) @ th64)
                    lo_ab = 0.25 * l_deb + 1.20 * llr_p
                    Lm[a_p, b_p] = lo_ab
                    Lm[b_p, a_p] = -lo_ab
                else:
                    Lm[a_p, b_p] += llr_p
                    Lm[b_p, a_p] -= llr_p
            rc_rows.append(-vp.logsumexp_axis1(-Lm))
        rc = np.stack(rc_rows)
        ya = np.asarray(y_list, dtype=np.int64)
        cnt_y = Counter(ya.tolist())
        wy = np.asarray([1.0 / (len(cnt_y) * cnt_y[int(c)]) for c in ya], dtype=np.float64)
        row_r = np.arange(len(ya))

        def _obj_tau(tau: float) -> float:
            sc = tau * rc
            return float(wy @ (logsumexp(sc, axis=1) - sc[row_r, ya])) + 0.004 * (math.log(tau) ** 2)

        res_t = minimize_scalar(_obj_tau, bounds=(0.85, 3.00), method="bounded")
        tau_opt = float(res_t.x) if np.isfinite(res_t.x) else default_tau
        sc_opt = tau_opt * rc
        pure_nll = float(wy @ (logsumexp(sc_opt, axis=1) - sc_opt[row_r, ya]))
        preds = np.argmax(rc, axis=1)
        correct = int(np.sum(preds == ya))
        return tau_opt, correct / len(ya), correct, len(ya), pure_nll

    tau_1, acc1, ok1, tot1, nll1 = _fit_oof_tau_and_acc(oof_F1_full, oof_b4_1, oof_y1_full, th1, 1.35)
    tau_3, acc3, ok3, tot3, nll3 = _fit_oof_tau_and_acc(oof_F3_nat_full, oof_b4_3, oof_y3_nat_full, th3, 1.85)
    tau_2_fb = 0.5 * (tau_1 + tau_3)
    tau_2, acc2, ok2, tot2, nll2 = _fit_oof_tau_and_acc(oof_F2_nat_full, oof_b4_2, oof_y2_nat_full, th2, tau_2_fb)
    cal_meta = {
        "1": {"beta": round(tau_1, 4), "cv_accuracy": round(acc1, 4) if acc1 is not None else None, "cv_correct": ok1, "cv_samples": tot1, "cv_nll": round(nll1, 4) if nll1 is not None else None, "fallback": acc1 is None},
        "2": {"beta": round(tau_2, 4), "cv_accuracy": round(acc2, 4) if acc2 is not None else None, "cv_correct": ok2, "cv_samples": tot2, "cv_nll": round(nll2, 4) if nll2 is not None else None, "fallback": acc2 is None},
        "3": {"beta": round(tau_3, 4), "cv_accuracy": round(acc3, 4) if acc3 is not None else None, "cv_correct": ok3, "cv_samples": tot3, "cv_nll": round(nll3, 4) if nll3 is not None else None, "fallback": acc3 is None},
    }
    return th1, th3, tau_1, tau_2, tau_3, cal_meta


def resolve_row_prompt_key(row: Mapping[str, Any]) -> str:
    pk = row.get("prompt_key") or row.get("prompt_bundle_sha256")
    if pk:
        base = str(pk)
    else:
        cid = str(row.get("challenge_id") or "")
        if cid in vp.CANONICAL_SUITE_PROMPT_KEYS:
            base = vp.CANONICAL_SUITE_PROMPT_KEYS[cid]
        else:
            cond = str(row.get("condition_id") or row.get("environment") or "manual")
            base = f"{cond}::{cid}"
    prov = str(row.get("provider") or "default").strip()
    cli_ver = str(row.get("cli_version") or "").strip()
    return f"{base}::{prov}_{cli_ver}" if cli_ver else f"{base}::{prov}"


def resolve_row_effort(row: Mapping[str, Any]) -> str:
    ef = str(row.get("reasoning_effort") or row.get("effort") or "unknown").strip().lower()
    return ef if ef in VALID_EFFORTS else "unknown"


def fit_v6_portable_bank(rows: Sequence[Mapping[str, Any]], model_ids: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fits full Absolute-Dominance v6 on `rows` and returns (v6_portable_json_dict, calibration_dict)."""
    present_sources = [str(m) for m in model_ids]
    M = len(present_sources)
    if M < 2:
        raise ValueError("Absolute-Dominance v6 requires at least 2 candidate models in a bank")
    s_map = {s: i for i, s in enumerate(present_sources)}

    items = []
    for row in rows:
        src = str(row["source"])
        if src not in s_map:
            continue
        nums = tuple(int(v) for v in row["numbers"] if vp.VALUE_MIN <= int(v) <= vp.VALUE_MAX)
        if len(nums) < 20:
            continue
        pk = resolve_row_prompt_key(row)
        ev = str(row.get("condition_id") or row.get("environment") or pk)
        items.append((nums, s_map[src], pk, resolve_row_effort(row), ev))

    if not items:
        raise ValueError("No valid reference sequences to fit v6 bank")
    items.sort(key=lambda it: (it[0], it[2], it[3]))

    raw_tuples = [it[0] for it in items]
    src_idx = np.asarray([it[1] for it in items], dtype=np.int64)
    prompt_keys = [it[2] for it in items]
    efforts = [it[3] for it in items]
    env_keys = [it[4] for it in items]

    pure_list = [get_cached_pure(t) for t in raw_tuples]
    arrs = [p[0] for p in pure_list]
    seq_lens = np.asarray([len(a) for a in arrs], dtype=np.int64)
    orders = [p[1] for p in pure_list]
    Xh = np.stack([p[2] for p in pure_list])
    Xpl = np.stack([p[3] for p in pure_list])
    Xdyn = np.stack([p[4] for p in pure_list])
    et_list = [p[5] for p in pure_list]
    Xfreq = np.stack([p[6] for p in pure_list])
    Xgeom = np.stack([p[7] for p in pure_list])
    marg99_list = [p[8] for p in pure_list]
    X1497 = np.stack([p[9] for p in pure_list])
    Xchunk = np.stack([p[10] for p in pure_list])

    seq_pair_list = [get_cached_seq(t, a) for t, a in zip(raw_tuples, arrs)]
    F_taboo = np.stack([sp[0] for sp in seq_pair_list])
    F_gram = np.stack([sp[1] for sp in seq_pair_list])

    s_cnt = np.bincount(src_idx, minlength=M).astype(np.float64)
    if np.any(s_cnt < 1):
        missing = [present_sources[c] for c in range(M) if s_cnt[c] < 1]
        raise ValueError(f"Candidates missing valid sequences: {missing}")

    bal_weights = (1.0 / s_cnt[src_idx]) / float(M)
    hazard = fit_pooled_hazard(orders, bal_weights)
    ar_b0, ar_a0, ar_M_base = fit_pooled_ar_law(arrs, src_idx, M)

    Xar = np.stack([vp.extract_ar_fisher_from_state(get_cached_ar_state(a), ar_b0, ar_a0, ar_M_base) for a in arrs])
    win_triples = [vp.extract_windowed_cdf_lex_orders(a, o, m9, hazard) for a, o, m9 in zip(arrs, orders, marg99_list)]
    Xcdf = np.stack([wt[0] for wt in win_triples])
    Xlex = np.stack([wt[1] for wt in win_triples])
    orders_w_list = [wt[2] for wt in win_triples]

    th1, th3, tau_1, tau_2, tau_3, cal_meta = _fit_inner_contrastive_heads(
        raw_tuples, orders_w_list, src_idx, M, prompt_keys, efforts, env_keys, seq_lens,
        Xcdf, Xlex, Xar, X1497, Xchunk, Xdyn, Xgeom, Xfreq, Xpl, Xh, F_taboo, F_gram, et_list, hazard,
    )

    pv1_p, _ = fit_portable_pair_view(Xcdf[:, :195], src_idx, M, prompt_keys, efforts, cot_pen=12.0, studentize=False, cov_cap_per_class=64, max_rank=96)
    pv2_p, _ = fit_portable_pair_view(np.c_[Xlex, Xcdf[:, 96:195], Xfreq], src_idx, M, prompt_keys, efforts, cot_pen=16.0, studentize=False, cov_cap_per_class=64, max_rank=48)
    pv3_p, _ = fit_portable_pair_view(Xar, src_idx, M, prompt_keys, efforts, cot_pen=14.0, studentize=True, cov_cap_per_class=64, max_rank=96)
    pv4_p, _ = fit_portable_pair_view(np.c_[X1497, Xchunk], src_idx, M, prompt_keys, efforts, cot_pen=12.0, studentize=True, cov_cap_per_class=64, max_rank=48)
    pv5_p, _ = fit_portable_pair_view(np.c_[Xcdf, Xdyn, Xgeom, Xpl], src_idx, M, prompt_keys, efforts, cot_pen=12.0, studentize=True, cov_cap_per_class=64, max_rank=96)

    rv_pl = fit_pl_rank_view([ow[0] for ow in orders_w_list], src_idx, M, hazard_init=hazard)
    kr_p, _ = fit_portable_krr(np.c_[Xar[:, :1088], X1497, Xcdf, Xgeom], src_idx, M, prompt_keys, efforts)

    sp_cache: dict[int, tuple[dict, dict]] = {}
    stable_cols = list(vp.STABLE_CDF_COLS)
    vAR_p, vAR_dec = fit_portable_vmf_channel(Xar, src_idx, M, prompt_keys, efforts, sp_cache, cot_pen=14.0, env_keys=env_keys)
    vA_p, vA_dec = fit_portable_vmf_channel(Xcdf[:, stable_cols], src_idx, M, prompt_keys, efforts, sp_cache, cot_pen=8.0, env_keys=env_keys)
    vB_p, vB_dec = fit_portable_vmf_channel(np.c_[Xdyn, Xgeom], src_idx, M, prompt_keys, efforts, sp_cache, cot_pen=12.0, env_keys=env_keys)
    vC_p, vC_dec = fit_portable_vmf_channel(np.c_[Xcdf, Xpl, Xh], src_idx, M, prompt_keys, efforts, sp_cache, cot_pen=16.0, env_keys=env_keys)
    vL_p, vL_dec = fit_portable_vmf_channel(Xlex, src_idx, M, prompt_keys, efforts, sp_cache, cot_pen=16.0, env_keys=env_keys)
    vM_p, vM_dec = fit_portable_vmf_channel(np.c_[X1497, Xchunk[:, 355:]], src_idx, M, prompt_keys, efforts, sp_cache, cot_pen=12.0, env_keys=env_keys)

    ref_p, _ = fit_portable_clean_refiner(Xcdf, Xgeom, et_list, src_idx, M, prompt_keys, efforts, vC_dec["M_unit_base"])
    sb4_p, _ = fit_portable_block4_sibling_expert(
        [Xcdf[:, :96], Xcdf[:, stable_cols], X1497[:, 355:], np.c_[X1497, Xfreq], F_taboo, F_gram],
        et_list,
        src_idx,
        M,
        prompt_keys,
        efforts,
        vC_dec["M_unit_base"],
        env_keys=env_keys,
        lam_cs=(0.08, 0.10, 0.08, 0.08, 0.08, 0.40),
        gamma_ef=0.0,
        mid_nl_cap=0.0,
        alpha_me=0.0,
        calibrate_oof_llr=True,
    )

    portable_dict = {
        "schema": "absolute_dominance_v6.portable.1",
        "M": int(M),
        "present_sources": present_sources,
        "hazard": vp.pack_f64(hazard),
        "ar_b0": vp.pack_f64(ar_b0),
        "ar_a0": vp.pack_f64(ar_a0),
        "rv_coefs": vp.pack_f32(rv_pl["coefs"].ravel()),
        "rv_shared": vp.pack_f64(rv_pl["shared"]),
        "psi_splines": {str(r_dim): sp_p for r_dim, (sp_p, _) in sp_cache.items()},
        "pvs": [pv1_p, pv2_p, pv3_p, pv4_p, pv5_p],
        "vmfs": [vAR_p, vA_p, vB_p, vC_p, vL_p, vM_p],
        "kr_mm": kr_p,
        "ref_et": ref_p,
        "sref_b4": sb4_p,
        "theta_1": vp.pack_f64(th1),
        "theta_3": vp.pack_f64(th3),
        "tau_1": float(tau_1),
        "tau_2": float(tau_2),
        "tau_3": float(tau_3),
    }
    return portable_dict, cal_meta
