"""Portable Absolute-Dominance v6 feature extraction, tensor codec, and inference engine.

Uses pure IEEE-754 float64 NumPy arithmetic with zero SciPy runtime dependency so that
Python (`fingerprint.py`) and Browser/Node ES-module JavaScript (`static/fingerprint-core.js`)
agree to < 1e-12 across all 56+ candidate models and the open-set Unknown source U.
"""
from __future__ import annotations

import base64
import itertools
import math
from typing import Any, Mapping, Sequence

import numpy as np

VALUE_MIN = 1
VALUE_MAX = 355
DIMENSION = 355
ET_ORIG_DIM = 148745
ET_EXT_DIM = 178622

ACCEPT_TAU_ABS = 0.46
ACCEPT_DOM_RIVAL = 1.65
ACCEPT_DOM_U = 1.40

OPERATING_POINTS: dict[str, dict[str, Any]] = {
    "relaxed": {
        "id": "relaxed",
        "label": "宽松召回档",
        "tau_abs": 0.40,
        "dom_rival": 1.80,
        "dom_u": 1.20,
    },
    "default": {
        "id": "default",
        "label": "默认平衡档",
        "tau_abs": 0.46,
        "dom_rival": 1.65,
        "dom_u": 1.40,
    },
    "strict": {
        "id": "strict",
        "label": "严格高精档",
        "tau_abs": 0.60,
        "dom_rival": 3.50,
        "dom_u": 2.00,
    },
}

GAMMA_COND_VMF = 0.38
VMF_WEIGHTS_6 = (0.22, 0.15, 0.15, 0.15, 0.16, 0.17)
U_UNIFORM_WEIGHT = 0.20
U_SIBLING_WEIGHT = 0.80
ALPHA_SIBLING = 0.65
SIBLING_CLIP = 6.0
DEBIASED_SIB_MASK = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float64)

STABLE_CDF_COLS: tuple[int, ...] = tuple(list(range(0, 64)) + list(range(86, 154)) + list(range(195, 225)))
SUBLIMINAL_COLS: tuple[int, ...] = tuple(
    list(range(0, 12))
    + list(range(22, 44))
    + list(range(54, 64))
    + list(range(86, 96))
    + list(range(96, 104))
    + list(range(113, 125))
    + list(range(125, 133))
    + list(range(142, 154))
    + list(range(195, 225))
)
MACRO_COLS: tuple[int, ...] = tuple(c for c in range(225) if c not in set(SUBLIMINAL_COLS))

CANONICAL_SUITE_PROMPT_KEYS: dict[str, str] = {
    "query-01": "9193a39b7a2d6dcc918fd94a7ec5532e7c71b0e3ab699ff27e112c2107e27eca",
    "query-02": "c433b4f9776239bc5d3902aa81974d6e918fd855d6f0675753a33c541160136c",
    "query-03": "5a7c96dfefadae622dafeadecd9691cdfc7c1f5e2bc90fc95ee2639714718dfd",
    "query-04": "6c6b8e99caef7a504a0d73c3e776a37c67a331d55bd7a6426ffe6b9f825d122f",
    "query-05": "97da2b4d4800fde156a88164d4ca569af1ae0a891430a0ca6306694f98cc7b41",
    "query-06": "505fe8562ddb73375b9e6bfc565918be9ae4ad6b11bb22ff32f7fc27fdf1e4ff",
    "query-07": "c0324aa3b5e85df01203e0140fdc75bb6e52d23a54495aef604b64e5ea9b0b1b",
    "query-08": "ab04bb52f281cfb2c3642f287dc4a2decf73aa91bac15f2fde989b92067253b8",
    "query-09": "1060c361c623c14d93fed6db02b1b2485be169dbfce7e4466b64c61dd75be17e",
    "query-10": "3ed070b1a6b51a3dd8c9e27f4f4d9a1b8e401a398e4859b751d9f68a3a2801f2",
    "query-11": "bdac571f19170f6ef64bfbaecc18ca99e7a9b34b3d699d605f9126c4c1681094",
    "query-12": "ea9bcbb4e5dd2b632b3d7a9aab337e2e83397165056fa6f505888f2f8aca2973",
    "query-13": "7ee6edcd8c2b7a35de95fb437ce1742178b3f440203c22dffdfb00e9eab88704",
    "query-14": "373e57037c0e0cd10c6c12c81fed29d39f04e271f25a75f5b8fc0e5d4b9ec476",
    "query-15": "eb2d2bd627bbd2e36c6a1d44ff226267b65e6bb23cd31559057f721e7e1605ed",
    "query-16": "4f945fc4c430f5931fca691081321a8c33ca7e7a2366c8ff0d6c9799cceca1ae",
    "query-17": "b2b41d952afa6553070c55afc74465a802e41b315de22ae0801eb7cc1832df20",
    "query-18": "f4fb8e2678a9f10a63d385bb38ed9673ef8c6ac19857e76e4864a7f172520bb6",
    "query-19": "4ca4e34aede6e6dca2f5dd305fc88707b398aab3a7b14867f7c015d3f3df9bde",
    "query-20": "530db425362ee28befd3f62c3f7dd62b7063c33cf92de6114b26f8c288289fce",
    "query-21": "582476227819e2f9b05db4a649b835542884357f7fe76f931a9aa11dd1709713",
    "query-22": "b9853c65d42cb2bfd0ef8ccb760284dbf98f1bb1e6c2afb8bb9c3ca64cac2707",
    "query-23": "7a17ed126de4b8ed1f5e827d41ff0dee4baf8c7a69956faa5ad5c2aa7f6009dd",
    "query-24": "509e646ae4fda54777ce0b04e45635365d9e12b6a620e092d994f0d6e11b6a2d",
    "query-25": "91eb1f6957e1595eb029d8329cd653bcb2e237b9f148cda37326c1d0e2f0fff9",
    "query-26": "e59bf41d5103d128ab2ea756237c910c794eca2767bf014033ab2fda54532b41",
    "query-27": "8a8167e82623f9f55a908d9fb38e2443140711948c3ff522c380abbce6f74c59",
    "query-28": "ea002726929ee4ca4ca1d8d4525126f49dfcc2b8991fed3f26d149af0b86dab2",
    "query-29": "3173b8424a02fcc77d1d2d6b64a67b8f2cb828bf48db703ae205760cfa6b8968",
    "query-30": "533d41ee76d5b6c56b1e39119b058e5fa6618d1fd1cc68436072aa08e59f51ed",
    "query-31": "a9bb6b157f4abb56a33ba3508d11f520cd5276828aececcff9c6f6b0f3dafd0a",
    "query-32": "1ea4a0919a5ca6177e32887f9f47200a258a98aa30ea07fa9b1bb08d8f652b09",
    "query-33": "27f6fba81e6e53f00e937f68ca66ad0dc4ee7c6d695bb2b17057afca8d2b2728",
    "query-34": "a8b5b17836a1facdde5e9554be860cea3975de6afdd3fe602a0fce7a295e4d90",
    "query-35": "d06818552560385730b4c4258a224d28e7c0f8e730c34eb21edc81ef047cb4b8",
    "query-36": "ba13e7f54f59bdb13d0d3f8af868724634c77742ee9b7c4ca3c9c043836301d5",
}


# =============================================================================
# 1. Binary Tensor Codec (Lossless Float64 & Row-Scaled Int16 -> Exact Float64)
# =============================================================================


def pack_f64(arr: Sequence[float] | np.ndarray) -> str:
    a = np.asarray(arr, dtype="<f8")
    return base64.b64encode(a.tobytes()).decode("ascii")


def pack_f32(arr: Sequence[float] | np.ndarray) -> dict[str, str]:
    a = np.asarray(arr, dtype="<f4")
    return {"f32": base64.b64encode(a.tobytes()).decode("ascii")}


def unpack_f64(val: str | Mapping[str, Any], shape: tuple[int, ...] | None = None) -> np.ndarray:
    if isinstance(val, Mapping) and "f32" in val:
        raw = base64.b64decode(str(val["f32"]).encode("ascii"))
        out = np.frombuffer(raw, dtype="<f4").astype(np.float64, copy=True)
    else:
        raw = base64.b64decode(str(val).encode("ascii"))
        out = np.frombuffer(raw, dtype="<f8").astype(np.float64, copy=True)
    return out.reshape(shape) if shape is not None else out


def pack_i32(arr: Sequence[int] | np.ndarray) -> str:
    a = np.asarray(arr, dtype="<i4")
    return base64.b64encode(a.tobytes()).decode("ascii")


def unpack_i32(b64: str) -> np.ndarray:
    raw = base64.b64decode(b64.encode("ascii"))
    return np.frombuffer(raw, dtype="<i4").astype(np.int64, copy=True)


def pack_mat_i8(mat: np.ndarray) -> dict[str, Any]:
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("pack_mat_i8 expects a 2D array")
    rows, cols = m.shape
    if rows == 0 or cols == 0:
        return {"shape": [rows, cols], "bits": 8, "scales": "", "data": ""}
    max_abs = np.max(np.abs(m), axis=1)
    scales_f32 = np.where(max_abs > 0.0, max_abs / 127.0, 0.0).astype("<f4")
    scales_f64 = scales_f32.astype(np.float64)
    safe_s = np.where(scales_f64 > 0.0, scales_f64, 1.0)
    q = np.where(
        scales_f64[:, None] > 0.0,
        np.rint(np.clip(m / safe_s[:, None], -127.0, 127.0)),
        0.0,
    ).astype("i1")
    return {
        "shape": [int(rows), int(cols)],
        "bits": 8,
        "scales": base64.b64encode(scales_f32.tobytes()).decode("ascii"),
        "data": base64.b64encode(q.tobytes()).decode("ascii"),
    }


def pack_mat_i16(mat: np.ndarray) -> dict[str, Any]:
    m = np.asarray(mat, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError("pack_mat_i16 expects a 2D array")
    rows, cols = m.shape
    if rows == 0 or cols == 0:
        return {"shape": [rows, cols], "scales": "", "data": ""}
    max_abs = np.max(np.abs(m), axis=1)
    scales_f32 = np.where(max_abs > 0.0, max_abs / 32767.0, 0.0).astype("<f4")
    scales_f64 = scales_f32.astype(np.float64)
    safe_s = np.where(scales_f64 > 0.0, scales_f64, 1.0)
    q = np.where(
        scales_f64[:, None] > 0.0,
        np.rint(np.clip(m / safe_s[:, None], -32767.0, 32767.0)),
        0.0,
    ).astype("<i2")
    return {
        "shape": [int(rows), int(cols)],
        "scales": base64.b64encode(scales_f32.tobytes()).decode("ascii"),
        "data": base64.b64encode(q.tobytes()).decode("ascii"),
    }


def unpack_mat_i16(obj: Mapping[str, Any]) -> np.ndarray:
    rows, cols = int(obj["shape"][0]), int(obj["shape"][1])
    if rows == 0 or cols == 0:
        return np.zeros((rows, cols), dtype=np.float64)
    scales = np.frombuffer(base64.b64decode(obj["scales"].encode("ascii")), dtype="<f4").astype(np.float64)
    raw_q = base64.b64decode(obj["data"].encode("ascii"))
    q_dtype = "i1" if int(obj.get("bits", 16)) == 8 else "<i2"
    q = np.frombuffer(raw_q, dtype=q_dtype).astype(np.float64).reshape(rows, cols)
    return q * scales[:, None]


# =============================================================================
# 2. Deterministic Basis & Static Lookup Tables (100% Identical to JS)
# =============================================================================


def leg_poly_matrix(x: np.ndarray, max_deg: int) -> np.ndarray:
    """Returns [len(x), max_deg] with columns P_1(x) .. P_{max_deg}(x) via Bonnet recurrence."""
    xv = np.asarray(x, dtype=np.float64)
    n = len(xv)
    out = np.empty((n, max_deg), dtype=np.float64)
    if max_deg >= 1:
        out[:, 0] = xv
    p_km1 = np.ones(n, dtype=np.float64)
    p_k = xv.copy()
    for k in range(1, max_deg):
        p_kp1 = ((2.0 * k + 1.0) * xv * p_k - float(k) * p_km1) / float(k + 1)
        out[:, k] = p_kp1
        p_km1 = p_k
        p_k = p_kp1
    return out


def _digit_fourier_basis(digits: np.ndarray) -> np.ndarray:
    df = digits.astype(np.float64)
    cols = [np.cos(2.0 * math.pi * k * df / 10.0) for k in range(1, 6)]
    cols.extend(np.sin(2.0 * math.pi * k * df / 10.0) for k in range(1, 5))
    return np.stack(cols, axis=1)


def _build_static_tables():
    vals = np.arange(1, 356, dtype=np.int64)
    coord = 2.0 * (vals.astype(np.float64) - 0.5) / 355.0 - 1.0
    x_grid = 2.0 * (vals.astype(np.float64) - 1.0) / 354.0 - 1.0
    hund = vals // 100
    ten = (vals // 10) % 10
    unit = vals % 10

    res_basis = np.concatenate(
        [
            leg_poly_matrix(coord, 12),
            np.eye(10, dtype=np.float64)[unit],
            np.eye(10, dtype=np.float64)[ten],
        ],
        axis=1,
    )
    marg_basis = np.concatenate(
        [
            leg_poly_matrix(x_grid, 8),
            _digit_fourier_basis(unit),
            _digit_fourier_basis(ten),
            leg_poly_matrix(2.0 * hund.astype(np.float64) / 3.0 - 1.0, 3),
        ],
        axis=1,
    )

    idx = np.arange(355, dtype=np.int64)
    perms = [
        idx,
        np.argsort((vals % 10) * 1000 + idx, kind="mergesort"),
        np.argsort(((vals // 10) % 10) * 1000 + idx, kind="mergesort"),
    ]
    inv_perms = [np.argsort(p) for p in perms]

    jump_order = np.empty((355, 355), dtype=np.int16)
    inv_jump = np.empty((355, 355), dtype=np.int16)
    for pv in range(355):
        jo = np.argsort(np.abs(idx - pv) * 1000 + idx, kind="mergesort")
        jump_order[pv] = jo.astype(np.int16)
        inv_jump[pv, jo] = idx.astype(np.int16)

    u = idx[:, None]
    v = idx[None, :]
    d = np.abs(v - u)
    B = np.zeros((355, 355, 11), dtype=np.float64)
    B[:, :, 0] = (d == 0).astype(np.float64)
    B[:, :, 1] = (d == 1).astype(np.float64)
    B[:, :, 2] = (d == 2).astype(np.float64)
    B[:, :, 3] = ((d >= 3) & (d <= 5)).astype(np.float64)
    B[:, :, 4] = ((d >= 6) & (d <= 12)).astype(np.float64)
    B[:, :, 5] = ((d >= 13) & (d <= 30)).astype(np.float64)
    B[:, :, 6] = d.astype(np.float64) / 355.0
    B[:, :, 7] = (v > u).astype(np.float64)
    B[:, :, 8] = ((v % 10) == (u % 10)).astype(np.float64)
    B[:, :, 9] = ((v % 10) == ((u + 1) % 10)).astype(np.float64)
    B[:, :, 10] = ((v // 100) == (u // 100)).astype(np.float64)

    dist_mat = d.astype(np.float64) / 355.0
    dist_sq_mat = dist_mat * dist_mat
    mod10_eye = np.eye(10, dtype=np.float64)[idx % 10]
    div100_eye = np.eye(4, dtype=np.float64)[idx // 100]

    lag_lut = np.zeros(52, dtype=np.int8)
    lag_lut[2] = 1
    lag_lut[3] = 2
    lag_lut[4:7] = 3
    lag_lut[7:13] = 4
    lag_lut[13:25] = 5
    lag_lut[25:51] = 6

    h_cum = np.cumsum(1.0 / np.arange(355, 0, -1, dtype=np.float64))
    perms3 = list(itertools.permutations(range(3)))
    perms4 = list(itertools.permutations(range(4)))
    return (
        res_basis,
        marg_basis,
        perms,
        inv_perms,
        jump_order,
        inv_jump,
        B,
        dist_mat,
        dist_sq_mat,
        mod10_eye,
        div100_eye,
        lag_lut,
        h_cum,
        perms3,
        perms4,
    )


(
    _RES_BASIS,
    _MARG_BASIS,
    _CDF_PERMS,
    _INV_CDF_PERMS,
    _JUMP_ORDER_TABLE,
    _INV_JUMP_ORDER_TABLE,
    _MARKOV_B,
    _DIST_MAT,
    _DIST_SQ_MAT,
    _MOD10_EYE,
    _DIV100_EYE,
    _LAG_BUCKET_LUT,
    _H_CUM,
    _PERMS3,
    _PERMS4,
) = _build_static_tables()

_RES_SCALES = (16.0, 64.0, 1e10)
_LAG_BINS_EDGE = (1, 2, 4, 8, 16, 32, 64, 128, 256, 400)


def _build_static_seq_luts():
    vals = np.arange(1, DIMENSION + 1, dtype=np.int64)
    is_prime = np.ones(DIMENSION, dtype=np.float64)
    is_prime[0] = 0.0
    for v in range(2, DIMENSION + 1):
        r = int(math.isqrt(v))
        for d in range(2, r + 1):
            if v % d == 0:
                is_prime[v - 1] = 0.0
                break

    hund = (vals // 100).astype(np.int32)
    ten = ((vals // 10) % 10).astype(np.int32)
    unit = (vals % 10).astype(np.int32)
    dlen = (vals >= 10).astype(np.int32) + (vals >= 100).astype(np.int32)
    dsum = (hund + ten + unit).astype(np.int32)

    is_repdigit = np.zeros(DIMENSION, dtype=np.float64)
    is_palin = np.zeros(DIMENSION, dtype=np.float64)
    is_asc = np.zeros(DIMENSION, dtype=np.float64)
    is_desc = np.zeros(DIMENSION, dtype=np.float64)
    for idx, v in enumerate(vals.tolist()):
        s = str(v)
        if len(s) >= 2 and len(set(s)) == 1:
            is_repdigit[idx] = 1.0
        if len(s) >= 2 and s == s[::-1]:
            is_palin[idx] = 1.0
        if len(s) >= 2 and all(s[k] < s[k + 1] for k in range(len(s) - 1)):
            is_asc[idx] = 1.0
        if len(s) >= 2 and all(s[k] > s[k + 1] for k in range(len(s) - 1)):
            is_desc[idx] = 1.0

    step_edges = np.array(
        [-200, -135, -90, -60, -40, -26, -17, -11, -7, -4, -3, -2, -1, 0,
          1, 2, 3, 5, 8, 12, 18, 27, 41, 61, 91, 136, 201],
        dtype=np.int64,
    )
    deltas = np.arange(-354, 355, dtype=np.int64)
    step_lut = np.searchsorted(step_edges, deltas, side="right").astype(np.int32)
    return is_prime, hund, ten, unit, dlen, dsum, is_repdigit, is_palin, is_asc, is_desc, step_lut


(
    _IS_PRIME,
    _HUND,
    _TEN,
    _UNIT,
    _DLEN,
    _DSUM,
    _IS_REPDIGIT,
    _IS_PALIN,
    _IS_ASC,
    _IS_DESC,
    _STEP_LUT,
) = _build_static_seq_luts()

_U355 = np.arange(DIMENSION, dtype=np.int32)
_PROX_LO = [np.maximum(_U355 - rad, 0) for rad in (1, 2, 5, 12, 25)]
_PROX_HI = [np.minimum(_U355 + rad + 1, DIMENSION) for rad in (1, 2, 5, 12, 25)]
_RADS_3D = np.array([1, 2, 5, 12, 25], dtype=np.int32)[:, None, None]


def _hellinger(counts: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    c = counts.astype(np.float64) + alpha
    return np.sqrt(c * (1.0 / float(c.sum())))


def _indep_res2(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> np.ndarray:
    n = len(a)
    if n < 1:
        return np.zeros(na * nb, dtype=np.float64)
    inv_n = 1.0 / float(n)
    pa = (np.bincount(a, minlength=na).astype(np.float64) + 0.5) * (1.0 / float(n + 0.5 * na))
    pb = (np.bincount(b, minlength=nb).astype(np.float64) + 0.5) * (1.0 / float(n + 0.5 * nb))
    obs = np.bincount(a * nb + b, minlength=na * nb).astype(np.float64) * inv_n
    exp = np.outer(pa, pb).ravel()
    return (obs - exp) / np.sqrt(exp * inv_n + 1e-3)


def extract_taboo_fast(arr: np.ndarray) -> np.ndarray:
    """323-dim Exchangeable-Centered Working-Memory Taboo / Anti-Collision Decay Curves."""
    L = len(arr)
    a32 = arr.astype(np.int32, copy=False)
    z0 = a32 - 1
    cnt355 = np.bincount(z0, minlength=DIMENSION).astype(np.float64)
    cs355 = np.empty(DIMENSION + 1, dtype=np.float64)
    cs355[0] = 0.0
    np.cumsum(cnt355, out=cs355[1:])
    denom_pairs = max(float(L * (L - 1)), 1.0)

    b32 = (z0 * 32) // DIMENSION
    b16 = (z0 * 16) // DIMENSION
    b8 = (z0 * 8) // DIMENSION
    un = _UNIT[z0]
    te = _TEN[z0]
    hu = _HUND[z0]
    dl = _DLEN[z0]
    m2 = a32 & 1
    m5 = a32 % 5
    m8 = a32 & 7

    P_mat = np.stack((z0, b32, b16, b8, un, te, hu, dl, m2, m5, m8), axis=0).astype(np.int16)
    e0_disc = np.array(
        [
            float(np.dot(cnt355, cnt355 - 1.0)) / denom_pairs,
            float(np.sum((c := np.bincount(b32, minlength=32).astype(np.float64)) * (c - 1.0))) / denom_pairs,
            float(np.sum((c := np.bincount(b16, minlength=16).astype(np.float64)) * (c - 1.0))) / denom_pairs,
            float(np.sum((c := np.bincount(b8, minlength=8).astype(np.float64)) * (c - 1.0))) / denom_pairs,
            float(np.sum((c := np.bincount(un, minlength=10).astype(np.float64)) * (c - 1.0))) / denom_pairs,
            float(np.sum((c := np.bincount(te, minlength=10).astype(np.float64)) * (c - 1.0))) / denom_pairs,
            float(np.sum((c := np.bincount(hu, minlength=4).astype(np.float64)) * (c - 1.0))) / denom_pairs,
            float(np.sum((c := np.bincount(dl, minlength=3).astype(np.float64)) * (c - 1.0))) / denom_pairs,
            float(np.sum((c := np.bincount(m2, minlength=2).astype(np.float64)) * (c - 1.0))) / denom_pairs,
            float(np.sum((c := np.bincount(m5, minlength=5).astype(np.float64)) * (c - 1.0))) / denom_pairs,
            float(np.sum((c := np.bincount(m8, minlength=8).astype(np.float64)) * (c - 1.0))) / denom_pairs,
        ],
        dtype=np.float64,
    )
    vb_disc = np.maximum(e0_disc * (1.0 - e0_disc), 1e-4)
    e0_prox = np.array(
        [max((float(np.dot(cnt355, cs355[_PROX_HI[ri]] - cs355[_PROX_LO[ri]])) - float(L)) / denom_pairs, 0.0) for ri in range(5)],
        dtype=np.float64,
    )
    vb_prox = np.maximum(e0_prox * (1.0 - e0_prox), 1e-4)

    max_k = min(36, L - 1)
    disc_out = np.zeros((11, 18), dtype=np.float64)
    prox_out = np.zeros((5, 18), dtype=np.float64)
    win_out = np.zeros(35, dtype=np.float64)
    if max_k >= 1:
        P_lpad = np.full((11, L + max_k), -1, dtype=np.int16)
        P_lpad[:, max_k:] = P_mat
        Pw_back = np.lib.stride_tricks.sliding_window_view(P_lpad, max_k + 1, axis=1)[:, :L, ::-1]
        eq_back = (Pw_back[:, :, 0:1] == Pw_back[:, :, 1:])
        cnt_d_all = np.count_nonzero(eq_back, axis=1).astype(np.float64)

        a_lpad = np.full(L + max_k, -10000, dtype=np.int32)
        a_lpad[max_k:] = a32
        aw_back = np.lib.stride_tricks.sliding_window_view(a_lpad, max_k + 1)[:L, ::-1]
        ad_back = np.abs(aw_back[:, 0:1] - aw_back[:, 1:])
        prox_bool = (ad_back[None, :, :] <= _RADS_3D)
        cnt_p_all = np.count_nonzero(prox_bool, axis=1).astype(np.float64)

        k16 = min(16, max_k)
        nks = (L - np.arange(1, k16 + 1, dtype=np.float64))[None, :]
        inv_nks = 1.0 / nks
        disc_out[:, :k16] = (cnt_d_all[:, :k16] * inv_nks - e0_disc[:, None]) / np.sqrt(vb_disc[:, None] * inv_nks + 2e-4)
        prox_out[:, :k16] = (cnt_p_all[:, :k16] * inv_nks - e0_prox[:, None]) / np.sqrt(vb_prox[:, None] * inv_nks + 2e-4)

        if max_k >= 17:
            k_hi1 = min(24, max_k)
            inv_b1 = 1.0 / float(np.sum(L - np.arange(17, k_hi1 + 1)))
            disc_out[:, 16] = (np.sum(cnt_d_all[:, 16:k_hi1], axis=1) * inv_b1 - e0_disc) / np.sqrt(vb_disc * inv_b1 + 2e-4)
            prox_out[:, 16] = (np.sum(cnt_p_all[:, 16:k_hi1], axis=1) * inv_b1 - e0_prox) / np.sqrt(vb_prox * inv_b1 + 2e-4)
        if max_k >= 25:
            inv_b2 = 1.0 / float(np.sum(L - np.arange(25, max_k + 1)))
            disc_out[:, 17] = (np.sum(cnt_d_all[:, 24:max_k], axis=1) * inv_b2 - e0_disc) / np.sqrt(vb_disc * inv_b2 + 2e-4)
            prox_out[:, 17] = (np.sum(cnt_p_all[:, 24:max_k], axis=1) * inv_b2 - e0_prox) / np.sqrt(vb_prox * inv_b2 + 2e-4)

        max_w = min(20, max_k)
        if max_w >= 3:
            W7 = np.stack(
                (
                    eq_back[0, :, :max_w],
                    prox_bool[1, :, :max_w],
                    prox_bool[2, :, :max_w],
                    prox_bool[3, :, :max_w],
                    eq_back[4, :, :max_w],
                    eq_back[5, :, :max_w],
                    eq_back[2, :, :max_w],
                ),
                axis=0,
            )
            cum_any = np.maximum.accumulate(W7, axis=2)
            base_es = np.array(
                [e0_disc[0], e0_prox[1], e0_prox[2], e0_prox[3], e0_disc[4], e0_disc[5], e0_disc[2]],
                dtype=np.float64,
            )
            one_minus_e = 1.0 - np.clip(base_es, 0.0, 0.999)
            pos_w = 0
            for W in (3, 5, 8, 12, 20):
                if L > W:
                    nw = float(L - W)
                    obs_w = np.count_nonzero(cum_any[:, W:, W - 1], axis=1).astype(np.float64) / nw
                    ew = 1.0 - (one_minus_e ** W)
                    vw = np.maximum(ew * (1.0 - ew), 1e-4)
                    win_out[pos_w : pos_w + 7] = (obs_w - ew) / np.sqrt(vw / nw + 5e-4)
                pos_w += 7

    return np.concatenate((disc_out.ravel(), prox_out.ravel(), win_out))


def extract_seq_grammar(arr: np.ndarray) -> np.ndarray:
    """1401-dim Marginal-Debiased Multi-Lag Transition Lift & Tokenization/Digit Grammar (`F_gram`)."""
    L = len(arr)
    a32 = arr.astype(np.int32, copy=False)
    z0 = a32 - 1
    b16 = (z0 * 16) // DIMENSION
    un = _UNIT[z0]
    te = _TEN[z0]
    hu = _HUND[z0]
    dl = _DLEN[z0]
    ds = _DSUM[z0] - 1

    parts: list[np.ndarray] = []
    parts.append(_indep_res2(b16[:-1], b16[1:], 16, 16) if L > 1 else np.zeros(256, dtype=np.float64))
    parts.append(_indep_res2(b16[:-2], b16[2:], 16, 16) if L > 2 else np.zeros(256, dtype=np.float64))
    parts.append(_indep_res2(un[:-1], un[1:], 10, 10) if L > 1 else np.zeros(100, dtype=np.float64))
    parts.append(_indep_res2(un[:-2], un[2:], 10, 10) if L > 2 else np.zeros(100, dtype=np.float64))

    c_dl = np.bincount(dl, minlength=3)
    if L >= 3:
        parts.append(_indep_res2(dl[:-1], dl[1:], 3, 3))
        p_dl = (c_dl.astype(np.float64) + 0.5) * (1.0 / float(L + 1.5))
        e_dl3 = (p_dl[:, None, None] * p_dl[None, :, None] * p_dl[None, None, :]).ravel()
        dl3 = (dl[:-2] * 3 + dl[1:-1]) * 3 + dl[2:]
        cnt_dl3 = np.bincount(dl3, minlength=27)
        inv_l2 = 1.0 / float(L - 2)
        parts.append((cnt_dl3.astype(np.float64) * inv_l2 - e_dl3) / np.sqrt(e_dl3 * inv_l2 + 1e-3))

        parts.append(_indep_res2(hu[:-1], hu[1:], 4, 4))
        parts.append(_indep_res2(hu[:-2], hu[2:], 4, 4))
        p_hu = (np.bincount(hu, minlength=4).astype(np.float64) + 0.5) * (1.0 / float(L + 2.0))
        e_hu3 = (p_hu[:, None, None] * p_hu[None, :, None] * p_hu[None, None, :]).ravel()
        hu3 = (hu[:-2] * 4 + hu[1:-1]) * 4 + hu[2:]
        cnt_hu3 = np.bincount(hu3, minlength=64)
        parts.append((cnt_hu3.astype(np.float64) * inv_l2 - e_hu3) / np.sqrt(e_hu3 * inv_l2 + 1e-3))
    else:
        cnt_dl3 = np.zeros(27, dtype=np.int64)
        cnt_hu3 = np.zeros(64, dtype=np.int64)
        parts.append(np.zeros(132, dtype=np.float64))

    ge10 = a32 >= 10
    te_ge = te[ge10]
    un_ge = un[ge10]
    n_ge10 = len(te_ge)
    parts.append(_indep_res2(te_ge, un_ge, 10, 10) if n_ge10 >= 5 else np.zeros(100, dtype=np.float64))

    m5 = a32 % 5
    if L >= 3:
        parts.append(_indep_res2(m5[:-1], m5[1:], 5, 5))
        parts.append(_hellinger(np.bincount(ds, minlength=20), 0.15))
    else:
        parts.append(np.zeros(45, dtype=np.float64))

    m2_i = a32 & 1
    is_m5_b = (m5 == 0)
    A9 = np.stack(
        (
            m2_i.astype(np.float64),
            is_m5_b.astype(np.float64),
            (un == 0).astype(np.float64),
            ((m2_i == 1) & (~is_m5_b)).astype(np.float64),
            _IS_PRIME[z0],
            _IS_REPDIGIT[z0],
            _IS_PALIN[z0],
            _IS_ASC[z0],
            _IS_DESC[z0],
        ),
        axis=0,
    )
    mvs = A9.mean(axis=1)
    if L >= 2:
        c1s = np.mean(A9[:, :-1] * A9[:, 1:], axis=1) - mvs * mvs
        acs = c1s / np.sqrt(np.maximum(mvs * (1.0 - mvs), 1e-3) / float(L - 1) + 1e-3)
    else:
        acs = np.zeros(9, dtype=np.float64)
    lex18 = np.empty(18, dtype=np.float64)
    lex18[0::2] = mvs
    lex18[1::2] = acs

    u_odd, t_odd, h_odd = un & 1, te & 1, hu & 1
    ge100 = a32 >= 100
    all_odd = np.where(ge100, u_odd & t_odd & h_odd, np.where(ge10, u_odd & t_odd, u_odd))
    all_even = np.where(ge100, (1 - u_odd) & (1 - t_odd) & (1 - h_odd), np.where(ge10, (1 - u_odd) & (1 - t_odd), 1 - u_odd))
    inv_L = 1.0 / float(max(L, 1))
    lex_tail = np.array(
        [
            float(np.count_nonzero(all_odd)) * inv_L,
            float(np.count_nonzero(all_even)) * inv_L,
            float(np.count_nonzero(un == te)) * inv_L if L > 0 else 0.0,
            float(np.count_nonzero(ge100 & ((hu == te) | (te == un) | (hu == un)))) * inv_L,
            float(np.count_nonzero((a32 % 25) == 0)) * inv_L,
            float(np.count_nonzero((a32 % 12) == 0)) * inv_L,
        ],
        dtype=np.float64,
    )
    parts.extend((lex18, lex_tail))

    parts.append(_hellinger(c_dl, 0.25))
    if L >= 4:
        dl2 = dl[:-1] * 3 + dl[1:]
        dl4 = ((dl[:-3] * 3 + dl[1:-2]) * 3 + dl[2:-1]) * 3 + dl[3:]
        parts.extend((_hellinger(np.bincount(dl2, minlength=9), 0.20), _hellinger(cnt_dl3, 0.12), _hellinger(np.bincount(dl4, minlength=81), 0.06)))
    else:
        parts.append(np.zeros(117, dtype=np.float64))

    if L >= 3:
        parts.extend((
            _hellinger(np.bincount(hu[:-1] * 4 + hu[1:], minlength=16), 0.15),
            _hellinger(np.bincount(hu[:-2] * 4 + hu[2:], minlength=16), 0.15),
            _hellinger(cnt_hu3, 0.08),
        ))
    else:
        parts.append(np.zeros(96, dtype=np.float64))

    parts.append(_hellinger(np.bincount(te_ge * 10 + un_ge, minlength=100), 0.08) if n_ge10 >= 5 else np.full(100, 0.1, dtype=np.float64))

    if L >= 4:
        m2_3 = (m2_i[:-2] * 2 + m2_i[1:-1]) * 2 + m2_i[2:]
        m2_4 = ((m2_i[:-3] * 2 + m2_i[1:-2]) * 2 + m2_i[2:-1]) * 2 + m2_i[3:]
        ds_diff = np.clip(ds[1:] - ds[:-1] + 12, 0, 23)
        parts.extend((
            _hellinger(np.bincount(m2_3, minlength=8), 0.20),
            _hellinger(np.bincount(m2_4, minlength=16), 0.15),
            _indep_res2(a32[:-1] % 3, a32[1:] % 3, 3, 3),
            _indep_res2(a32[:-1] & 3, a32[1:] & 3, 4, 4)[:15],
            _hellinger(np.bincount(ds_diff, minlength=24), 0.15),
        ))
    else:
        parts.append(np.zeros(72, dtype=np.float64))

    return np.concatenate(parts)


def extract_sparse_seq_events(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extended sparse sequential event extractor `[0 .. 178622)`."""
    L = len(arr)
    if L < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
    a32 = arr.astype(np.int32, copy=False)
    z0 = a32 - 1
    b32 = (z0 * 32) // DIMENSION

    ev1_list = (
        z0[:-1] * DIMENSION + z0[1:],
        126025 + b32[:-1] * DIMENSION + z0[1:],
        137385 + z0[:-1] * 32 + b32[1:],
    )
    if L < 4:
        u_idx, cnts = np.unique(np.concatenate(ev1_list), return_counts=True)
        return u_idx.astype(np.int64), cnts.astype(np.float64) / float(L - 1)

    b16 = (z0 * 16) // DIMENSION
    b8 = (z0 * 8) // DIMENSION
    un = _UNIT[z0]
    te = _TEN[z0]
    dl = _DLEN[z0]
    d1 = a32[1:] - a32[:-1]
    d2 = a32[2:] - a32[:-2]

    g_l1 = np.concatenate((
        ev1_list[0],
        ev1_list[1],
        ev1_list[2],
        171465 + b32[:-1] * 32 + b32[1:],
        173769 + d1 + 354,
        175187 + b32[:-1] * 28 + _STEP_LUT[d1 + 354],
        177595 + (te[:-1] * 10 + un[:-1]) * 10 + te[1:],
    ))
    g_l2 = np.concatenate((
        148745 + b32[:-2] * DIMENSION + z0[2:],
        160105 + z0[:-2] * 32 + b32[2:],
        172489 + b32[:-2] * 32 + b32[2:],
        174478 + d2 + 354,
        176083 + (b8[:-2] * 8 + b8[1:-1]) * 8 + b8[2:],
        176595 + (un[:-2] * 10 + un[1:-1]) * 10 + un[2:],
        178595 + (dl[:-2] * 3 + dl[1:-1]) * 3 + dl[2:],
    ))
    g_l3 = 173513 + b16[:-3] * 16 + b16[3:]

    u1, c1 = np.unique(g_l1, return_counts=True)
    u2, c2 = np.unique(g_l2, return_counts=True)
    u3, c3 = np.unique(g_l3, return_counts=True)

    u_all = np.concatenate((u1, u2, u3)).astype(np.int64)
    v_all = np.concatenate((
        c1.astype(np.float64) / float(L - 1),
        c2.astype(np.float64) / float(L - 2),
        c3.astype(np.float64) / float(L - 3),
    ))
    order = np.argsort(u_all, kind="mergesort")
    return u_all[order], v_all[order]


# =============================================================================
# 3. Exact JS-Identical Sequence Feature Extractors
# =============================================================================


def split_chunks(arr: np.ndarray, k: int) -> list[np.ndarray]:
    n = len(arr)
    base = n // k
    rem = n % k
    out = []
    pos = 0
    for i in range(k):
        sz = base + (1 if i < rem else 0)
        out.append(arr[pos : pos + sz])
        pos += sz
    return out


def quantile_linear(vals: np.ndarray, qs: Sequence[float] | np.ndarray) -> np.ndarray:
    s = np.sort(np.asarray(vals, dtype=np.float64))
    n = len(s)
    out = np.empty(len(qs), dtype=np.float64)
    for i, q in enumerate(qs):
        idx = float(q) * float(n - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, n - 1)
        out[i] = s[lo] + (idx - float(lo)) * (s[hi] - s[lo])
    return out


def corrcoef_1d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    da = a - float(np.mean(a))
    db = b - float(np.mean(b))
    den = math.sqrt(float(np.dot(da, da)) * float(np.dot(db, db)))
    return float(np.dot(da, db)) / den if den > 1e-12 else 0.0


def distinct_order(arr: np.ndarray) -> np.ndarray:
    seen = np.zeros(DIMENSION, dtype=bool)
    order: list[int] = []
    for v in arr.tolist():
        z = int(v) - 1
        if 0 <= z < DIMENSION and not seen[z]:
            seen[z] = True
            order.append(z)
    return np.asarray(order, dtype=np.int64)


def precompute_cdf_marg99(arr: np.ndarray) -> np.ndarray:
    af = arr.astype(np.float64)
    marg_mat = _MARG_BASIS[arr - 1]
    jumps_mat = np.stack([(np.abs(np.diff(af)) / 355.0) ** k for k in range(1, 5)], axis=1)
    parts = []
    for mat in (marg_mat, jumps_mat):
        idx_m = np.arange(len(mat), dtype=np.float64)
        for tau in (16.0, 64.0, 1e9):
            w = np.exp(-idx_m / tau)
            w /= float(np.sum(w))
            parts.append(w @ mat)
    return np.concatenate(parts)


def fast_extract_cdf225(order: np.ndarray, marg99: np.ndarray, hazard: np.ndarray) -> np.ndarray:
    n = len(order)
    prev = np.r_[177, order[:-1]]
    jump_order = _JUMP_ORDER_TABLE[prev]
    jump_sel = _INV_JUMP_ORDER_TABLE[prev, order]
    chosen = hazard[order]
    total = np.maximum(float(np.sum(hazard)) - np.r_[0.0, np.cumsum(chosen)[:-1]], 1e-9)

    wb = hazard[:, None] * _RES_BASIS
    wb2 = hazard[:, None] * (_RES_BASIS * _RES_BASIS)
    wb_sum = np.sum(wb, axis=0)
    wb2_sum = np.sum(wb2, axis=0)
    already = np.vstack([np.zeros((1, 32)), np.cumsum(wb[order], axis=0)[:-1]])
    already2 = np.vstack([np.zeros((1, 32)), np.cumsum(wb2[order], axis=0)[:-1]])
    expectation = (wb_sum[None, :] - already) / total[:, None]
    variance = np.maximum((wb2_sum[None, :] - already2) / total[:, None] - expectation * expectation, 0.0)
    residual = _RES_BASIS[order] - expectation

    idx_o = np.arange(n, dtype=np.float64)
    res_parts = []
    for tau in _RES_SCALES:
        w = np.exp(-idx_o / tau)
        res_parts.append(np.sum(w[:, None] * residual, axis=0) / np.sqrt(np.sum((w * w)[:, None] * variance, axis=0) + 0.1))

    rank_pos = np.full(355, n, dtype=np.int32)
    rank_pos[order] = np.arange(n, dtype=np.int32)
    prob = np.where(np.arange(n, dtype=np.int32)[:, None] <= rank_pos[None, :], hazard[None, :] / total[:, None], 0.0)
    prob /= np.sum(prob, axis=1, keepdims=True)

    # Stable argsort of hazard via (hazard, index)
    h_perm = np.argsort(hazard, kind="mergesort")
    inv_h_perm = np.argsort(h_perm)
    row_ar = np.arange(n)
    uppers, sel_masses = [], []
    for p, inv_p in zip(_CDF_PERMS, _INV_CDF_PERMS):
        mass = prob[:, p]
        s_ix = inv_p[order]
        uppers.append(np.cumsum(mass, axis=1)[row_ar, s_ix])
        sel_masses.append(mass[row_ar, s_ix])

    mass_j = np.take_along_axis(prob, jump_order, axis=1)
    uppers.append(np.cumsum(mass_j, axis=1)[row_ar, jump_sel])
    sel_masses.append(mass_j[row_ar, jump_sel])

    mass_h = prob[:, h_perm]
    s_ix_h = inv_h_perm[order]
    uppers.append(np.cumsum(mass_h, axis=1)[row_ar, s_ix_h])
    sel_masses.append(mass_h[row_ar, s_ix_h])

    U_mat = np.stack(uppers, axis=1)
    S_mat = np.stack(sel_masses, axis=1)
    L_mat = U_mat - S_mat
    hi, lo = 2.0 * U_mat - 1.0, 2.0 * L_mat - 1.0
    P_hi = [np.ones_like(hi), hi]
    P_lo = [np.ones_like(lo), lo]
    for k in range(1, 7):
        P_hi.append(((2.0 * k + 1.0) * hi * P_hi[k] - float(k) * P_hi[k - 1]) / float(k + 1))
        P_lo.append(((2.0 * k + 1.0) * lo * P_lo[k] - float(k) * P_lo[k - 1]) / float(k + 1))

    w16 = np.exp(-idx_o / 16.0)
    w16_sq_sum = float(np.sum(w16 * w16))
    inv_S = 1.0 / np.maximum(S_mat, 1e-12)
    pit_cols = []
    for deg in range(1, 7):
        denom_d = 2.0 * (2.0 * deg + 1.0)
        coord = ((P_hi[deg + 1] - P_hi[deg - 1]) - (P_lo[deg + 1] - P_lo[deg - 1])) / denom_d * inv_S
        pit_cols.append((w16 @ coord) / math.sqrt(w16_sq_sum / (2.0 * deg + 1.0) + 0.1))
    pit_vec = np.stack(pit_cols, axis=1).ravel()
    return np.concatenate([*res_parts, marg99, pit_vec])


def fast_full_residual(order: np.ndarray, hazard: np.ndarray) -> np.ndarray:
    n = len(order)
    total = np.maximum(float(np.sum(hazard)) - np.r_[0.0, np.cumsum(hazard[order])[:-1]], 1e-12)
    m_steps = np.full(DIMENSION, n, dtype=np.int64)
    m_steps[order] = np.arange(1, n + 1, dtype=np.int64)
    idx_n = np.arange(n, dtype=np.float64)
    out = []
    for tau in _RES_SCALES:
        w = np.exp(-idx_n / tau)
        w2 = w * w
        A = np.r_[0.0, np.cumsum(w / total)]
        B = np.r_[0.0, np.cumsum(w2 / total)]
        C = np.r_[0.0, np.cumsum(w2 / (total * total))]
        num = -hazard * A[m_steps]
        num[order] += w
        var = np.maximum(hazard * B[m_steps] - (hazard * hazard) * C[m_steps], 0.0)
        out.append(num / np.sqrt(var + 0.1))
    return np.concatenate(out)


def extract_sibling_open(arr: np.ndarray, hazard: np.ndarray) -> np.ndarray:
    """Pure float64 early-opening priors + early-PL choice residuals (2105d), 0-ULP JS-identical."""
    Lp = min(len(arr), 360)
    sub = arr[:Lp]
    z0 = sub - 1
    b64 = (z0 * 64) // DIMENSION
    dec = sub // 10
    m10 = sub % 10
    idx = np.arange(Lp, dtype=np.float64)
    parts = []
    for tau in (3.0, 10.0, 35.0):
        w = np.exp(-idx / tau)
        wn = w / float(np.sum(w))
        parts.extend(
            [
                np.sqrt(np.bincount(z0, weights=wn, minlength=DIMENSION) + 1e-4),
                np.sqrt(np.bincount(b64, weights=wn, minlength=64) + 1e-3),
                np.sqrt(np.bincount(dec, weights=wn, minlength=36) + 1e-3),
                np.sqrt(np.bincount(m10, weights=wn, minlength=10) + 2e-3),
            ]
        )
    order = distinct_order(sub)
    n = len(order)
    total = np.maximum(float(np.sum(hazard)) - np.r_[0.0, np.cumsum(hazard[order])[:-1]], 1e-12)
    m_steps = np.full(DIMENSION, n, dtype=np.int64)
    m_steps[order] = np.arange(1, n + 1, dtype=np.int64)
    idx_n = np.arange(n, dtype=np.float64)
    for tau in (4.5, 10.0):
        w = np.exp(-idx_n / tau)
        w2 = w * w
        A = np.r_[0.0, np.cumsum(w / total)]
        B = np.r_[0.0, np.cumsum(w2 / total)]
        C = np.r_[0.0, np.cumsum(w2 / (total * total))]
        num = -hazard * A[m_steps]
        num[order] += w
        var = np.maximum(hazard * B[m_steps] - (hazard * hazard) * C[m_steps], 0.0)
        parts.append(num / np.sqrt(var + 0.08))
    return np.concatenate(parts)


def extract_windowed_cdf_lex_orders(
    arr: np.ndarray,
    order_full: np.ndarray,
    marg99_full: np.ndarray,
    hazard: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    if len(arr) <= 460:
        return (
            fast_extract_cdf225(order_full, marg99_full, hazard),
            fast_full_residual(order_full, hazard),
            [order_full],
        )
    # Use math.floor(x + 0.5) away from half-ties for deterministic JS/Python rounding
    n_win = max(2, int(math.floor(len(arr) / 325.0 + 0.5)))
    chunks = split_chunks(arr, n_win)
    cdfs, lexs, ords_w = [], [], []
    for ch in chunks:
        if len(ch) >= 50:
            ord_c = distinct_order(ch)
            m99_c = precompute_cdf_marg99(ch)
            cdfs.append(fast_extract_cdf225(ord_c, m99_c, hazard))
            lexs.append(fast_full_residual(ord_c, hazard))
            ords_w.append(ord_c)
    if not cdfs:
        return (
            fast_extract_cdf225(order_full, marg99_full, hazard),
            fast_full_residual(order_full, hazard),
            [order_full],
        )
    return np.mean(cdfs, axis=0), np.mean(lexs, axis=0), ords_w


def periodogram16(zc_scaled: np.ndarray) -> np.ndarray:
    """Explicit 16-harmonic cosine/sine periodogram (bit-exact between Python and JS)."""
    t = np.arange(len(zc_scaled), dtype=np.float64)
    pwr = np.empty(16, dtype=np.float64)
    for m in range(16):
        ang = (2.0 * math.pi * float(m + 1) / 34.0) * t
        re = float(np.dot(zc_scaled, np.cos(ang)))
        im = float(np.dot(zc_scaled, np.sin(ang)))
        pwr[m] = re * re + im * im
    return pwr / (float(np.sum(pwr)) + 1e-9)


def extract_model_free(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    af = arr.astype(np.float64)
    L = len(arr)
    z0 = arr - 1

    cnt = np.bincount(z0, minlength=DIMENSION).astype(np.float64)
    p_tok = cnt / float(L)
    ent_tok = -float(np.sum(p_tok[p_tok > 0] * np.log(p_tok[p_tok > 0]))) / math.log(DIMENSION)

    b16 = np.minimum((z0 * 16) // DIMENSION, 15)
    h16 = np.sqrt((np.bincount(b16, minlength=16) + 0.10) / (L + 1.6))
    h_mod10 = np.sqrt((np.bincount(arr % 10, minlength=10) + 0.10) / (L + 1.0))
    h_tens = np.sqrt((np.bincount((arr // 10) % 10, minlength=10) + 0.10) / (L + 1.0))
    h_hund = np.sqrt((np.bincount(arr // 100, minlength=4) + 0.10) / (L + 0.4))

    seg_edges = [int(math.floor(s * L / 4.0)) if s < 4 else L for s in range(5)]
    # Match np.linspace(0, L, 5, dtype=np.int64): int(s * L / 4)
    seg_h16, seg_means, seg_stds = [], [], []
    for s in range(4):
        sub = z0[seg_edges[s] : seg_edges[s + 1]]
        if len(sub) > 0:
            sb16 = np.minimum((sub * 16) // DIMENSION, 15)
            seg_h16.append(np.sqrt((np.bincount(sb16, minlength=16) + 0.10) / (len(sub) + 1.6)))
            seg_means.append(float(np.mean(sub)) / DIMENSION)
            seg_stds.append(float(np.std(sub)) / DIMENSION)
        else:
            seg_h16.append(np.full(16, 0.25))
            seg_means.append(0.5)
            seg_stds.append(0.28)

    diffs = np.diff(af)
    adiffs = np.abs(diffs)
    n_step = max(len(diffs), 1)
    step_signs = np.array([np.sum(diffs < 0), np.sum(diffs == 0), np.sum(diffs > 0)], dtype=np.float64) / n_step
    d_quant = quantile_linear(adiffs / DIMENSION, (0.1, 0.25, 0.5, 0.75, 0.9, 0.98)) if len(adiffs) else np.zeros(6)
    small_steps = np.array([np.mean(adiffs <= k) for k in (1, 2, 3, 5, 10, 20, 50)], dtype=np.float64)

    markov8 = np.zeros(64, dtype=np.float64)
    b8 = np.minimum((z0 * 8) // DIMENSION, 7)
    if len(b8) >= 2:
        np.add.at(markov8, b8[:-1] * 8 + b8[1:], 1.0)
    markov8 = np.sqrt((markov8 + 0.10) / (n_step + 6.4))

    mod10_tr = np.zeros(100, dtype=np.float64)
    m10 = arr % 10
    if len(m10) >= 2:
        np.add.at(mod10_tr, m10[:-1] * 10 + m10[1:], 1.0)
    mod10_tr = np.sqrt((mod10_tr + 0.05) / (n_step + 5.0))

    acf = []
    zc = af - float(np.mean(af))
    denom = float(np.dot(zc, zc)) + 1e-9
    for lag in range(1, 9):
        acf.append(float(np.dot(zc[:-lag], zc[lag:]) / denom) if L > lag else 0.0)

    seen: dict[int, int] = {}
    rep_lags = []
    first_dup_pos = 1.0
    found_dup = False
    for idx_pos, val in enumerate(arr.tolist()):
        if val in seen:
            if not found_dup:
                first_dup_pos = idx_pos / float(L)
                found_dup = True
            rep_lags.append(idx_pos - seen[val])
        seen[val] = idx_pos
    n_uniq = len(seen)
    uniq_ratio = n_uniq / float(L)
    expected_uniq = DIMENSION * (1.0 - ((DIMENSION - 1.0) / DIMENSION) ** L)
    coupon_excess = (n_uniq - expected_uniq) / max(math.sqrt(L), 1.0)
    lag_hist = np.zeros(len(_LAG_BINS_EDGE) + 1, dtype=np.float64)
    if rep_lags:
        for lg in rep_lags:
            bi = 0
            for edge in _LAG_BINS_EDGE:
                if lg >= edge:
                    bi += 1
                else:
                    break
            lag_hist[bi] += 1.0
        lag_hist /= float(len(rep_lags))

    u_grid = np.array([0.05 * i for i in range(1, 20)], dtype=np.float64)
    emp_cdf = quantile_linear((af - 0.5) / DIMENSION, u_grid) - u_grid
    spec_bins = periodogram16(zc / DIMENSION)

    turn_rate = float(np.mean(diffs[:-1] * diffs[1:] < 0)) if len(diffs) >= 2 else 0.5
    f_h = np.concatenate(
        [
            h16,
            h_mod10,
            h_tens,
            h_hund,
            *seg_h16[:2],
            np.asarray(seg_means),
            np.asarray(seg_stds),
            step_signs,
            d_quant,
            small_steps,
            markov8,
            np.asarray(acf),
            np.asarray([math.log(L / 200.0), L / 400.0, uniq_ratio, coupon_excess, ent_tok, first_dup_pos, len(rep_lags) / L, turn_rate]),
            lag_hist,
            emp_cdf,
            spec_bins,
        ]
    )

    order = distinct_order(arr)
    m = len(order)
    order_f = order.astype(np.float64)
    f_pl = np.zeros(DIMENSION + 28, dtype=np.float64)
    pos = np.arange(m, dtype=np.float64)
    w_rank = 1.0 - pos / max(m, 1)
    f_pl[order] = w_rank - (0.5 * m / DIMENSION)
    h_slice = _H_CUM[:m]
    unobs = (DIMENSION - m) / DIMENSION
    f_pl[DIMENSION : DIMENSION + 16] = [
        m / DIMENSION,
        unobs,
        math.log((m + 1.0) / 200.0),
        float(np.mean(order_f)) / DIMENSION - 0.5,
        float(np.std(order_f)) / DIMENSION,
        float(order_f[0]) / DIMENSION - 0.5,
        float(order_f[-1]) / DIMENSION - 0.5,
        float(np.mean(order_f[: min(10, m)])) / DIMENSION - 0.5,
        float(np.mean(order_f[: min(30, m)])) / DIMENSION - 0.5,
        float(np.mean(order_f[max(0, m - 30) :])) / DIMENSION - 0.5,
        corrcoef_1d(pos, order_f) if m >= 3 else 0.0,
        float(np.mean(np.abs(np.diff(order_f)))) / DIMENSION if m >= 2 else 0.0,
        float(np.mean(np.diff(order) == 1)) if m >= 2 else 0.0,
        float(np.mean(np.diff(order) == -1)) if m >= 2 else 0.0,
        float(np.mean(np.abs(np.diff(order)) <= 5)) if m >= 2 else 0.0,
        float(h_slice[-1]) / 6.5,
    ]
    for q in range(12):
        lo_q = int(math.floor(q * m / 12.0))
        hi_q = int(math.floor((q + 1) * m / 12.0)) if q < 11 else m
        sub_o = order_f[lo_q:hi_q]
        f_pl[DIMENSION + 16 + q] = (float(np.mean(sub_o)) / DIMENSION - 0.5) if len(sub_o) else 0.0

    b32 = np.minimum((z0 * 32) // DIMENSION, 31)
    tr32 = np.zeros(32 * 32, dtype=np.float64)
    if L >= 2:
        np.add.at(tr32, b32[:-1] * 32 + b32[1:], 1.0)
    tr32 = np.sqrt((tr32 + 0.02) / (n_step + 32 * 32 * 0.02))
    tr16_l2 = np.zeros(16 * 16, dtype=np.float64)
    if L >= 3:
        np.add.at(tr16_l2, b16[:-2] * 16 + b16[2:], 1.0)
    tr16_l2 = np.sqrt((tr16_l2 + 0.05) / (max(L - 2, 1) + 256 * 0.05))
    f_dyn = np.concatenate([tr32[::4], tr16_l2[::2], mod10_tr, seg_h16[2], seg_h16[3], np.sqrt((cnt + 0.1) / (L + 35.5))[::16]])

    c_idx, c_val = extract_sparse_seq_events(arr)
    return f_h, f_pl, f_dyn, (c_idx, c_val)


def extract_freq_and_replag(arr: np.ndarray) -> np.ndarray:
    L = len(arr)
    z0 = arr - 1
    c_all = np.bincount(z0, minlength=DIMENSION).astype(np.float64)
    h_all = np.sqrt((c_all + 0.10) / (L + 0.10 * DIMENSION))
    half = L // 2
    c1 = np.bincount(z0[:half], minlength=DIMENSION).astype(np.float64)
    c2 = np.bincount(z0[half:], minlength=DIMENSION).astype(np.float64)
    h1 = np.sqrt((c1 + 0.05) / (max(half, 1) + 0.05 * DIMENSION))
    h2 = np.sqrt((c2 + 0.05) / (max(L - half, 1) + 0.05 * DIMENSION))
    drift_l2 = float(np.linalg.norm(h1 - h2))
    drift_cos = float(np.dot(h1, h2) / (float(np.linalg.norm(h1)) * float(np.linalg.norm(h2)) + 1e-12))
    mult_hist = np.array([np.sum(c_all == k) for k in range(6)] + [np.sum(c_all >= 6)], dtype=np.float64) / DIMENSION
    last_pos: dict[int, int] = {}
    lags = []
    imm_rep = 0
    for i, x in enumerate(arr.tolist()):
        if x in last_pos:
            lg = i - last_pos[x]
            lags.append(lg)
            if lg <= 3:
                imm_rep += 1
        last_pos[x] = i
    rep_rate = len(lags) / max(L, 1)
    mean_lag = float(np.mean(lags) / L) if lags else 1.0
    q_lag = quantile_linear(np.asarray(lags, dtype=np.float64) / L, (0.1, 0.25, 0.5, 0.75, 0.9)) if lags else np.ones(5)
    return np.concatenate([h_all, mult_hist, np.array([drift_l2, drift_cos, rep_rate, imm_rep / max(L, 1), mean_lag]), q_lag])


def extract_geometry_fast(arr: np.ndarray) -> np.ndarray:
    n = len(arr)
    af = arr.astype(np.float64)
    u = (af - 0.5) / 355.0
    x = 2.0 * u - 1.0

    # Exact O(n) average-tie rank on integer alphabet 1..355 (0 ULP difference vs JS!)
    cnt = np.bincount(arr - 1, minlength=DIMENSION).astype(np.float64)
    cum = np.cumsum(cnt)
    avg_rank = cum - cnt + 0.5 * (cnt + 1.0)
    r = (avg_rank[arr - 1] - 0.5) / float(n)
    rx = 2.0 * r - 1.0
    feats: list[float] = []

    P_leg = leg_poly_matrix(x, 16)
    feats.extend(np.mean(P_leg, axis=0).tolist())
    for k in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        ang = 2.0 * math.pi * k * u
        feats.append(float(np.mean(np.cos(ang))))
        feats.append(float(np.mean(np.sin(ang))))

    qs = np.array([0.05 * i for i in range(1, 20)], dtype=np.float64)
    uq = quantile_linear(u, qs)
    feats.extend((uq - qs).tolist())
    feats.extend(
        [
            float(np.mean(u)),
            float(np.std(u)),
            float(uq[14] - uq[4]),
            float(uq[17] - uq[1]),
            float(np.mean(u < 0.05)),
            float(np.mean(u > 0.95)),
            float(np.mean((u > 0.45) & (u < 0.55))),
            math.log(n / 200.0),
        ]
    )

    z = (af - float(np.mean(af))) / (float(np.std(af)) + 1e-9)
    rz = (r - float(np.mean(r))) / (float(np.std(r)) + 1e-9)
    for lag in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32):
        if n > lag:
            l_v, r_v = x[:-lag], x[lag:]
            rl_v, rr_v = rx[:-lag], rx[lag:]
            d_v = u[lag:] - u[:-lag]
            feats.extend(
                [
                    float(np.dot(z[:-lag], z[lag:]) / (n - lag)),
                    float(np.dot(rz[:-lag], rz[lag:]) / (n - lag)),
                    float(np.mean(l_v * r_v)),
                    float(np.mean((l_v * l_v) * r_v)),
                    float(np.mean(l_v * (r_v * r_v))),
                    float(np.mean((l_v * l_v) * (r_v * r_v))),
                    float(np.mean(np.abs(d_v))),
                    float(np.mean(d_v * d_v)),
                    float(np.mean(d_v * d_v * d_v)),
                    float(np.mean(np.sign(d_v))),
                    float(np.mean(np.abs(d_v) < 0.02)),
                    float(np.mean(np.abs(d_v) < 0.10)),
                    float(np.mean(rl_v * rr_v)),
                ]
            )
        else:
            feats.extend([0.0] * 13)

    for a in (x, rx):
        for lag in (1, 2):
            if n > lag:
                l_v, r_v = a[:-lag], a[lag:]
                d_v = r_v - l_v
                feats.extend(
                    [
                        float(np.mean(l_v * r_v)),
                        float(np.mean((l_v * l_v) * r_v)),
                        float(np.mean(l_v * (r_v * r_v))),
                        float(np.mean((l_v * l_v) * (r_v * r_v))),
                        float(np.mean(np.abs(d_v))),
                        float(np.mean(d_v * d_v)),
                        float(np.mean(d_v * d_v * d_v)),
                        float(np.mean(np.sign(d_v))),
                        float(np.mean(np.abs(d_v) < 0.1)),
                    ]
                )
            else:
                feats.extend([0.0] * 9)
        for blk in split_chunks(a, 4):
            if len(blk) >= 2:
                df = np.diff(blk)
                feats.extend(
                    [
                        float(np.mean(blk)),
                        float(np.std(blk)),
                        float(np.mean(np.abs(df))),
                        float(np.mean(df > 0)),
                    ]
                )
            else:
                feats.extend([0.0] * 4)
        bins = np.clip(np.floor((a + 1.0) * 2.0).astype(np.int64), 0, 3)
        feats.extend((np.bincount(bins[:-1] * 4 + bins[1:], minlength=16) / max(n - 1, 1)).tolist())

    if n >= 4:
        arr_l = arr.tolist()
        n3 = n - 2
        c3_cnt = [0] * 27
        tie3 = 0
        for t in range(n3):
            a0, a1, a2 = arr_l[t], arr_l[t + 1], arr_l[t + 2]
            if a0 == a1 or a1 == a2 or a0 == a2:
                tie3 += 1
            # Stable order of (a0, 0), (a1, 1), (a2, 2)
            idx3 = [0, 1, 2]
            vals3 = (a0, a1, a2)
            idx3.sort(key=lambda j: (vals3[j], j))
            c3_cnt[idx3[0] * 9 + idx3[1] * 3 + idx3[2]] += 1
        for p in _PERMS3:
            feats.append(c3_cnt[p[0] * 9 + p[1] * 3 + p[2]] / float(n3))
        feats.append(tie3 / float(n3))

        n4 = n - 3
        c4_cnt = [0] * 256
        tie4 = 0
        for t in range(n4):
            b0, b1, b2, b3 = arr_l[t], arr_l[t + 1], arr_l[t + 2], arr_l[t + 3]
            if b0 == b1 or b0 == b2 or b0 == b3 or b1 == b2 or b1 == b3 or b2 == b3:
                tie4 += 1
            idx4 = [0, 1, 2, 3]
            vals4 = (b0, b1, b2, b3)
            idx4.sort(key=lambda j: (vals4[j], j))
            c4_cnt[idx4[0] * 64 + idx4[1] * 16 + idx4[2] * 4 + idx4[3]] += 1
        for p in _PERMS4:
            feats.append(c4_cnt[p[0] * 64 + p[1] * 16 + p[2] * 4 + p[3]] / float(n4))
        feats.append(tie4 / float(n4))
    else:
        feats.extend([0.0] * 32)

    for dig in (arr % 10, (arr // 10) % 10):
        for lag in (1, 2, 4, 8):
            if n > lag:
                md = ((dig[lag:] - dig[:-lag]) % 10 + 10) % 10
                feats.extend((np.bincount(md, minlength=10) / len(md)).tolist())
            else:
                feats.extend([0.1] * 10)

    if n >= 3:
        dz = np.diff(z)
        feats.extend(
            [
                float(np.mean(dz[:-1] * dz[1:])),
                float(np.mean(np.abs(dz[:-1]) - np.abs(dz[1:]))),
                float(np.mean(np.sign(dz[:-1]) != np.sign(dz[1:]))),
                float(np.mean(np.abs(dz[1:] + dz[:-1]))),
                float(np.mean(np.abs(dz[1:] - dz[:-1]))),
            ]
        )
    else:
        feats.extend([0.0] * 5)

    return np.asarray(feats, dtype=np.float64)


def extract_markov1497(arr: np.ndarray) -> np.ndarray:
    def _rh(vals: np.ndarray, sz: int) -> np.ndarray:
        c = np.bincount(vals, minlength=sz).astype(np.float64) + 0.5
        return np.sqrt(c / float(np.sum(c)))

    pieces = [_rh(arr - 1, 355)]
    for ch in split_chunks(arr, 4):
        pieces.append(_rh(((ch - 1) * 16) // 355, 16))
    pieces.append(_rh(arr % 10, 10))
    bins = ((arr - 1) * 16) // 355
    digs = arr % 10
    for lag in (1, 2, 4):
        if len(arr) > lag:
            pieces.append(_rh(16 * bins[:-lag] + bins[lag:], 256))
            pieces.append(_rh(10 * digs[:-lag] + digs[lag:], 100))
        else:
            pieces.append(np.full(256, 1.0 / 16.0))
            pieces.append(np.full(100, 0.1))
    return np.concatenate(pieces)


def extract_chunk1065(arr: np.ndarray) -> np.ndarray:
    hs = []
    for ch in split_chunks(arr, 3):
        c = np.bincount(ch - 1, minlength=355).astype(np.float64) + 0.5
        hs.append(np.sqrt(c / float(np.sum(c))))
    a, b, c = hs
    return np.r_[(a + b + c) / math.sqrt(3.0), (a - c) / math.sqrt(2.0), (a - 2.0 * b + c) / math.sqrt(6.0)]


def precompute_ar_state(arr: np.ndarray):
    z0 = arr - 1
    L = len(z0)
    row_ix = np.arange(L, dtype=np.int32)
    one_hot = np.zeros((L, 355), dtype=np.int16)
    one_hot[row_ix, z0] = 1

    occ_before = np.empty((L, 355), dtype=np.int16)
    occ_before[0] = 0
    if L > 1:
        np.cumsum(one_hot[:-1], axis=0, out=occ_before[1:])
    occ_c = np.minimum(occ_before, 2).astype(np.int8)

    pos_mat = np.where(one_hot, row_ix[:, None], -10000)
    last_seen = np.empty((L, 355), dtype=np.int32)
    last_seen[0] = -10000
    if L > 1:
        np.maximum.accumulate(pos_mat[:-1], axis=0, out=last_seen[1:])
    lags_clipped = np.clip(row_ix[:, None] - last_seen, 0, 51)
    lag_b = _LAG_BUCKET_LUT[lags_clipped]

    if L >= 3:
        targ = 2 * z0[1:-1] - z0[:-2]
        ap_lo = np.clip(targ - 2, 0, 355).astype(np.int16)
        ap_hi = np.clip(targ + 3, 0, 355).astype(np.int16)
        ap_cols = targ[:, None] + np.arange(-2, 3, dtype=np.int64)[None, :]
        ap_valid = (ap_cols >= 0) & (ap_cols < 355)
        r_grid = np.broadcast_to(np.arange(2, L, dtype=np.int32)[:, None], (L - 2, 5))
        ap_r2 = r_grid[ap_valid]
        ap_c2 = ap_cols[ap_valid].astype(np.int16)
    else:
        ap_lo = np.zeros(0, dtype=np.int16)
        ap_hi = np.zeros(0, dtype=np.int16)
        ap_r2 = np.zeros(0, dtype=np.int32)
        ap_c2 = np.zeros(0, dtype=np.int16)
    return (z0.astype(np.int16), lag_b, occ_c, ap_lo, ap_hi, ap_r2, ap_c2)


def logsumexp_1d(v: np.ndarray) -> float:
    m = float(np.max(v))
    if not math.isfinite(m):
        return m
    return m + math.log(float(np.sum(np.exp(v - m))))


def logsumexp_axis1(mat: np.ndarray, keepdims: bool = False) -> np.ndarray:
    m = np.max(mat, axis=1, keepdims=True)
    res = m + np.log(np.sum(np.exp(mat - m), axis=1, keepdims=True))
    return res if keepdims else res[:, 0]


def logsumexp_axis2(mat3: np.ndarray) -> np.ndarray:
    m = np.max(mat3, axis=2, keepdims=True)
    return (m + np.log(np.sum(np.exp(mat3 - m), axis=2, keepdims=True)))[:, :, 0]


def extract_ar_fisher_from_state(st: tuple[Any, ...], beta0: np.ndarray, alpha0: np.ndarray, M_base: np.ndarray) -> np.ndarray:
    z0, lag_b, occ_c, ap_lo, ap_hi, ap_r2, ap_c2 = st
    L = len(z0)
    a_hi = alpha0[11:]
    lag_w_lut = np.array([0.0, a_hi[0], a_hi[1], a_hi[2], a_hi[3], a_hi[4], a_hi[5]], dtype=np.float64)
    occ_w_lut = np.array([0.0, a_hi[6], a_hi[6] + a_hi[7]], dtype=np.float64)
    a_ap = float(a_hi[8])

    logits = lag_w_lut[lag_b] + occ_w_lut[occ_c]
    logits[0] += beta0
    if L > 1:
        logits[1:] += M_base[z0[:-1]]
    if len(ap_r2) and a_ap != 0.0:
        logits[ap_r2, ap_c2] += a_ap

    lse = logsumexp_axis1(logits)
    P = np.exp(logits - lse[:, None])
    row_ix = np.arange(L)
    p_chosen = P[row_ix, z0]
    surprisal = -np.log(np.maximum(p_chosen, 1e-12))
    step_ent = -np.sum(P * np.log(np.maximum(P, 1e-12)), axis=1)

    C_pad = np.empty((L, 356), dtype=np.float64)
    C_pad[:, 0] = 0.0
    np.cumsum(P, axis=1, out=C_pad[:, 1:])

    def _rsum(rows_s: np.ndarray, lo_idx: np.ndarray, hi_idx: np.ndarray | int) -> np.ndarray:
        lo_c = np.clip(lo_idx, 0, 355)
        hi_c = np.clip(hi_idx, 0, 355)
        return np.maximum(C_pad[rows_s, hi_c] - C_pad[rows_s, lo_c], 0.0)

    obs_d = np.zeros((L, 20), dtype=np.float64)
    exp_d = np.zeros((L, 20), dtype=np.float64)
    var_d = np.zeros((L, 20), dtype=np.float64)

    if L > 1:
        u = z0[:-1].astype(np.int64)
        r1 = np.arange(1, L)
        obs_d[1:, :11] = _MARKOV_B[u, z0[1:]]
        exp_d[1:, 0] = P[r1, u]
        exp_d[1:, 1] = _rsum(r1, u - 1, u) + _rsum(r1, u + 1, u + 2)
        exp_d[1:, 2] = _rsum(r1, u - 2, u - 1) + _rsum(r1, u + 2, u + 3)
        exp_d[1:, 3] = _rsum(r1, u - 5, u - 2) + _rsum(r1, u + 3, u + 6)
        exp_d[1:, 4] = _rsum(r1, u - 12, u - 5) + _rsum(r1, u + 6, u + 13)
        exp_d[1:, 5] = _rsum(r1, u - 30, u - 12) + _rsum(r1, u + 13, u + 31)
        exp_d[1:, 6] = np.sum(P[1:] * _DIST_MAT[u], axis=1)
        exp_d[1:, 7] = _rsum(r1, u + 1, 355)
        P_m10 = P[1:] @ _MOD10_EYE
        rr = np.arange(L - 1)
        exp_d[1:, 8] = P_m10[rr, u % 10]
        exp_d[1:, 9] = P_m10[rr, (u + 1) % 10]
        P_d100 = P[1:] @ _DIV100_EYE
        exp_d[1:, 10] = P_d100[rr, u // 100]

        em = exp_d[1:, :11]
        em2 = em.copy()
        em2[:, 6] = np.sum(P[1:] * _DIST_SQ_MAT[u], axis=1)
        var_d[1:, :11] = np.maximum(em2 - em * em, 0.0)

    lb_ch = lag_b[row_ix, z0]
    oc_ch = occ_c[row_ix, z0]
    for b_id in range(1, 7):
        obs_d[:, 10 + b_id] = lb_ch == b_id
    obs_d[:, 17] = oc_ch >= 1
    obs_d[:, 18] = oc_ch >= 2

    lag_grid = np.bincount((row_ix[:, None] * 7 + lag_b).ravel(), weights=P.ravel(), minlength=L * 7).reshape(L, 7)
    exp_d[:, 11:17] = lag_grid[:, 1:7]
    occ_grid = np.bincount((row_ix[:, None] * 3 + occ_c).ravel(), weights=P.ravel(), minlength=L * 3).reshape(L, 3)
    exp_d[:, 17] = occ_grid[:, 1] + occ_grid[:, 2]
    exp_d[:, 18] = occ_grid[:, 2]
    if L >= 3:
        z_tail = z0[2:]
        obs_d[2:, 19] = (z_tail >= ap_lo) & (z_tail < ap_hi)
        r2 = np.arange(2, L)
        exp_d[2:, 19] = C_pad[r2, ap_hi] - C_pad[r2, ap_lo]
    eh = exp_d[:, 11:]
    var_d[:, 11:] = np.maximum(eh * (1.0 - eh), 0.0)

    res_d = obs_d - exp_d
    idx_t = np.arange(L, dtype=np.float64)
    tok_parts, dyn_parts = [], []
    P_var = P * (1.0 - P)
    for tau in (32.0, 128.0, 1e10):
        w = np.exp(-idx_t / tau)
        w2 = w * w
        obs_tw = np.bincount(z0, weights=w, minlength=355)
        exp_tw = w @ P
        var_tw = w2 @ P_var
        tok_parts.append((obs_tw - exp_tw) / np.sqrt(var_tw + 0.10))
        dyn_parts.append((w @ res_d) / np.sqrt((w2 @ var_d) + 0.05))

    u_mid = C_pad[row_ix, z0 + 1] - 0.5 * p_chosen
    x_pit = 2.0 * u_mid - 1.0
    s_mean, s_std = float(np.mean(surprisal)), float(np.std(surprisal))
    s_q = quantile_linear(surprisal, (0.1, 0.25, 0.5, 0.75, 0.9)).tolist()
    s_tail = [float(np.mean(surprisal > thr)) for thr in (4.5, 5.5, 6.5, 7.5)]
    e_stats = [float(np.mean(step_ent)), float(np.std(step_ent)), float(np.mean(surprisal - step_ent))]
    pit_leg = [
        float(np.mean(x_pit)),
        float(np.mean(0.5 * (3.0 * x_pit * x_pit - 1.0))),
        float(np.mean(0.5 * (5.0 * x_pit * x_pit * x_pit - 3.0 * x_pit))),
        float(np.mean(0.125 * (35.0 * (x_pit ** 4) - 30.0 * (x_pit ** 2) + 3.0))),
        float(np.mean(u_mid < 0.05)),
        float(np.mean(u_mid > 0.95)),
        corrcoef_1d(surprisal[:-1], surprisal[1:]) if L >= 4 and s_std > 1e-8 else 0.0,
        corrcoef_1d(x_pit[:-1], x_pit[1:]) if L >= 4 and float(np.std(x_pit)) > 1e-8 else 0.0,
        math.log(L / 300.0),
    ]
    return np.concatenate([*tok_parts, *dyn_parts, np.asarray([s_mean, s_std, *s_q, *s_tail, *e_stats, *pit_leg], dtype=np.float64)])


def extract_pure_portable_tuple(seq: Sequence[int]):
    arr = np.asarray([int(v) for v in seq if VALUE_MIN <= int(v) <= VALUE_MAX], dtype=np.int64)
    ord_a = distinct_order(arr)
    fh, fpl, fdyn, fet = extract_model_free(arr)
    ffreq = extract_freq_and_replag(arr)
    fgeom = extract_geometry_fast(arr)
    marg99 = precompute_cdf_marg99(arr)
    f1497 = extract_markov1497(arr)
    fchunk = extract_chunk1065(arr)
    return (arr, ord_a, fh, fpl, fdyn, fet, ffreq, fgeom, marg99, f1497, fchunk)


def score_pl_rank_one_portable(coefs: np.ndarray, shared: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, float]:
    all_b = np.vstack([coefs, shared[None, :]])
    unobs = np.ones(355, dtype=bool)
    unobs[order] = False
    if np.any(unobs):
        sub_u = all_b[:, unobs]
        mu = np.max(sub_u, axis=1, keepdims=True)
        outside = (mu + np.log(np.sum(np.exp(sub_u - mu), axis=1, keepdims=True)))[:, 0]
    else:
        outside = np.full(len(all_b), -np.inf)
    chosen = all_b[:, order]
    # Reverse cumulative logaddexp
    n_o = len(order)
    suffix = np.empty_like(chosen)
    suffix[:, n_o - 1] = chosen[:, n_o - 1]
    for t in range(n_o - 2, -1, -1):
        a = suffix[:, t + 1]
        b = chosen[:, t]
        mx = np.maximum(a, b)
        suffix[:, t] = mx + np.log(np.exp(a - mx) + np.exp(b - mx))
    mx2 = np.maximum(suffix, outside[:, None])
    log_risk = mx2 + np.log(np.exp(suffix - mx2) + np.where(np.isfinite(outside[:, None]), np.exp(outside[:, None] - mx2), 0.0))
    sc = -np.mean(log_risk - chosen, axis=1)
    return sc[:-1], float(sc[-1])


# =============================================================================
# 4. Portable Bank Decoder & Exact K=1/2/3 Inference Engine
# =============================================================================


def eval_psi_spline(spline: Mapping[str, Any], x: np.ndarray | float) -> np.ndarray | float:
    """Evaluates psi_vmf(x, r) - psi_vmf(0, r) via exact cubic Hermite spline + asymptotic tail."""
    x_max = float(spline["x_max"])
    y = spline["y"]
    dy = spline["dy"]
    n = len(y)
    h = x_max / float(n - 1)
    scalar = np.isscalar(x)
    xv = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(xv)

    in_mask = xv < x_max
    if np.any(in_mask):
        xi = np.maximum(xv[in_mask], 0.0)
        pos = xi / h
        k = np.minimum(np.floor(pos).astype(np.int64), n - 2)
        t = pos - k.astype(np.float64)
        t2 = t * t
        t3 = t2 * t
        h00 = 2.0 * t3 - 3.0 * t2 + 1.0
        h10 = t3 - 2.0 * t2 + t
        h01 = -2.0 * t3 + 3.0 * t2
        h11 = t3 - t2
        out[in_mask] = h00 * y[k] + h10 * h * dy[k] + h01 * y[k + 1] + h11 * h * dy[k + 1]
    if np.any(~in_mask):
        xo = xv[~in_mask]
        out[~in_mask] = y[-1] + dy[-1] * (xo - x_max)
    return float(out[0]) if scalar else out


def decode_portable_bank(portable: Mapping[str, Any]) -> dict[str, Any]:
    """Unpacks Base64 Float64 & Int16 tensors once into float64 NumPy arrays."""
    M = int(portable["M"])
    ar_b0 = unpack_f64(portable["ar_b0"])
    ar_a0 = unpack_f64(portable["ar_a0"])
    ar_M_base = (_MARKOV_B @ ar_a0[:11]) + ar_b0[None, :]

    splines = {
        int(k): {
            "x_max": float(sp["x_max"]),
            "y": unpack_f64(sp["y"]),
            "dy": unpack_f64(sp["dy"]),
        }
        for k, sp in portable["psi_splines"].items()
    }

    pvs = []
    for pv in portable["pvs"]:
        q_shared = unpack_mat_i16(pv["Q"]) if "Q" in pv else None
        pvs.append(
            {
                "mu0": unpack_f64(pv["mu0"]),
                "sc": unpack_f64(pv["sc"]),
                "u_cot": unpack_f64(pv["u_cot"]),
                "damp_c": float(pv["damp_c"]),
                "W1": unpack_mat_i16(pv["W1"]),
                "Q1": q_shared if q_shared is not None else unpack_mat_i16(pv["Q1"]),
                "b1": unpack_f64(pv["b1"]),
                "r1": float(pv["r1"]),
                "W3": unpack_mat_i16(pv["W3"]),
                "Q3": q_shared if q_shared is not None else unpack_mat_i16(pv["Q3"]),
                "b3": unpack_f64(pv["b3"]),
                "r3": float(pv["r3"]),
            }
        )

    vmfs = []
    for vm in portable["vmfs"]:
        r = int(vm["r"])
        vmfs.append(
            {
                "r": r,
                "spline": splines[r],
                "mu0": unpack_f64(vm["mu0"]),
                "sc": unpack_f64(vm["sc"]),
                "P_proj": unpack_mat_i16(vm["P_proj"]),
                "offset_sub": unpack_f64(vm["offset_sub"]),
                "M_unit_base": unpack_f64(vm["M_unit_base"], (M, r)),
                "kappa_c": unpack_f64(vm["kappa_c"]),
                "rho_c": unpack_f64(vm["rho_c"]),
                "rho_sib": unpack_f64(vm["rho_sib"]),
                "eff_n": unpack_f64(vm["eff_n"]),
                "open_est": {k: unpack_f64(v) for k, v in vm["open_est"].items()},
            }
        )

    kr = portable["kr_mm"]
    kr_dec = {
        "mu": unpack_f64(kr["mu"]),
        "sd": unpack_f64(kr["sd"]),
        "P_krr": unpack_mat_i16(kr["P_krr"]),
        "Z_lm": unpack_mat_i16(kr["Z_lm"]),
        "bw": float(kr["bw"]),
        "alpha_rbf": unpack_f64(kr["alpha_rbf"], (int(kr["n_lm"]), M)),
        "alpha_lap": unpack_f64(kr["alpha_lap"], (int(kr["n_lm"]), M)),
        "H_rbf": unpack_mat_i16(kr["H_rbf"]),
        "H_lap": unpack_mat_i16(kr["H_lap"]),
    }

    ref = portable["ref_et"]
    ref_pairs = []
    for pm in ref["pairs"]:
        wi = unpack_i32(pm["wi"]) if pm["wi"] else np.zeros(0, dtype=np.int64)
        wv = unpack_f64(pm["wv"]) if pm["wv"] else np.zeros(0, dtype=np.float64)
        ref_pairs.append(
            {
                "a": int(pm["a"]),
                "b": int(pm["b"]),
                "w1": unpack_f64(pm["w1"]),
                "m1": unpack_f64(pm["m1"]),
                "wi": wi,
                "wv": wv,
                "w_map": {int(ik): float(vk) for ik, vk in zip(wi.tolist(), wv.tolist())} if len(wi) else {},
                "emid": float(pm["emid"]),
                "w2": unpack_f64(pm["w2"]),
                "m2": unpack_f64(pm["m2"]),
                "mu_a": unpack_f64(pm["mu_a"]),
                "mu_b": unpack_f64(pm["mu_b"]),
                "var2": unpack_f64(pm["var2"]),
                "rel": float(pm["rel"]),
            }
        )
    ref_dec = {
        "mu_c": unpack_f64(ref["mu_c"]),
        "sc_c": unpack_f64(ref["sc_c"]),
        "mu_g": unpack_f64(ref["mu_g"]),
        "sc_g": unpack_f64(ref["sc_g"]),
        "pairs": ref_pairs,
    }

    sb4 = portable["sref_b4"]
    sb4_scalers = []
    for s in sb4["scalers"]:
        mu_b = unpack_f64(s["mu"])
        sc_b = unpack_f64(s["sc"])
        if "rc" in s:
            rc = int(s["rc"])
            uc_mat = unpack_f64(s["uc"], (rc, len(mu_b))) if rc > 0 else np.zeros((0, len(mu_b)), dtype=np.float64)
            dmps = unpack_f64(s["dmp"]) if rc > 0 else np.zeros(0, dtype=np.float64)
        else:
            uc_v = unpack_f64(s["uc"])
            dmp_v = float(s["dmp"])
            uc_mat = uc_v[None, :] if dmp_v > 0.0 else np.zeros((0, len(mu_b)), dtype=np.float64)
            dmps = np.array([dmp_v], dtype=np.float64) if dmp_v > 0.0 else np.zeros(0, dtype=np.float64)
        sb4_scalers.append((mu_b, sc_b, uc_mat, dmps))
    sb4_pairs = []
    for pm in sb4["pairs"]:
        wi = unpack_i32(pm["wi"]) if pm["wi"] else np.zeros(0, dtype=np.int64)
        wv = unpack_f64(pm["wv"]) if pm["wv"] else np.zeros(0, dtype=np.float64)
        sb4_pairs.append(
            {
                "a": int(pm["a"]),
                "b": int(pm["b"]),
                "ws": [unpack_f64(w) for w in pm["ws"]],
                "mids": [unpack_f64(m) for m in pm["mids"]],
                "wi": wi,
                "wv": wv,
                "w_map": {int(ik): float(vk) for ik, vk in zip(wi.tolist(), wv.tolist())} if len(wi) else {},
                "emid_s": float(pm["emid_s"]),
                "bias": float(pm["bias"]),
                "rel": float(pm["rel"]),
                "vmf_mid": float(pm.get("vmf_mid", 0.0)),
            }
        )

    return {
        "M": M,
        "present_sources": list(portable["present_sources"]),
        "hazard": unpack_f64(portable["hazard"]),
        "ar_b0": ar_b0,
        "ar_a0": ar_a0,
        "ar_M_base": ar_M_base,
        "rv_coefs": unpack_f64(portable["rv_coefs"], (M, 355)),
        "rv_shared": unpack_f64(portable["rv_shared"]),
        "pvs": pvs,
        "vmfs": vmfs,
        "kr_mm": kr_dec,
        "ref_et": ref_dec,
        "sref_b4": {"scalers": sb4_scalers, "pairs": sb4_pairs},
        "theta_1": unpack_f64(portable["theta_1"]),
        "theta_3": unpack_f64(portable["theta_3"]),
        "tau_1": float(portable["tau_1"]),
        "tau_2": float(portable.get("tau_2", 0.5 * (float(portable["tau_1"]) + float(portable["tau_3"])))),
        "tau_3": float(portable["tau_3"]),
    }


def _cauchy_radial(raw: np.ndarray, radius: float) -> np.ndarray:
    r_safe = max(float(radius), 1e-6)
    u = raw / r_safe
    rms = np.sqrt(np.mean(u * u, axis=1, keepdims=True))
    return (1.35 * u) / np.sqrt(1.0 + 0.65 * (rms * rms))


def _score_pv_one_packet(pv: Mapping[str, Any], X_pkt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    K = len(X_pkt)
    Z = (X_pkt - pv["mu0"][None, :]) / pv["sc"][None, :]
    if pv["damp_c"] > 0.0:
        u_cot = pv["u_cot"]
        Z = Z - pv["damp_c"] * ((Z @ u_cot)[:, None] * u_cot[None, :])
    h1 = Z @ pv["Q1"].T
    raw1 = h1 @ pv["W1"].T + pv["b1"][None, :]
    h3 = h1 if pv["Q1"] is pv["Q3"] else (Z @ pv["Q3"].T)
    raw3 = h3 @ pv["W3"].T + pv["b3"][None, :]
    t1 = _cauchy_radial(raw1, pv["r1"])
    t3 = _cauchy_radial(raw3, pv["r3"])

    if K == 1:
        tb = 0.72 * t1[0] + 0.28 * t3[0]
        return tb, np.tanh(tb)
    if K == 2:
        tb = 0.50 * t1 + 0.50 * t3
        m_out = np.mean(tb, axis=0)
        th = np.tanh(tb)
        c_out = np.mean(th, axis=0) * np.maximum(0.25, 1.0 - 0.60 * np.var(th, axis=0))
        return m_out, c_out
    tb = 0.30 * t1 + 0.70 * t3
    # Exact median for K=3 (middle element of sorted 3 values; generalized via np.median)
    s_tb = np.sort(tb, axis=0)
    med = s_tb[K // 2] if (K % 2 == 1) else 0.5 * (s_tb[K // 2 - 1] + s_tb[K // 2])
    m_out = 0.65 * np.mean(tb, axis=0) + 0.35 * med
    th = np.tanh(tb)
    c_out = np.mean(th, axis=0) * np.maximum(0.25, 1.0 - 0.85 * np.var(th, axis=0))
    return m_out, c_out


def _score_vmf_one_packet(vm: Mapping[str, Any], X_pkt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    K = len(X_pkt)
    M = len(vm["kappa_c"])
    sp = vm["spline"]
    Mu = vm["M_unit_base"]
    kap_c = vm["kappa_c"]
    # Exact median of kap_c
    s_kc = np.sort(kap_c)
    med_kc = float(s_kc[M // 2]) if (M % 2 == 1) else 0.5 * float(s_kc[M // 2 - 1] + s_kc[M // 2])
    kap_u0 = np.minimum(kap_c, med_kc * 1.25)

    Rb = Mu * vm["rho_c"][:, None]
    pb0 = eval_psi_spline(sp, vm["rho_c"])
    Rsb = Mu * vm["rho_sib"][:, None]
    ps0 = eval_psi_spline(sp, vm["rho_sib"])
    rho_ring = np.minimum(vm["rho_sib"], 0.52 * vm["rho_c"])
    Rring = Mu * rho_ring[:, None]
    pr0 = eval_psi_spline(sp, rho_ring)

    log_norm_c = -eval_psi_spline(sp, kap_c)
    log_norm_u0 = -eval_psi_spline(sp, kap_u0)

    Z = (X_pkt - vm["mu0"][None, :]) / vm["sc"][None, :]
    Zw = Z @ vm["P_proj"].T - vm["offset_sub"][None, :]
    U = Zw / np.maximum(np.linalg.norm(Zw, axis=1, keepdims=True), 1e-12)

    Su = np.sum(U, axis=0)
    zbar = np.mean(Zw, axis=0)
    su_norm = float(np.linalg.norm(Su))

    Uq_c = kap_c[:, None] * Su[None, :]
    norm_b = np.linalg.norm(Rb + Uq_c, axis=1)
    norm_sb = np.linalg.norm(Rsb + Uq_c, axis=1)

    ll_no_norm = eval_psi_spline(sp, norm_b) - pb0
    llc = float(K) * log_norm_c + ll_no_norm
    lls = float(K) * log_norm_c + eval_psi_spline(sp, norm_sb) - ps0
    llu0 = float(K) * log_norm_u0 + eval_psi_spline(sp, su_norm * kap_u0)

    a_u = math.log(U_UNIFORM_WEIGHT / M) + logsumexp_1d(llu0)
    b_u = math.log(U_SIBLING_WEIGHT / M) + logsumexp_1d(lls)
    mx_u = max(a_u, b_u)
    llu = mx_u + math.log(math.exp(a_u - mx_u) + math.exp(b_u - mx_u))
    coh = llc - llu

    norm_ring = np.linalg.norm(Rring + Uq_c, axis=1)
    ll_unif_r = eval_psi_spline(sp, su_norm * kap_c)
    ll_halo_r = eval_psi_spline(sp, norm_ring) - pr0
    a_r = math.log(0.28) + ll_unif_r
    b_r = math.log(0.72) + ll_halo_r
    mx_r = np.maximum(a_r, b_r)
    ring = ll_no_norm - (mx_r + np.log(np.exp(a_r - mx_r) + np.exp(b_r - mx_r)))

    ubar = zbar / max(float(np.linalg.norm(zbar)), 1e-12)
    cos_v = Mu @ ubar
    est = vm["open_est"]
    if K >= 3:
        kap_o, ln_o, kh_o, lnh_o = est["kap3"], est["ln3"], est["kh3"], est["lnh3"]
    elif K == 2:
        kap_o = 0.5 * (est["kap1"] + est["kap3"])
        ln_o = eval_psi_spline(sp, kap_o)
        kh_o = 0.5 * (est["kh1"] + est["kh3"])
        lnh_o = eval_psi_spline(sp, kh_o)
    else:
        kap_o, ln_o, kh_o, lnh_o = est["kap1"], est["ln1"], est["kh1"], est["lnh1"]

    ll_c_o = kap_o * cos_v - ln_o
    ll_h_o = kh_o * cos_v - lnh_o
    a_m = math.log(0.30)
    b_m = math.log(0.70 / M) + logsumexp_1d(ll_h_o)
    mx_m = max(a_m, b_m)
    ll_u_mix = mx_m + math.log(math.exp(a_m - mx_m) + math.exp(b_m - mx_m))
    cent_all = ll_c_o - ll_u_mix

    b_s = math.log(0.70) + ll_h_o
    mx_s = np.maximum(a_m, b_s)
    cent_self = ll_c_o - (mx_s + np.log(np.exp(a_m - mx_s) + np.exp(b_s - mx_s)))
    return coh, ring, cent_all, cent_self


def _score_krr_one_packet(kr: Mapping[str, Any], X_pkt: np.ndarray) -> np.ndarray:
    Zq = ((X_pkt - kr["mu"][None, :]) / kr["sd"][None, :]) @ kr["P_krr"].T
    # Squared Euclidean distances [K, n_lm]
    q_n2 = np.sum(Zq * Zq, axis=1, keepdims=True)
    l_n2 = np.sum(kr["Z_lm"] * kr["Z_lm"], axis=1)[None, :]
    Dq = np.maximum(q_n2 + l_n2 - 2.0 * (Zq @ kr["Z_lm"].T), 0.0)
    bw = kr["bw"]
    Kq_rbf = np.exp(-Dq / bw)
    Kq_lap = np.exp(-np.sqrt(Dq / bw))
    fit_rbf = Kq_rbf @ kr["alpha_rbf"]
    fit_lap = Kq_lap @ kr["alpha_lap"]
    ex_rbf = np.sum(Kq_rbf * (Kq_rbf @ kr["H_rbf"]), axis=1)
    ex_lap = np.sum(Kq_lap * (Kq_lap @ kr["H_lap"]), axis=1)
    var_rbf = np.maximum(1.0 - ex_rbf, 0.0)
    var_lap = np.maximum(1.0 - ex_lap, 0.0)
    sup_q = 0.45 * (fit_rbf / np.sqrt(var_rbf[:, None] + 0.08)) + 0.55 * (fit_lap / np.sqrt(var_lap[:, None] + 0.08))
    return np.mean(sup_q, axis=0)


def evaluate_selective_decision(
    known_probs: Sequence[float] | np.ndarray,
    p_u: float,
    present_sources: Sequence[str],
    *,
    tau_abs: float = ACCEPT_TAU_ABS,
    dom_rival: float = ACCEPT_DOM_RIVAL,
    dom_u: float = ACCEPT_DOM_U,
) -> dict[str, Any]:
    kp = np.asarray(known_probs, dtype=np.float64)
    M = len(present_sources)
    top_idx = int(np.argmax(kp))
    p_1 = float(kp[top_idx])
    if M >= 2:
        masked = kp.copy()
        masked[top_idx] = -1.0
        runner_idx = int(np.argmax(masked))
        p_2 = float(kp[runner_idx])
        runner_label = str(present_sources[runner_idx])
    else:
        p_2 = 0.0
        runner_label = None

    rival_ratio = p_1 / max(p_2, 1e-12)
    u_ratio = p_1 / max(float(p_u), 1e-12)

    if p_1 >= tau_abs and rival_ratio >= dom_rival and u_ratio >= dom_u:
        decision = "accept"
        predicted_source = str(present_sources[top_idx])
        reject_reason = None
        reject_reason_zh = None
    else:
        decision = "reject"
        predicted_source = None
        if u_ratio < dom_u:
            reject_reason = "outside_library_or_incompatible"
            reject_reason_zh = f"未知/库外来源概率过高（P(U) = {p_u * 100:.1f}%，Top-1 / U = {u_ratio:.2f}× < {dom_u:.1f}×），疑似未收录型号或受强系统提示词干扰"
        elif rival_ratio < dom_rival:
            reject_reason = "insufficient_margin_over_rival"
            reject_reason_zh = f"候选近亲区分度不足（领先次优 {runner_label} 仅 {rival_ratio:.2f}× < {dom_rival:.1f}×），建议提供完整 3 条挑战回答"
        else:
            reject_reason = "below_absolute_probability_floor"
            reject_reason_zh = f"最高候选绝对概率不足（P_abs = {p_1 * 100:.1f}% < {tau_abs * 100:.0f}%）"

    return {
        "decision": decision,
        "predicted_source": predicted_source,
        "reject_reason": reject_reason,
        "reject_reason_zh": reject_reason_zh,
        "top_candidate": str(present_sources[top_idx]),
        "absolute_probability": p_1,
        "runner_up_candidate": runner_label,
        "runner_up_probability": p_2,
        "unknown_probability": float(p_u),
        "rival_dominance_ratio": rival_ratio,
        "unknown_dominance_ratio": u_ratio,
        "thresholds": {
            "tau_abs": float(tau_abs),
            "dom_rival": float(dom_rival),
            "dom_u": float(dom_u),
        },
    }


def score_v6_portable_packet(
    sequences: Sequence[Sequence[int]],
    decoded: Mapping[str, Any],
    *,
    pi_unknown: float = 0.5,
    tau_abs: float = ACCEPT_TAU_ABS,
    dom_rival: float = ACCEPT_DOM_RIVAL,
    dom_u: float = ACCEPT_DOM_U,
) -> dict[str, Any]:
    canon_members = sorted(
        tuple(int(v) for v in seq if VALUE_MIN <= int(v) <= VALUE_MAX)
        for seq in sequences
    )
    if not canon_members:
        raise ValueError("Empty packet")
    # Cap at 3 canonical responses if >3 supplied, or evaluate all K>=1
    K = len(canon_members)
    M = int(decoded["M"])
    hazard = decoded["hazard"]
    ar_b0, ar_a0, ar_M_base = decoded["ar_b0"], decoded["ar_a0"], decoded["ar_M_base"]
    rv_coefs, rv_shared = decoded["rv_coefs"], decoded["rv_shared"]
    stable_cols = list(STABLE_CDF_COLS)

    v1s, v2s, v3s, v4s, v5s, krrs, m14fs = [], [], [], [], [], [], []
    cdfs, lexs, dgs, fulls, vms, geoms, ets, opens = [], [], [], [], [], [], [], []
    fhs, taboos, grams = [], [], []
    n_sb4_scalers = len(decoded["sref_b4"]["scalers"])
    has_b5 = bool(decoded["sref_b4"]["pairs"]) and n_sb4_scalers == 5
    has_b6 = bool(decoded["sref_b4"]["pairs"]) and n_sb4_scalers == 6
    pl_sum = np.zeros(M, dtype=np.float64)
    pl_pop_sum = 0.0

    for tup in canon_members:
        p = extract_pure_portable_tuple(tup)
        arr, ord_a, fh, fpl, fdyn, fet, ffreq, fgeom, marg99, f1497, fchunk = p
        far = extract_ar_fisher_from_state(precompute_ar_state(arr), ar_b0, ar_a0, ar_M_base)
        fc, fl, ow = extract_windowed_cdf_lex_orders(arr, ord_a, marg99, hazard)
        cs_w, ps_w = zip(*(score_pl_rank_one_portable(rv_coefs, rv_shared, o) for o in ow))
        pl_sum += np.mean(cs_w, axis=0)
        pl_pop_sum += float(np.mean(ps_w))

        v1s.append(fc[:195])
        v2s.append(np.r_[fl, fc[96:195], ffreq])
        v3s.append(far)
        v4s.append(np.r_[f1497, fchunk])
        v5s.append(np.r_[fc, fdyn, fgeom, fpl])
        krrs.append(np.r_[far[:1088], f1497, fc, fgeom])
        m14fs.append(np.r_[f1497, ffreq])

        cdfs.append(fc)
        lexs.append(fl)
        dgs.append(np.r_[fdyn, fgeom])
        fulls.append(np.r_[fc, fpl, fh])
        vms.append(np.r_[f1497, fchunk[355:]])
        geoms.append(fgeom)
        ets.append(fet)
        if has_b5:
            opens.append(extract_sibling_open(arr, hazard))
        if has_b6:
            fhs.append(fh)
            taboos.append(extract_taboo_fast(arr))
            grams.append(extract_seq_grammar(arr))

    pl_gain = pl_sum - pl_pop_sum
    Xcdf_m = np.stack(cdfs)
    X_v3 = np.stack(v3s)
    X_B = np.stack(dgs)
    X_C = np.stack(fulls)
    X_L = np.stack(lexs)
    X_M = np.stack(vms)
    X_geom = np.stack(geoms)
    X_14f = np.stack(m14fs)

    pairs = np.asarray(list(itertools.combinations(range(M), 2)), dtype=np.int64)
    p_first, p_second = pairs[:, 0], pairs[:, 1]

    pvs = decoded["pvs"]
    m1, c1 = _score_pv_one_packet(pvs[0], np.stack(v1s))
    m2, c2 = _score_pv_one_packet(pvs[1], np.stack(v2s))
    m3, c3_p = _score_pv_one_packet(pvs[2], X_v3)
    m4, c4 = _score_pv_one_packet(pvs[3], np.stack(v4s))
    m5, c5 = _score_pv_one_packet(pvs[4], np.stack(v5s))
    m_pl = pl_gain[p_first] - pl_gain[p_second]

    kr_s = _score_krr_one_packet(decoded["kr_mm"], np.stack(krrs))
    s_kr = np.sort(kr_s)
    med_kr = float(s_kr[M // 2]) if (M % 2 == 1) else 0.5 * float(s_kr[M // 2 - 1] + s_kr[M // 2])
    kr_z = (kr_s - med_kr) / max(float(np.std(kr_s)), 0.12)
    m_kr = kr_z[p_first] - kr_z[p_second]

    vmfs = decoded["vmfs"]
    v_inputs = (X_v3, Xcdf_m[:, stable_cols], X_B, X_C, X_L, X_M)
    coh_tot = np.zeros(M, dtype=np.float64)
    rng_tot = np.zeros(M, dtype=np.float64)
    cent_all = np.zeros(M, dtype=np.float64)
    cent_self = np.zeros(M, dtype=np.float64)
    for w_v, vm_ch, X_in in zip(VMF_WEIGHTS_6, vmfs, v_inputs):
        c_i, r_i, ca_i, cs_i = _score_vmf_one_packet(vm_ch, X_in)
        coh_tot += w_v * c_i
        rng_tot += w_v * r_i
        cent_all += w_v * ca_i
        cent_self += w_v * cs_i

    # Clean ET + Subliminal/Macro Sibling Refiner (skips pairs handled by multi-panel sref_b4)
    sref_b4 = decoded["sref_b4"]
    b4_covered = {(pm["a"], pm["b"]) for pm in sref_b4["pairs"]}
    ref_et = decoded["ref_et"]
    r_vmf0 = GAMMA_COND_VMF * coh_tot
    r_vmf = r_vmf0.copy()
    if ref_et["pairs"]:
        q0 = np.exp(r_vmf0 - logsumexp_1d(r_vmf0))
        adj = np.zeros(M, dtype=np.float64)
        zc = np.mean((Xcdf_m - ref_et["mu_c"][None, :]) / ref_et["sc_c"][None, :], axis=0)
        zg = np.mean((X_geom - ref_et["mu_g"][None, :]) / ref_et["sc_g"][None, :], axis=0)
        z_sub = zc[list(SUBLIMINAL_COLS)]
        z_mg = np.r_[zc[list(MACRO_COLS)], zg]
        for pm in ref_et["pairs"]:
            a, b = pm["a"], pm["b"]
            if (a, b) in b4_covered or q0[a] + q0[b] < 0.01:
                continue
            et_sum = 0.0
            w_map = pm.get("w_map")
            if w_map:
                for c_idx, c_val in ets:
                    for ik, vk in zip(c_idx.tolist(), c_val.tolist()):
                        w_val = w_map.get(ik)
                        if w_val is not None:
                            et_sum += vk * w_val
            et_sc = (et_sum / float(K)) - pm["emid"]
            s1 = 0.55 * float(np.dot(z_sub - pm["m1"], pm["w1"])) + 0.45 * et_sc
            s2 = float(np.dot(z_mg - pm["m2"], pm["w2"]))
            q2 = np.array([s1, s2], dtype=np.float64)
            vk = pm["var2"] / float(K)
            lla = -0.5 * float(np.sum(((q2 - pm["mu_a"]) ** 2) / vk))
            llb = -0.5 * float(np.sum(((q2 - pm["mu_b"]) ** 2) / vk))
            lam_ab = float(np.clip(ALPHA_SIBLING * pm["rel"] * (lla - llb), -SIBLING_CLIP, SIBLING_CLIP))
            adj[a] += q0[b] * lam_ab
            adj[b] -= q0[a] * lam_ab
        r_vmf = r_vmf0 + adj

    # Multi-Block Unpaired-Covariance + Sequential OOF-Calibrated Sibling Expert
    b4_pair_mat = np.zeros((M, M), dtype=np.float64)
    if sref_b4["pairs"]:
        k_gain = float(K) / (1.0 + float(K - 1) * 0.12)
        if has_b6 and len(sref_b4["scalers"][0][0]) == 96:
            raw_blocks = (Xcdf_m[:, :96], Xcdf_m[:, stable_cols], X_14f[:, 355:1497], X_14f, np.stack(taboos), np.stack(grams))
        elif has_b6:
            raw_blocks = (Xcdf_m, np.c_[np.stack(fhs), X_geom], X_14f, X_L, np.stack(taboos), np.stack(grams))
        elif n_sb4_scalers == 4 and len(sref_b4["scalers"][0][0]) == 96:
            raw_blocks = (Xcdf_m[:, :96], Xcdf_m[:, stable_cols], X_14f[:, 355:1497], X_14f)
        elif has_b5:
            raw_blocks = (Xcdf_m[:, :96], Xcdf_m, np.c_[X_14f[:, :355], X_14f[:, 1497:]], X_L, np.stack(opens))
        else:
            raw_blocks = (Xcdf_m, X_B, X_14f, X_L)
        Zbars = []
        for X_blk, (mu_b, sc_b, uc_mat, dmps) in zip(raw_blocks, sref_b4["scalers"]):
            Z_b = (X_blk - mu_b[None, :]) / sc_b[None, :]
            if len(dmps) > 0:
                Z_b = Z_b - ((Z_b @ uc_mat.T) * dmps[None, :]) @ uc_mat
            Zbars.append(np.mean(Z_b, axis=0))
        n_b_use = len(Zbars)
        for pm in sref_b4["pairs"]:
            a, b = pm["a"], pm["b"]
            et_sc = 0.0
            w_map = pm.get("w_map")
            if w_map:
                et_sum = 0.0
                for c_idx, c_val in ets:
                    for ik, vk in zip(c_idx.tolist(), c_val.tolist()):
                        w_val = w_map.get(ik)
                        if w_val is not None:
                            et_sum += vk * w_val
                et_sc = (et_sum / float(K)) - pm["emid_s"]
            tot_1 = et_sc + pm["bias"]
            for bi in range(n_b_use):
                tot_1 += float(np.dot(Zbars[bi] - pm["mids"][bi], pm["ws"][bi]))
            llr_k = float(np.clip(pm["rel"] * k_gain * tot_1, -9.5, 9.5))
            b4_pair_mat[a, b] = llr_k
            b4_pair_mat[b, a] = -llr_k

    m_vmf = r_vmf[p_first] - r_vmf[p_second]

    F_pair = np.stack([m1, c1, m2, c2, m3, c3_p, m4, c4, m5, c5, m_pl, m_vmf, m_kr], axis=1)
    if K >= 3:
        th_use = decoded["theta_3"]
        tau_use = decoded["tau_3"]
    elif K == 2:
        th_use = 0.45 * decoded["theta_1"] + 0.55 * decoded["theta_3"]
        tau_use = float(decoded.get("tau_2", 0.50 * (decoded["tau_1"] + decoded["tau_3"])))
    else:
        th_use = decoded["theta_1"]
        tau_use = decoded["tau_1"]

    pair_logits = F_pair @ th_use
    L_mat = np.zeros((M, M), dtype=np.float64)
    L_mat[p_first, p_second] = pair_logits
    L_mat[p_second, p_first] = -pair_logits
    for pm in sref_b4["pairs"]:
        a, b = pm["a"], pm["b"]
        if pm["rel"] >= 0.70:
            if a < b:
                p_idx = a * M - (a * (a + 1)) // 2 + (b - a - 1)
                l_debiased = float((F_pair[p_idx] * DEBIASED_SIB_MASK) @ th_use)
            else:
                p_idx = b * M - (b * (b + 1)) // 2 + (a - b - 1)
                l_debiased = -float((F_pair[p_idx] * DEBIASED_SIB_MASK) @ th_use)
            lo_ab = 0.25 * l_debiased + 1.20 * float(b4_pair_mat[a, b])
            L_mat[a, b] = lo_ab
            L_mat[b, a] = -lo_ab
        else:
            sib_boost = 1.00 * float(b4_pair_mat[a, b])
            L_mat[a, b] += sib_boost
            L_mat[b, a] -= sib_boost
    raw_cond = -logsumexp_axis1(-L_mat)

    scaled_cond = tau_use * raw_cond
    log_q_cond = scaled_cond - logsumexp_1d(scaled_cond)
    top_idx = int(np.argmax(scaled_cond))

    log_prior_odds = math.log((1.0 - float(pi_unknown)) / float(pi_unknown))
    n_top = float(vmfs[3]["eff_n"][top_idx])
    size_scale = n_top / (n_top + 4.5)

    lbf_coh = max(0.76 * (logsumexp_1d(coh_tot) - math.log(M)), -3.60)
    if K >= 3:
        tau_ring, w_krr, krr_flr, cent_thr = 1.06 * size_scale, 2.20, 0.12, 1.48 * size_scale
    elif K == 2:
        tau_ring, w_krr, krr_flr, cent_thr = 0.75 * size_scale, 1.75, 0.11, 1.01 * size_scale
    else:
        tau_ring, w_krr, krr_flr, cent_thr = 0.44 * size_scale, 1.30, 0.10, 0.54 * size_scale

    lbf_ring = 0.85 * (float(rng_tot[top_idx]) - tau_ring) + w_krr * (float(kr_s[top_idx]) - krr_flr)
    cent_lse = logsumexp_1d(cent_all) - math.log(M)
    lbf_cent = 0.45 * (0.45 * cent_lse + 0.55 * float(cent_self[top_idx]) - cent_thr)
    lbf_known_total = 0.44 * lbf_coh + 0.44 * lbf_ring + 0.12 * lbf_cent + log_prior_odds

    # Stable log1p(exp(.))
    def _logaddexp0(val: float) -> float:
        return val + math.log1p(math.exp(-val)) if val > 0.0 else math.log1p(math.exp(val))

    log_pU = -_logaddexp0(lbf_known_total)
    log_pKnown = -_logaddexp0(-lbf_known_total)
    joint_lp = np.r_[log_pKnown + log_q_cond, log_pU]
    joint_lp = joint_lp - logsumexp_1d(joint_lp)
    probs = np.exp(joint_lp)
    probs /= float(np.sum(probs))

    known_probs = probs[:M]
    p_u = float(probs[M])
    cond_probs = np.exp(log_q_cond)
    cond_probs /= float(np.sum(cond_probs))

    decision_info = evaluate_selective_decision(
        known_probs,
        p_u,
        decoded["present_sources"],
        tau_abs=tau_abs,
        dom_rival=dom_rival,
        dom_u=dom_u,
    )
    return {
        "known_probabilities": known_probs,
        "unknown_probability": p_u,
        "conditional_probabilities": cond_probs,
        "raw_scores": scaled_cond,
        **decision_info,
    }
