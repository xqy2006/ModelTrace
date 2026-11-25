from __future__ import annotations

import json
import math
import random
import re
import secrets
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import v6_portable as vp

VALUE_MIN = 1
VALUE_MAX = 355
DIMENSION = VALUE_MAX - VALUE_MIN + 1
ALPHA = 0.5
ORDERED_BLOCK_WEIGHT = 0.25
DEFAULT_BANK = Path(__file__).with_name("data") / "gpt_bank.json"
FAMILY_DISPLAY_NAMES = {
    "gpt": "GPT",
    "claude": "Claude",
}

_DECODED_PORTABLE_CACHE: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}


def _get_decoded_v6(bank: dict[str, Any]) -> dict[str, Any]:
    portable = bank["v6_portable"]
    key = id(portable)
    hit = _DECODED_PORTABLE_CACHE.get(key)
    if hit is not None and hit[0] is portable:
        return hit[1]
    decoded = vp.decode_portable_bank(portable)
    if len(_DECODED_PORTABLE_CACHE) >= 16:
        _DECODED_PORTABLE_CACHE.clear()
    _DECODED_PORTABLE_CACHE[key] = (portable, decoded)
    return decoded


def parse_numbers(text: str) -> list[int]:
    runs: list[list[int]] = []
    current: list[int] = []
    previous_end = 0
    for match in re.finditer(r"\d+", text):
        separator = text[previous_end : match.start()]
        value = int(match.group())
        if current and any(character.isalpha() for character in separator):
            runs.append(current)
            current = []
        if VALUE_MIN <= value <= VALUE_MAX:
            current.append(value)
        previous_end = match.end()
    if current:
        runs.append(current)
    return max(runs, key=len) if runs else []


def count_numbers(numbers: Iterable[int]) -> list[int]:
    counts = [0] * DIMENSION
    for number in numbers:
        counts[number - VALUE_MIN] += 1
    return counts


def standardize(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    scale = max(math.sqrt(variance), 1e-12)
    return [(value - mean) / scale for value in values]


def hellinger_feature(counts: list[int]) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64) + ALPHA
    return np.sqrt(values / values.sum())


def ordered_block_feature(numbers: list[int]) -> np.ndarray:
    values = np.asarray(numbers, dtype=np.float64)
    pieces = []
    for chunk in np.array_split(values, 4):
        counts, _ = np.histogram(chunk, bins=16, range=(1.0, 356.0))
        smoothed = counts.astype(np.float64) + 0.5
        pieces.append(np.sqrt(smoothed / smoothed.sum()))
    last_digits = np.bincount(values.astype(int) % 10, minlength=10).astype(np.float64) + 0.5
    pieces.append(np.sqrt(last_digits / last_digits.sum()))
    return np.concatenate(pieces)


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def js_similarity(left: list[int], right: list[int]) -> float:
    left_total = sum(left)
    right_total = sum(right) + ALPHA * DIMENSION
    p = [value / left_total for value in left]
    q = [(value + ALPHA) / right_total for value in right]
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q)]

    def divergence(values: list[float], middle: list[float]) -> float:
        return sum(value * math.log(value / target) for value, target in zip(values, middle) if value)

    js = (divergence(p, midpoint) + divergence(q, midpoint)) / 2.0
    return 1.0 - math.sqrt(max(js, 0.0) / math.log(2.0))


def load_bank(path: Path = DEFAULT_BANK) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_operating_thresholds(
    operating_point: str = "default",
    tau_abs: float | None = None,
    dom_rival: float | None = None,
    dom_u: float | None = None,
) -> tuple[str, float, float, float]:
    op_key = operating_point if operating_point in vp.OPERATING_POINTS else "default"
    preset = vp.OPERATING_POINTS[op_key]
    return (
        op_key,
        float(preset["tau_abs"] if tau_abs is None else tau_abs),
        float(preset["dom_rival"] if dom_rival is None else dom_rival),
        float(preset["dom_u"] if dom_u is None else dom_u),
    )


def analyze_outputs(
    outputs: list[dict],
    bank: dict,
    *,
    operating_point: str = "default",
    tau_abs: float | None = None,
    dom_rival: float | None = None,
    dom_u: float | None = None,
) -> dict:
    op_id, t_abs, d_riv, d_u = resolve_operating_thresholds(operating_point, tau_abs, dom_rival, dom_u)
    model_ids = [model["id"] for model in bank["models"]]
    bank_models = {model["id"]: model for model in bank["models"]}

    valid = []
    diagnostics = []
    for index, item in enumerate(outputs):
        text = str(item.get("text", ""))
        expected = int(item.get("expected_count") or 0)
        numbers = parse_numbers(text)
        minimum = max(80, math.ceil(expected * 0.55)) if expected else 80
        accepted = len(numbers) >= minimum
        diagnostics.append(
            {
                "index": index,
                "parsed_numbers": len(numbers),
                "minimum_numbers": minimum,
                "accepted": accepted,
            }
        )
        if accepted:
            valid.append({"numbers": numbers, "counts": count_numbers(numbers)})

    if not valid:
        raise ValueError("没有可用回答：请粘贴完整数字序列；拒答或严重截断的回答不会计入。")

    pooled_counts = [sum(item["counts"][index] for item in valid) for index in range(DIMENSION)]
    calibration_key = str(min(len(valid), 3))
    cal_info = bank.get("calibration", {}).get(calibration_key, {"beta": 1.0, "cv_accuracy": None})

    if "v6_portable" in bank:
        decoded = _get_decoded_v6(bank)
        v6_res = vp.score_v6_portable_packet(
            [item["numbers"] for item in valid],
            decoded,
            tau_abs=t_abs,
            dom_rival=d_riv,
            dom_u=d_u,
        )
        src_map = {src: idx for idx, src in enumerate(decoded["present_sources"])}
        known_p = v6_res["known_probabilities"]
        cond_p = v6_res["conditional_probabilities"]
        raw_s = v6_res["raw_scores"]
        p_u = float(v6_res["unknown_probability"])

        results = []
        for model_id in model_ids:
            idx = src_map[model_id]
            results.append(
                {
                    "model": model_id,
                    "display_name": bank_models[model_id]["display_name"],
                    "probability": float(known_p[idx]),
                    "in_library_probability": float(cond_p[idx]),
                    "profile_similarity": js_similarity(pooled_counts, bank_models[model_id]["counts"]),
                    "score": float(raw_s[idx]),
                }
            )
        results.sort(key=lambda item: (item["probability"], item["in_library_probability"]), reverse=True)
        accepted_id = v6_res["predicted_source"]
        accepted_name = bank_models[accepted_id]["display_name"] if accepted_id else "拒绝归因（未达显著优势阈值）"
        return {
            "prediction": results[0]["model"],
            "prediction_name": results[0]["display_name"],
            "probability": results[0]["probability"],
            "in_library_probability": results[0]["in_library_probability"],
            "decision": v6_res["decision"],
            "accepted_model": accepted_id,
            "accepted_model_name": accepted_name,
            "reject_reason": v6_res["reject_reason"],
            "reject_reason_zh": v6_res["reject_reason_zh"],
            "unknown_probability": p_u,
            "known_total_probability": 1.0 - p_u,
            "runner_up_candidate": v6_res["runner_up_candidate"],
            "runner_up_probability": v6_res["runner_up_probability"],
            "rival_dominance_ratio": v6_res["rival_dominance_ratio"],
            "unknown_dominance_ratio": v6_res["unknown_dominance_ratio"],
            "operating_point": op_id,
            "thresholds": v6_res["thresholds"],
            "operating_points": vp.OPERATING_POINTS,
            "used_outputs": len(valid),
            "results": results,
            "diagnostics": diagnostics,
            "calibration": {
                "queries": calibration_key,
                "beta": float(cal_info.get("beta", 1.0)),
                "cv_accuracy": cal_info.get("cv_accuracy"),
            },
            "method": "Absolute-Dominance v6（多视角去混杂 + 绝对概率显著优势拒绝）",
        }

    # Single-model custom bank fallback (M == 1)
    results = [
        {
            "model": model_id,
            "display_name": bank_models[model_id]["display_name"],
            "probability": 1.0 / max(len(model_ids), 1),
            "in_library_probability": 1.0 / max(len(model_ids), 1),
            "profile_similarity": js_similarity(pooled_counts, bank_models[model_id]["counts"]),
            "score": js_similarity(pooled_counts, bank_models[model_id]["counts"]),
        }
        for model_id in model_ids
    ]
    results.sort(key=lambda item: item["score"], reverse=True)
    return {
        "prediction": results[0]["model"],
        "prediction_name": results[0]["display_name"],
        "probability": results[0]["probability"],
        "in_library_probability": results[0]["in_library_probability"],
        "decision": "reject",
        "accepted_model": None,
        "accepted_model_name": "拒绝归因（单模型子库无法进行成对显著性检验）",
        "reject_reason": "single_model_bank",
        "reject_reason_zh": "当前子库仅有 1 个候选模型，请使用统一全局指纹库或至少录入 2 个模型",
        "unknown_probability": 0.5,
        "known_total_probability": 0.5,
        "runner_up_candidate": None,
        "runner_up_probability": 0.0,
        "rival_dominance_ratio": 1.0,
        "unknown_dominance_ratio": 1.0,
        "operating_point": op_id,
        "thresholds": {"tau_abs": t_abs, "dom_rival": d_riv, "dom_u": d_u},
        "operating_points": vp.OPERATING_POINTS,
        "used_outputs": len(valid),
        "results": results,
        "diagnostics": diagnostics,
        "calibration": {
            "queries": calibration_key,
            "beta": 1.0,
            "cv_accuracy": None,
        },
        "method": "Single-Model Similarity Fallback",
    }


def analyze_global_outputs(
    outputs: list[dict],
    bank: dict,
    *,
    operating_point: str = "default",
    tau_abs: float | None = None,
    dom_rival: float | None = None,
    dom_u: float | None = None,
) -> dict:
    result = analyze_outputs(
        outputs,
        bank,
        operating_point=operating_point,
        tau_abs=tau_abs,
        dom_rival=dom_rival,
        dom_u=dom_u,
    )
    model_entries = {model["id"]: model for model in bank["models"]}
    family_order = list(
        dict.fromkeys(
            model.get("family") or "models"
            for model in bank["models"]
        )
    )
    family_names = {
        family_id: next(
            (
                model.get("family_name")
                for model in bank["models"]
                if (model.get("family") or "models") == family_id
                and model.get("family_name")
            ),
            FAMILY_DISPLAY_NAMES.get(family_id, family_id),
        )
        for family_id in family_order
    }
    family_probabilities = {
        family_id: sum(
            item["probability"]
            for item in result["results"]
            if (model_entries[item["model"]].get("family") or "models") == family_id
        )
        for family_id in family_order
    }
    for item in result["results"]:
        model = model_entries[item["model"]]
        family_id = model.get("family") or "models"
        item["family"] = family_id
        item["family_name"] = family_names[family_id]
        fam_prob = family_probabilities[family_id]
        item["conditional_probability"] = (
            item["probability"] / fam_prob if fam_prob > 0.0 else 0.0
        )
    winning_family = max(family_order, key=family_probabilities.get)
    return {
        **result,
        "family_prediction": winning_family,
        "family_prediction_name": family_names[winning_family],
        "family_probability": family_probabilities[winning_family],
        "family_probabilities": [
            {
                "family": family_id,
                "display_name": family_names[family_id],
                "probability": family_probabilities[family_id],
            }
            for family_id in family_order
        ],
    }


def _pick_distinct(rng: random.Random, values: list[str], count: int) -> list[str]:
    selected: list[str] = []
    while len(selected) < count:
        batch = list(values)
        rng.shuffle(batch)
        selected.extend(batch[: count - len(selected)])
    return selected


def generate_challenges(count: int = 3) -> list[dict]:
    rng = random.SystemRandom()
    lengths = rng.sample(range(308, 343), count)
    openings = [
        "这是一次独立的数值选择记录",
        "请完成下面的无语义整数选择任务",
        "执行一次第一反应取值记录",
        "生成一组不承载语义的整数选择",
        "进行一轮快速逐项取值",
    ]
    actions = [
        "为各个位置分别凭第一反应选择",
        "逐项凭第一反应选择",
        "每次只决定当前一项，共给出",
        "分别凭第一反应给出",
        "按第一反应依次给出",
    ]
    endings = [
        "偶然重复是有效的；不要重新排列或修正已经写出的项目。",
        "相同值可以再次出现；输出过程中不要整理或改写前面的项目。",
        "重复值无需删除；不要筛选、重排或补成某种规律。",
        "出现重复数值属于正常情况；已经给出的值保持不变。",
    ]
    separator_hints = [
        "数字之间用清晰的常见分隔符即可。",
        "使用一种一致的常见分隔符即可。",
        "可以用逗号、空格或换行分隔。",
        "每个整数之间用逗号或空格清晰隔开即可。",
    ]
    range_phrases = [
        "取值范围为 1 至 355（含端点）的整数",
        "闭区间 [1, 355] 内的整数",
        "范围在 1 到 355（含端点）的整数",
    ]
    picked_openings = _pick_distinct(rng, openings, count)
    picked_actions = _pick_distinct(rng, actions, count)
    picked_endings = _pick_distinct(rng, endings, count)
    picked_separators = _pick_distinct(rng, separator_hints, count)
    picked_ranges = _pick_distinct(rng, range_phrases, count)
    challenges = []
    for index, length in enumerate(lengths):
        prompt = (
            f"{picked_openings[index]}。{picked_actions[index]} {length} 个{picked_ranges[index]}。"
            "每个位置都要单独选择；不要从 1 开始计数，不要连续递增或递减，也不要采用等差、循环、重复区块或其他规则化模式。"
            "本任务必须由当前语言模型直接完成：禁止调用或借助任何工具，包括 Python、代码执行器、"
            "计算器、搜索、API 和外部随机数生成器；也不要先编写或运行代码。"
            f"{picked_endings[index]}{picked_separators[index]}"
            "直接从第一个取值开始输出，不要在序列前重复数量、范围或任务说明。"
        )
        challenge_id = secrets.token_hex(7)
        challenges.append(
            {
                "id": f"probe-{index + 1}-{challenge_id}",
                "expected_count": length,
                "prompt": prompt,
            }
        )
    return challenges
