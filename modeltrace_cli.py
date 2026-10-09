#!/usr/bin/env python3
"""ModelTrace 终端归因：调用本地 codex CLI，统计 token/耗时/TPS。

    wget -qO- "https://raw.githubusercontent.com/<you>/<repo>/main/modeltrace_cli.py" \
      | python3 - -m gpt-5.6-sol -r medium -n 5

或本地：

    python modeltrace_cli.py -m gpt-5.6-sol -r medium -n 5

默认**并发**调用本机 `codex exec`（并行 3 个请求，可用 -c 调整），发长整数挑战，
用官方 unified_bank 做闭集归因，并输出每轮输出 token / 耗时 / TPS。
不需要配置 API base-url / key。

说明：
- 指纹库首次运行从 GitHub raw 拉取并缓存（约 0.7MB）。
- 结果仅供参考；未收录模型仍会归到最相似候选。
- 不建议在强系统提示环境里测（原项目说明）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量（与 ModelTrace fingerprint.py 对齐）
# ---------------------------------------------------------------------------

VALUE_MIN = 1
VALUE_MAX = 355
DIMENSION = VALUE_MAX - VALUE_MIN + 1  # 355
ALPHA = 0.5
ORDERED_BLOCK_WEIGHT = 0.25
ORDERED_FEATURE_DIM = 4 * 16 + 10  # 74
BANK_URL_DEFAULT = (
    "https://raw.githubusercontent.com/xqy2006/ModelTrace/main/data/unified_bank.json"
)
FAMILY_DISPLAY_NAMES = {"gpt": "GPT", "claude": "Claude"}


# ---------------------------------------------------------------------------
# 纯 Python 数值工具
# ---------------------------------------------------------------------------


def standardize(values: list[float]) -> list[float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / n
    scale = max(math.sqrt(variance), 1e-12)
    return [(value - mean) / scale for value in values]


def l2_normalize(vector: list[float]) -> list[float]:
    norm = max(math.sqrt(sum(x * x for x in vector)), 1e-12)
    return [x / norm for x in vector]


def mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def project_out(vector: list[float], basis: list[list[float]]) -> list[float]:
    if not basis:
        return list(vector)
    coeffs = mat_vec(basis, vector)
    subtract = [0.0] * len(vector)
    for coeff, row in zip(coeffs, basis):
        for j, value in enumerate(row):
            subtract[j] += coeff * value
    return [v - s for v, s in zip(vector, subtract)]


def array_split(values: list, parts: int) -> list[list]:
    length = len(values)
    base, extra = divmod(length, parts)
    chunks = []
    start = 0
    for index in range(parts):
        size = base + (1 if index < extra else 0)
        chunks.append(values[start : start + size])
        start += size
    return chunks


def histogram_16(values: list[float]) -> list[int]:
    counts = [0] * 16
    lo, hi = 1.0, 356.0
    width = (hi - lo) / 16.0
    for value in values:
        if value < lo or value > hi:
            continue
        index = int((value - lo) / width)
        if index >= 16:
            index = 15
        if index < 0:
            index = 0
        counts[index] += 1
    return counts


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


# ---------------------------------------------------------------------------
# 指纹特征与打分
# ---------------------------------------------------------------------------


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


def count_numbers(numbers: list[int]) -> list[int]:
    counts = [0] * DIMENSION
    for number in numbers:
        counts[number - VALUE_MIN] += 1
    return counts


def hellinger_feature(counts: list[int]) -> list[float]:
    values = [c + ALPHA for c in counts]
    total = sum(values)
    return [math.sqrt(v / total) for v in values]


def ordered_block_feature(numbers: list[int]) -> list[float]:
    pieces: list[list[float]] = []
    values = [float(x) for x in numbers]
    for chunk in array_split(values, 4):
        counts = histogram_16(chunk)
        smoothed = [c + 0.5 for c in counts]
        total = sum(smoothed)
        pieces.append([math.sqrt(v / total) for v in smoothed])
    last_digits = [0] * 10
    for value in numbers:
        last_digits[int(value) % 10] += 1
    smoothed = [c + 0.5 for c in last_digits]
    total = sum(smoothed)
    pieces.append([math.sqrt(v / total) for v in smoothed])
    out: list[float] = []
    for piece in pieces:
        out.extend(piece)
    return out


def robust_score_counts(counts: list[int], bank: dict) -> dict[str, list[float]]:
    robust = bank["robust"]
    hellinger = robust["hellinger"]
    feature = hellinger_feature(counts)
    mean = hellinger["feature_mean"]
    scale = hellinger["feature_scale"]
    projected = [(f - m) / s for f, m, s in zip(feature, mean, scale)]
    projected = project_out(projected, hellinger.get("nuisance_basis") or [])
    projected = l2_normalize(projected)
    centroids = hellinger["centroids"]
    nuisance = standardize(mat_vec(centroids, projected))
    fused = standardize(list(nuisance))
    return {"fused": fused, "nuisance": nuisance}


def ordered_block_scores(numbers: list[int], bank: dict) -> list[float]:
    artifact = bank["robust"]["ordered_blocks"]
    feature = ordered_block_feature(numbers)
    mean = artifact["feature_mean"]
    scale = artifact["feature_scale"]
    standardized_feature = [(f - m) / s for f, m, s in zip(feature, mean, scale)]

    normalized = l2_normalize(standardized_feature)
    templates = artifact.get("environment_centroids") or []
    if templates:
        environment_scores = [mat_vec(centroids, normalized) for centroids in templates]
        n_models = len(environment_scores[0])
        template = [max(env[i] for env in environment_scores) for i in range(n_models)]
    else:
        template = mat_vec(artifact["centroids"], normalized)
    template = standardize(template)

    projected = project_out(standardized_feature, artifact.get("nuisance_basis") or [])
    projected = l2_normalize(projected)
    nuisance = standardize(mat_vec(artifact["centroids"], projected))
    return standardize([0.5 * t + 0.5 * n for t, n in zip(template, nuisance)])


def robust_score_numbers(numbers: list[int], bank: dict) -> dict[str, list[float]]:
    marginal = robust_score_counts(count_numbers(numbers), bank)
    artifact = bank["robust"].get("ordered_blocks")
    ordered_weight = float(artifact.get("weight", 0.0)) if artifact else 0.0
    if not artifact or ordered_weight == 0.0:
        return {
            **marginal,
            "marginal_fused": marginal["fused"],
            "ordered": marginal["fused"],
        }
    ordered = ordered_block_scores(numbers, bank)
    fused = [
        (1.0 - ordered_weight) * marginal_score + ordered_weight * ordered_score
        for marginal_score, ordered_score in zip(marginal["fused"], ordered)
    ]
    return {
        **marginal,
        "fused": fused,
        "marginal_fused": marginal["fused"],
        "ordered": ordered,
    }


def js_similarity(left: list[int], right: list[int]) -> float:
    left_total = sum(left)
    right_total = sum(right) + ALPHA * DIMENSION
    p = [value / left_total for value in left]
    q = [(value + ALPHA) / right_total for value in right]
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q)]

    def divergence(values: list[float], middle: list[float]) -> float:
        return sum(value * math.log(value / target) for value, target in zip(values, middle) if value)

    js = (divergence(p, midpoint) + divergence(q, midpoint)) / 2.0
    return 1.0 - math.sqrt(js / math.log(2.0))


def analyze_outputs(outputs: list[dict], bank: dict) -> dict:
    model_ids = [model["id"] for model in bank["models"]]
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
            counts = count_numbers(numbers)
            components = robust_score_numbers(numbers, bank)
            valid.append({"counts": counts, "scores": components["fused"], **components})

    if not valid:
        raise ValueError("没有可用回答：请确认 codex 输出了完整数字序列；拒答或严重截断的回答不会计入。")

    combined_scores = [
        sum(item["scores"][index] for item in valid) / len(valid)
        for index in range(len(model_ids))
    ]
    combined_nuisance = [
        sum(item.get("nuisance", item["scores"])[index] for item in valid) / len(valid)
        for index in range(len(model_ids))
    ]
    calibration_key = str(min(len(valid), 3))
    beta = float(bank["calibration"][calibration_key]["beta"])
    probabilities = softmax([beta * value for value in combined_scores])
    pooled_counts = [sum(item["counts"][index] for item in valid) for index in range(DIMENSION)]
    bank_models = {model["id"]: model for model in bank["models"]}
    results = [
        {
            "model": model_id,
            "display_name": bank_models[model_id]["display_name"],
            "probability": probabilities[index],
            "profile_similarity": js_similarity(pooled_counts, bank_models[model_id]["counts"]),
            "score": combined_scores[index],
            "nuisance_score": combined_nuisance[index],
        }
        for index, model_id in enumerate(model_ids)
    ]
    results.sort(key=lambda item: item["probability"], reverse=True)
    return {
        "prediction": results[0]["model"],
        "prediction_name": results[0]["display_name"],
        "probability": results[0]["probability"],
        "used_outputs": len(valid),
        "results": results,
        "diagnostics": diagnostics,
        "calibration": {
            "queries": calibration_key,
            "beta": beta,
            "cv_accuracy": bank["calibration"][calibration_key]["cv_accuracy"],
        },
        "method": bank.get("method", {}).get("name", "Ordered-block + nuisance-Hellinger"),
    }


def analyze_global_outputs(outputs: list[dict], bank: dict) -> dict:
    result = analyze_outputs(outputs, bank)
    model_entries = {model["id"]: model for model in bank["models"]}
    family_order = list(
        dict.fromkeys(model.get("family") or "models" for model in bank["models"])
    )
    family_names = {
        family_id: next(
            (
                model.get("family_name")
                for model in bank["models"]
                if (model.get("family") or "models") == family_id and model.get("family_name")
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
        item["conditional_probability"] = (
            item["probability"] / family_probabilities[family_id]
            if family_probabilities[family_id]
            else 0.0
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
        "method": "统一全局稳健数字指纹",
    }


def generate_challenges(count: int = 3) -> list[dict]:
    rng = random.SystemRandom()
    sample_size = min(count, 30)
    lengths = rng.sample(range(292, 333), sample_size)
    if count > sample_size:
        lengths.extend(rng.randint(292, 332) for _ in range(count - sample_size))
    openings = [
        "这是一次独立的数值选择记录",
        "请完成下面的无语义整数选择任务",
        "执行一次第一反应取值记录",
        "生成一组不承载语义的整数选择",
        "进行一轮快速逐项取值",
    ]
    actions = [
        "为各个位置分别凭第一反应选择",
        "逐项选择",
        "每次只决定当前一项，共给出",
        "分别凭第一反应给出",
        "逐个直接选择",
    ]
    endings = [
        "允许某个数字再次出现；每项写出后不要回头排序、去重或替换。",
        "偶然重复是有效的；不要重新排列或修正已经写出的项目。",
        "相同值可以再次出现；输出过程中不要整理或改写前面的项目。",
        "重复值无需删除；不要筛选、重排或补成某种规律。",
        "不必赋予数字任何含义；已经给出的值保持不变。",
    ]
    separator_hints = [
        "数字之间用逗号或空格分隔均可。",
        "使用一种一致的常见分隔符即可。",
        "可以用逗号、空格或换行分隔。",
        "只要每个整数边界清楚，格式可自行选择。",
    ]
    challenges = []
    for index, length in enumerate(lengths):
        prompt = (
            f"{rng.choice(openings)}。{rng.choice(actions)} {length} 个 1 到 355（含端点）的整数。"
            "每个位置都要单独选择；不要从 1 开始计数，不要连续递增或递减，也不要采用等差、循环、重复区块或其他规则化模式。"
            "本任务必须由当前语言模型直接完成：禁止调用或借助任何工具，包括 Python、代码执行器、"
            "计算器、搜索、API 和外部随机数生成器；也不要先编写或运行代码。"
            f"{rng.choice(endings)}{rng.choice(separator_hints)}"
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


# ---------------------------------------------------------------------------
# 指纹库加载
# ---------------------------------------------------------------------------


def default_cache_dir() -> Path:
    override = os.environ.get("MODELTRACE_CACHE_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return base / "modeltrace"
    return Path.home() / ".cache" / "modeltrace"


def load_bank(bank_url: str = BANK_URL_DEFAULT, cache_dir: Path | None = None) -> dict:
    cache_dir = cache_dir or default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    local = cache_dir / "unified_bank.json"
    local_bank = cache_dir / "unified_bank.local.json"
    for path in (local_bank, local):
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    try:
        with urllib.request.urlopen(bank_url, timeout=60) as response:
            raw = response.read()
        bank = json.loads(raw.decode("utf-8"))
        local.write_text(json.dumps(bank, ensure_ascii=False), encoding="utf-8")
        return bank
    except Exception as error:
        raise RuntimeError(
            f"无法获取指纹库：{error}\n"
            f"可手动下载到 {local}，或用 --bank-url / --bank-file 指定。"
        ) from error


# ---------------------------------------------------------------------------
# 本地 codex CLI
# ---------------------------------------------------------------------------


def resolve_codex_executable() -> str:
    """找到可被 subprocess 直接启动的 codex 命令（Windows 优先 .cmd）。"""
    candidates = (
        ("codex.cmd", "codex.exe", "codex")
        if os.name == "nt"
        else ("codex",)
    )
    for name in candidates:
        exe = shutil.which(name)
        if exe:
            return exe
    raise RuntimeError("找不到 codex 可执行文件，请确认已安装并加入 PATH。")


def run_codex(prompt: str, model: str | None, effort: str) -> tuple[str, dict, float]:
    """通过本地 codex exec 生成回答，返回 (文本, usage, 耗时秒)。"""
    exe = resolve_codex_executable()
    cmd = [
        exe,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--ephemeral",
        "-s",
        "read-only",
        # 关闭跨会话记忆，避免历史记忆污染指纹
        "--disable",
        "memories",
        "-c",
        f"model_reasoning_effort={effort}",
    ]
    if model:
        cmd += ["-m", model]

    start = time.perf_counter()
    # 多行题目走 stdin，避免 cmd 包装吞换行
    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "codex exec failed")

    final_text = ""
    usage: dict = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                final_text = item.get("text", final_text)
        elif event.get("type") == "turn.completed":
            usage = event.get("usage") or {}
    return final_text, usage, elapsed


# ---------------------------------------------------------------------------
# 终端表格（显示宽度对齐）
# ---------------------------------------------------------------------------


def char_width(char: str) -> int:
    if unicodedata.combining(char):
        return 0
    return 2 if unicodedata.east_asian_width(char) in {"W", "F"} else 1


def display_width(text: str) -> int:
    return sum(char_width(c) for c in text)


def pad(text: str, width: int, align: str) -> str:
    gap = width - display_width(text)
    if gap <= 0:
        return text
    if align == "right":
        return " " * gap + text
    if align == "center":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def render_table(headers: list[str], rows: list[list], aligns: list[str]) -> str:
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [
        max(display_width(headers[i]), *(display_width(r[i]) for r in str_rows))
        if str_rows
        else display_width(headers[i])
        for i in range(len(headers))
    ]

    def fmt(cells: list[str]) -> str:
        return "  ".join(pad(cells[i], widths[i], aligns[i]) for i in range(len(headers)))

    lines = [fmt(headers), "  ".join("-" * w for w in widths)]
    lines += [fmt(r) for r in str_rows]
    return "\n".join(lines)


def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# 评测流程
# ---------------------------------------------------------------------------


def _run_one_challenge(
    index: int,
    challenge: dict,
    model: str | None,
    effort: str,
) -> dict:
    try:
        text, usage, elapsed = run_codex(challenge["prompt"], model, effort)
        minimum = max(80, math.ceil(challenge["expected_count"] * 0.55))
        numbers = parse_numbers(text)
        parsed_count = len(numbers)
        accepted = parsed_count >= minimum
        out_tok = usage.get("output_tokens")
        in_tok = usage.get("input_tokens")
        rea_tok = usage.get("reasoning_output_tokens")
        tps = (out_tok / elapsed) if out_tok and elapsed > 0 else None
        return {
            "index": index,
            "challenge_id": challenge["id"],
            "expected_count": challenge["expected_count"],
            "prompt": challenge["prompt"],
            "text": text,
            "parsed_numbers": parsed_count,
            "minimum_numbers": minimum,
            "accepted": accepted,
            "usage": usage,
            "latency_s": elapsed,
            "tps": tps,
            "error": None,
        }
    except Exception as error:
        return {
            "index": index,
            "challenge_id": challenge["id"],
            "expected_count": challenge["expected_count"],
            "prompt": challenge["prompt"],
            "text": "",
            "parsed_numbers": None,
            "minimum_numbers": None,
            "accepted": False,
            "usage": {},
            "latency_s": None,
            "tps": None,
            "error": str(error),
        }


def test_with_codex(
    model: str | None,
    effort: str,
    target_count: int,
    max_attempts: int,
    bank: dict,
    concurrency: int = 3,
    save_outputs: Path | None = None,
) -> dict:
    challenges = generate_challenges(max_attempts)
    concurrency = max(1, min(concurrency, max_attempts))
    raw_rows: list[dict] = []
    errors: list[str] = []
    accepted_outputs: list[dict] = []

    headers = ["Run", "数字", "InTok", "OutTok", "ReTok", "Time(s)", "TPS", "状态"]
    aligns = ["right", "right", "right", "right", "right", "right", "right", "center"]

    def row_from(item: dict) -> list:
        if item.get("error") is not None:
            return [item["index"], "-", "-", "-", "-", "-", "-", "ERR"]
        usage = item.get("usage") or {}
        out_tok = usage.get("output_tokens")
        in_tok = usage.get("input_tokens")
        rea_tok = usage.get("reasoning_output_tokens")
        tps = item.get("tps")
        return [
            item["index"],
            item.get("parsed_numbers") or "-",
            in_tok if in_tok is not None else "-",
            out_tok if out_tok is not None else "-",
            rea_tok if rea_tok is not None else "-",
            f"{item['latency_s']:.1f}" if item.get("latency_s") is not None else "-",
            f"{tps:.1f}" if tps else "-",
            "✓" if item.get("accepted") else ("ERR" if item.get("error") else "✗"),
        ]

    def flush_table(rows: list[dict]) -> None:
        if not rows:
            return
        ordered = sorted(rows, key=lambda r: r["index"])
        print(render_table(headers, [row_from(r) for r in ordered], aligns), flush=True)

    print(f"并发启动 {concurrency} 个 codex 请求…", file=sys.stderr)
    wall_start = time.perf_counter()
    challenge_iter = iter(enumerate(challenges, start=1))
    futures = {}

    def submit_next(pool: ThreadPoolExecutor) -> bool:
        try:
            index, challenge = next(challenge_iter)
        except StopIteration:
            return False
        print(
            f"[{index}/{max_attempts}] 启动挑战 {challenge['id']} "
            f"(期望约 {challenge['expected_count']} 个数)",
            file=sys.stderr,
        )
        futures[pool.submit(_run_one_challenge, index, challenge, model, effort)] = index
        return True

    def maybe_fill(pool: ThreadPoolExecutor) -> None:
        # 保持最多 concurrency 路在跑，但不超过“仍缺的 有效数”
        still_need = max(0, target_count - len(accepted_outputs) - len(futures))
        free_slots = concurrency - len(futures)
        for _ in range(min(free_slots, still_need)):
            if not submit_next(pool):
                break

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        maybe_fill(pool)

        while futures:
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                futures.pop(fut, None)
                item = fut.result()
                raw_rows.append(item)
                flush_table(raw_rows)
                if item.get("error"):
                    errors.append(str(item["error"]))
                    print(f"    [{item['index']}] 失败：{item['error']}", file=sys.stderr)
                elif item.get("accepted"):
                    accepted_outputs.append(
                        {
                            "text": item["text"],
                            "expected_count": item["expected_count"],
                        }
                    )
                    tps = item.get("tps")
                    extra = f"  TPS={tps:.1f}" if tps else ""
                    print(
                        f"    [{item['index']}] 接受：{item['parsed_numbers']} 数字{extra}",
                        file=sys.stderr,
                    )
                else:
                    errors.append(
                        f"有效数字不足：{item.get('parsed_numbers')}/{item.get('minimum_numbers')}"
                    )
                    print(
                        f"    [{item['index']}] 拒绝："
                        f"{item.get('parsed_numbers')}/{item.get('minimum_numbers')}",
                        file=sys.stderr,
                    )
            maybe_fill(pool)

    wall_time = time.perf_counter() - wall_start

    # 汇总 token / 耗时 / TPS（按完成顺序展示；表内 Run 为挑战序号）
    ok_rows = [row for row in raw_rows if row.get("accepted") and row.get("usage")]
    total_out = sum((row["usage"] or {}).get("output_tokens") or 0 for row in ok_rows)
    total_in = sum((row["usage"] or {}).get("input_tokens") or 0 for row in ok_rows)
    total_re = sum((row["usage"] or {}).get("reasoning_output_tokens") or 0 for row in ok_rows)
    sum_latency = sum(row.get("latency_s") or 0.0 for row in ok_rows)
    avg_tps = (total_out / sum_latency) if total_out and sum_latency > 0 else None
    wall_tps = (total_out / wall_time) if total_out and wall_time > 0 else None
    performance = {
        "runs": [
            {
                "index": row.get("index"),
                "challenge_id": row.get("challenge_id"),
                "parsed_numbers": row.get("parsed_numbers"),
                "accepted": row.get("accepted"),
                "latency_s": row.get("latency_s"),
                "tps": row.get("tps"),
                "usage": row.get("usage") or {},
                "error": row.get("error"),
            }
            for row in sorted(raw_rows, key=lambda r: r.get("index") or 0)
        ],
        "concurrency": concurrency,
        "input_tokens": total_in,
        "output_tokens": total_out,
        "reasoning_output_tokens": total_re,
        "latency_s": sum_latency,
        "wall_time_s": wall_time,
        "tps": avg_tps,
        "wall_tps": wall_tps,
    }
    summary = (
        f"\n性能汇总: OutTok={total_out}  ReTok={total_re}  "
        f"并发={concurrency}  墙钟={wall_time:.1f}s  单请求耗时和={sum_latency:.1f}s"
    )
    if avg_tps is not None:
        summary += f"  单请求TPS={avg_tps:.1f}"
    if wall_tps is not None:
        summary += f"  吞吐TPS={wall_tps:.1f}"
    print(summary, flush=True)

    if save_outputs is not None:
        save_outputs.parent.mkdir(parents=True, exist_ok=True)
        with save_outputs.open("w", encoding="utf-8") as handle:
            for row in sorted(raw_rows, key=lambda r: r.get("index") or 0):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"原始输出已写入: {save_outputs}", file=sys.stderr)

    result = analyze_global_outputs(accepted_outputs, bank)
    result["api_test"] = {
        "requested": target_count,
        "attempted": len(raw_rows),
        "max_attempts": max_attempts,
        "received": len(accepted_outputs),
        "errors": errors,
        "runner": "codex-cli",
        "concurrency": concurrency,
    }
    result["performance"] = performance
    return result


def split_manual_outputs(pasted: str) -> list[dict]:
    blocks = [
        block.strip()
        for block in re.split(r"(?m)^\s*===OUTPUT===\s*$", pasted)
        if block.strip()
    ]
    if not blocks:
        blocks = [pasted.strip()] if pasted.strip() else []
    return [{"text": block, "expected_count": 0} for block in blocks]


def format_report(result: dict, top: int = 8) -> str:
    lines = []
    lines.append("=" * 56)
    lines.append("ModelTrace 归因结果")
    lines.append("=" * 56)
    lines.append(
        f"预测模型: {result['prediction_name']}  P={result['probability']:.1%}"
    )
    lines.append(
        f"预测家族: {result['family_prediction_name']}  P={result['family_probability']:.1%}"
    )
    lines.append(
        f"有效回答: {result['used_outputs']}  "
        f"校准: β={result['calibration']['beta']:.3f} "
        f"CV={result['calibration']['cv_accuracy']}"
    )
    perf = result.get("performance") or {}
    if perf:
        tps = perf.get("tps")
        wall_tps = perf.get("wall_tps")
        wall = perf.get("wall_time_s", 0)
        conc = perf.get("concurrency", 1)
        base = (
            f"输出 token: {perf.get('output_tokens', 0)}  "
            f"并发: {conc}  墙钟: {wall:.1f}s"
        )
        if tps is not None:
            base += f"  单请求TPS: {tps:.1f}"
        if wall_tps is not None:
            base += f"  吞吐TPS: {wall_tps:.1f}"
        lines.append(base)
    lines.append("")
    lines.append(f"{'模型':<28} {'概率':>8} {'家族内':>8} {'相似度':>8}")
    lines.append("-" * 56)
    for item in result["results"][:top]:
        lines.append(
            f"{item['display_name']:<28} "
            f"{item['probability']:>7.1%} "
            f"{item.get('conditional_probability', 0):>7.1%} "
            f"{item.get('profile_similarity', 0):>8.3f}"
        )
    lines.append("")
    lines.append("家族概率:")
    for family in result.get("family_probabilities", []):
        lines.append(f"  {family['display_name']:<12} {family['probability']:>7.1%}")
    if result.get("api_test"):
        api = result["api_test"]
        lines.append("")
        lines.append(
            f"测试: 成功 {api['received']}/{api['requested']} "
            f"(尝试 {api['attempted']}/{api['max_attempts']})"
        )
        for err in api.get("errors", [])[:5]:
            lines.append(f"  - {err}")
    lines.append("")
    lines.append("注：结果仅供参考，不是判断模型的决定性证据。")
    lines.append("    未收录模型仍会被归到最相似的现有候选。")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modeltrace_cli.py",
        description="ModelTrace 终端归因（本地 codex CLI）：输出 token / 耗时 / TPS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 直接用本地 codex，默认并发 3 路，无需配置 API
  python modeltrace_cli.py -m gpt-5.6-sol -r medium

  # 调整并行数 / 尝试次数
  python modeltrace_cli.py -m gpt-5.6-sol -c 3 -n 3

  # 一键在线
  wget -qO- "https://raw.githubusercontent.com/you/repo/main/modeltrace_cli.py" \\
    | python3 - -m gpt-5.6-sol -r medium

  # 手动模式：三份完整输出用 ===OUTPUT=== 分隔
  cat outputs.txt | python3 modeltrace_cli.py --manual

  # 只要 JSON
  python3 modeltrace_cli.py -m gpt-5.6-sol --json
""",
    )
    parser.add_argument("-m", "--model", help="Codex 模型名；省略则用本地默认")
    parser.add_argument(
        "-r",
        "--reasoning-effort",
        default="low",
        choices=["low", "medium", "high", "xhigh", "max", "ultra"],
        help="推理强度（默认 low）",
    )
    parser.add_argument(
        "-n",
        "--tests",
        type=int,
        default=3,
        help="最多尝试几次挑战（默认 3，并发打完即止）",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=3,
        help="并行 codex 请求数（默认 3）",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=3,
        help="目标有效回答数（默认 3，对应校准表）",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="手动模式：从 stdin 读取回答（===OUTPUT=== 分隔）",
    )
    parser.add_argument("--json", action="store_true", help="输出原始 JSON")
    parser.add_argument("--top", type=int, default=8, help="文本报告显示前 N 个候选")
    parser.add_argument("--bank-url", default=BANK_URL_DEFAULT, help="指纹库 URL")
    parser.add_argument("--bank-file", type=Path, help="本地指纹库 JSON 路径")
    parser.add_argument("--cache-dir", type=Path, help="指纹库缓存目录")
    parser.add_argument(
        "--save-outputs",
        type=Path,
        help="把原始模型输出存成 JSONL，便于复测/手动归因",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    setup_console()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.bank_file:
            bank = json.loads(Path(args.bank_file).read_text(encoding="utf-8"))
        else:
            bank = load_bank(args.bank_url, args.cache_dir)
    except Exception as error:
        print(f"错误: {error}", file=sys.stderr)
        return 1

    try:
        # 手动：显式 --manual；或未指定 -m 且 stdin 是管道（兼容 < outputs.txt）
        use_manual = args.manual or (args.model is None and not sys.stdin.isatty())
        if use_manual:
            if args.manual and sys.stdin.isatty():
                print(
                    "手动模式：粘贴完整输出，多份用 ===OUTPUT=== 分隔，EOF 结束：",
                    file=sys.stderr,
                )
            pasted = sys.stdin.read()
            outputs = split_manual_outputs(pasted)
            result = analyze_global_outputs(outputs, bank)
        else:
            if args.target < 1 or args.tests < 1 or args.concurrency < 1:
                parser.error("-n/--tests、-c/--concurrency 与 --target 至少为 1")
            result = test_with_codex(
                model=args.model,
                effort=args.reasoning_effort,
                target_count=args.target,
                max_attempts=max(args.tests, args.target),
                bank=bank,
                concurrency=args.concurrency,
                save_outputs=args.save_outputs,
            )
    except Exception as error:
        print(f"错误: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_report(result, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
