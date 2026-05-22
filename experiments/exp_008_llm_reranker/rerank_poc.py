"""exp_008_llm_reranker / rerank_poc.py -- Solar API rerank on TIFU top-50.

Pipeline (per user):
    1. TIFU top-50 candidates 로드
    2. user history (cache) + 50 candidate items metadata (cache) 합쳐 prompt 빌드
    3. Solar API call (async + retries)
    4. JSON 응답 parse -> top-10 ranked list (item indices A-Z + AA-AX 매핑)
    5. Invalid response -> TIFU top-10 그대로 (fallback)

PoC scope: 처음 poc_n_users 명에 prompt/parsing 검증 후,
full eval_users 928 batch run.

Output:
    reranked_predictions.parquet   user_id, item_id, score (rank-based), rank
    poc_log.json                   per-user 결과 + LLM raw response + parse status
    eval_summary.json              NDCG@10 vs TIFU baseline + cost

Usage:
    python rerank_poc.py --poc       # 처음 100 user 만 (config 의 poc_n_users)
    python rerank_poc.py             # full eval_users 928 (검증 후)
    python rerank_poc.py --no-wandb  # wandb 건너뜀
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from core.metrics import ndcg_at_k_from_df, recall_at_k_from_df  # noqa: E402

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Prompt template
# ----------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a recommender ranker for an e-commerce site. Re-rank 50 candidate items to predict which the user will purchase in the next week.

USER'S RECENT ACTIVITY (oldest first):
{history}

CANDIDATE ITEMS (from a base ranker, with current rank score):
{candidates}

INSTRUCTIONS:
- Re-order the candidates. Pick the 10 most likely to be purchased next week.
- Consider: brand affinity, price range, recently viewed/carted items, category interest.
- Output ONLY a JSON array of 10 item codes in your ranked order, e.g. ["C03","C12","C01",...].
- No explanation, no markdown, just the JSON array."""


def _make_item_code(idx: int) -> str:
    """0->C00, 1->C01, ..., 49->C49."""
    return f"C{idx:02d}"


def build_prompt(
    history_text: str,
    candidates_df: pd.DataFrame,   # rows: rank, item_id, item_meta cols
) -> tuple[str, dict]:
    """Return (prompt_text, code_to_item_id mapping)."""
    code_map = {}
    cand_lines = []
    for i, (_, row) in enumerate(candidates_df.iterrows()):
        code = _make_item_code(i)
        code_map[code] = row["item_id"]
        cat = row.get("category") or "unknown"
        brand = row.get("brand") or "unknown"
        price = row.get("avg_price")
        price_str = f"${price:.0f}" if pd.notna(price) else "$?"
        score = row.get("score", 0.0)
        cand_lines.append(
            f"{code}: {cat} / {brand} / {price_str} (rank={int(row['rank'])}, score={score:.3f})"
        )
    candidates_block = "\n".join(cand_lines)
    history_block = history_text if history_text else "(no history)"
    prompt = PROMPT_TEMPLATE.format(history=history_block, candidates=candidates_block)
    return prompt, code_map


# ----------------------------------------------------------------------
# Response parsing
# ----------------------------------------------------------------------
JSON_ARRAY_RE = re.compile(r"\[(.*?)\]", re.DOTALL)


def parse_response(text: str, code_map: dict) -> list[str] | None:
    """Extract top-10 item_ids from LLM response. None on parse failure."""
    # Find first JSON-like array
    m = JSON_ARRAY_RE.search(text)
    if not m:
        return None
    body = "[" + m.group(1) + "]"
    try:
        arr = json.loads(body)
    except json.JSONDecodeError:
        # try cleanup: replace single quotes
        try:
            arr = json.loads(body.replace("'", '"'))
        except json.JSONDecodeError:
            return None
    if not isinstance(arr, list):
        return None
    # Map codes -> item_ids, dedup preserving order
    seen = set()
    out: list[str] = []
    for code in arr:
        if not isinstance(code, str):
            continue
        code = code.strip().upper()
        if code not in code_map:
            continue
        item_id = code_map[code]
        if item_id in seen:
            continue
        seen.add(item_id)
        out.append(item_id)
        if len(out) == 10:
            break
    if not out:
        return None
    return out


# ----------------------------------------------------------------------
# Solar API call (async, with retry)
# ----------------------------------------------------------------------
async def solar_call(
    client,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
) -> tuple[str | None, dict]:
    """Return (response_text, usage_dict) or (None, {...}) on failure."""
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            text = resp.choices[0].message.content
            usage = {
                "input_tokens": resp.usage.prompt_tokens,
                "output_tokens": resp.usage.completion_tokens,
            }
            return text, usage
        except Exception as e:    # noqa: BLE001
            last_err = e
            await asyncio.sleep(1.5 ** attempt)
    return None, {"error": str(last_err)}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
async def rerank_user_batch(
    client,
    users: list[str],
    tifu_top50_per_user: dict,        # user_id -> top50 DataFrame
    user_history_map: dict,            # user_id -> history_text
    cfg: dict,
    sem: asyncio.Semaphore,
) -> list[dict]:
    """Async rerank a batch of users."""
    async def _one(u):
        async with sem:
            cand_df = tifu_top50_per_user[u]
            prompt, code_map = build_prompt(user_history_map.get(u, ""), cand_df)
            text, usage = await solar_call(
                client,
                prompt,
                model=cfg["solar_model"],
                temperature=cfg["solar_temperature"],
                max_tokens=cfg["solar_max_tokens"],
                timeout=cfg["solar_timeout_sec"],
                max_retries=cfg["solar_max_retries"],
            )
            parsed = parse_response(text, code_map) if text else None
            return {
                "user_id": u,
                "raw_response": text,
                "parsed": parsed,
                "usage": usage,
                "fallback": parsed is None,
            }

    return await asyncio.gather(*[_one(u) for u in users])


async def main_async(cfg: dict, args) -> None:
    # ---- API client -----------------------------------------------
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("UPSTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("UPSTAGE_API_KEY not in env. Check .env file.")

    from openai import AsyncOpenAI  # noqa: PLC0415
    client = AsyncOpenAI(api_key=api_key, base_url=cfg["solar_base_url"])

    # ---- Load caches -----------------------------------------------
    cache_dir = HERE / "cache"
    if not (cache_dir / "item_metadata.parquet").exists():
        raise RuntimeError("Cache missing. Run prepare_data.py first.")
    meta = pd.read_parquet(cache_dir / "item_metadata.parquet")
    history = pd.read_parquet(cache_dir / "user_history.parquet")
    user_history_map = dict(zip(history["user_id"], history["history_text"]))

    # ---- TIFU top-50 -----------------------------------------------
    tifu = pd.read_parquet((HERE / cfg["tifu_predictions"]).resolve())
    tifu = tifu[tifu["rank"] <= cfg["n_candidates"]]
    # Attach item metadata
    tifu_m = tifu.merge(
        meta[["item_id", "category", "brand", "avg_price"]],
        on="item_id",
        how="left",
    )

    # ---- eval_users ------------------------------------------------
    ease_saved = (HERE / cfg["ease_saved"]).resolve()
    val_gt_df = pd.read_parquet(ease_saved / "val_gt.parquet")
    with open(ease_saved / "eval_users.json", encoding="utf-8") as f:
        eval_users = json.load(f)

    # PoC scope
    if args.poc:
        users = eval_users[: cfg["poc_n_users"]]
        logger.info("PoC scope: %d users", len(users))
    else:
        users = eval_users
        logger.info("Full eval scope: %d users", len(users))

    # Build per-user top50 dict (only for selected users)
    tifu_subset = tifu_m[tifu_m["user_id"].isin(users)].sort_values(
        ["user_id", "rank"], kind="mergesort"
    )
    tifu_top50_per_user = {
        u: g.reset_index(drop=True)
        for u, g in tifu_subset.groupby("user_id", sort=False)
    }
    users = [u for u in users if u in tifu_top50_per_user]   # drop missing
    logger.info("users with TIFU candidates: %d", len(users))

    # ---- Rerank batches --------------------------------------------
    sem = asyncio.Semaphore(cfg["solar_concurrent_calls"])
    results: list[dict] = []
    chunk = 50
    t0 = time.time()
    for start in range(0, len(users), chunk):
        batch = users[start : start + chunk]
        batch_res = await rerank_user_batch(
            client, batch, tifu_top50_per_user, user_history_map, cfg, sem
        )
        results.extend(batch_res)
        n_done = len(results)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        n_fallback = sum(1 for r in results if r["fallback"])
        logger.info(
            "  %d/%d (%.1f%%), %.1fs, %.1f users/s, fallback=%d (%.1f%%)",
            n_done, len(users), 100 * n_done / len(users),
            elapsed, rate, n_fallback, 100 * n_fallback / n_done,
        )

    total_elapsed = time.time() - t0
    logger.info("rerank done in %.1fs", total_elapsed)

    # ---- Build reranked predictions.parquet ------------------------
    rows = []
    for r in results:
        u = r["user_id"]
        if r["parsed"]:
            items = r["parsed"]
        else:
            # Fallback to TIFU top-10
            items = tifu_top50_per_user[u].head(10)["item_id"].tolist()
        for rank, it in enumerate(items, start=1):
            rows.append({
                "user_id": u,
                "item_id": it,
                "score": float(11 - rank),    # rank-derived score (10..1)
                "rank": rank,
            })
    rerank_df = pd.DataFrame(rows)
    out_path = HERE / "reranked_predictions.parquet"
    rerank_df.to_parquet(out_path)
    logger.info("wrote %s (%d rows, %d users)",
                out_path, len(rerank_df), rerank_df["user_id"].nunique())

    # ---- Save raw log ----------------------------------------------
    log_path = HERE / "poc_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    logger.info("wrote raw log %s", log_path)

    # ---- Eval vs TIFU baseline -------------------------------------
    val_gt_eval = val_gt_df[val_gt_df["user_id"].isin(users)]
    ndcg_rerank = ndcg_at_k_from_df(rerank_df, val_gt_eval, k=10)
    recall_rerank = recall_at_k_from_df(rerank_df, val_gt_eval, k=10)

    # TIFU baseline on the same users
    tifu_subset_top10 = tifu_subset[tifu_subset["rank"] <= 10]
    ndcg_tifu = ndcg_at_k_from_df(tifu_subset_top10, val_gt_eval, k=10)
    recall_tifu = recall_at_k_from_df(tifu_subset_top10, val_gt_eval, k=10)

    # Cost estimate
    total_in = sum(r["usage"].get("input_tokens", 0) for r in results)
    total_out = sum(r["usage"].get("output_tokens", 0) for r in results)
    # Approximate Solar pricing -- to be confirmed by upstage dashboard
    # solar-mini ballpark: $0.0001/$0.0001 per 1k tokens (assumption, verify)
    cost_est = (total_in / 1000) * 0.0001 + (total_out / 1000) * 0.0001

    summary = {
        "n_users": len(users),
        "n_fallback": sum(1 for r in results if r["fallback"]),
        "fallback_pct": float(sum(1 for r in results if r["fallback"]) / len(users)),
        "elapsed_sec": total_elapsed,
        "users_per_sec": len(users) / total_elapsed,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "cost_estimate_usd": cost_est,
        "ndcg10_rerank": ndcg_rerank,
        "ndcg10_tifu_baseline": ndcg_tifu,
        "ndcg10_delta": ndcg_rerank - ndcg_tifu,
        "recall10_rerank": recall_rerank,
        "recall10_tifu_baseline": recall_tifu,
        "model": cfg["solar_model"],
    }
    sum_path = HERE / "eval_summary.json"
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("eval summary:")
    logger.info("  TIFU baseline NDCG@10  : %.4f", ndcg_tifu)
    logger.info("  Rerank NDCG@10         : %.4f", ndcg_rerank)
    logger.info("  Δ NDCG                 : %+.4f", ndcg_rerank - ndcg_tifu)
    logger.info("  TIFU baseline recall@10: %.4f", recall_tifu)
    logger.info("  Rerank recall@10       : %.4f", recall_rerank)
    logger.info("  fallback users         : %d (%.1f%%)",
                summary["n_fallback"], summary["fallback_pct"] * 100)
    logger.info("  tokens (in/out)        : %s / %s",
                f"{total_in:,}", f"{total_out:,}")
    logger.info("  cost estimate          : $%.4f", cost_est)

    await client.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.yaml"))
    parser.add_argument("--poc", action="store_true",
                        help="PoC scope (poc_n_users) instead of full eval_users")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    asyncio.run(main_async(cfg, args))


if __name__ == "__main__":
    main()
