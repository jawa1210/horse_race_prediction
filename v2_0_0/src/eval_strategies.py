# eval_strategies.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from feature_engineering import RAW_DATA_DIR, DATA_DIR
from eval_bets import (
    venue_name_from_race_id,
    actual_outcomes_from_results_map,
    payout_map_for_race,
    eval_one_race_tickets,
    summarize_eval_df,
)
from strategies import build_strategy_catalog


def default_paths() -> dict[str, Path]:
    return {
        "population_csv": (RAW_DATA_DIR / "prediction_population" / "population.csv").resolve(),
        "results_flat": (DATA_DIR / "results_cache" / "results_flat.csv").resolve(),
        "payouts_flat": (DATA_DIR / "results_cache" / "payouts_flat.csv").resolve(),
        "eval_log": (DATA_DIR / "results_cache" / "eval_log.csv").resolve(),
    }


def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _load_results_map_from_flat(results_flat: Path, target_date: str) -> dict[tuple[str, int], int]:
    if not results_flat.exists() or results_flat.stat().st_size == 0:
        return {}
    df = pd.read_csv(results_flat, dtype={"race_id": str})
    need = {"date", "race_id", "umaban", "rank"}
    if not need.issubset(df.columns):
        return {}
    df = df[df["date"].astype(str) == str(target_date)].copy()
    df["race_id"] = df["race_id"].astype(str)
    df["umaban"] = pd.to_numeric(df["umaban"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.dropna(subset=["race_id", "umaban", "rank"])
    out = {}
    for r in df.itertuples(index=False):
        out[(str(r.race_id), int(r.umaban))] = int(r.rank)
    return out


def eval_strategies_for_date(
    target_date: str,
    race_ids: list[str],
    pred_order_by_race: dict[str, list[int]],
    odds_map_by_race: dict[str, dict[int, float]],
    stake_per_ticket: int = 100,
    results_flat: Path | None = None,
    payouts_flat: Path | None = None,
    eval_log: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    pred_order_by_race[rid] = [予測1位,2位,3位... の馬番]
    odds_map_by_race[rid] = {umaban: odds}
    戻り:
      - per_race_eval: レース×戦略の明細
      - summary_by_venue: 開催場所別サマリ
      - summary_total: 全体サマリ
    """
    paths = default_paths()
    results_flat = (results_flat or paths["results_flat"]).resolve()
    payouts_flat = (payouts_flat or paths["payouts_flat"]).resolve()
    eval_log = (eval_log or paths["eval_log"]).resolve()

    # 必須CSVチェック
    if not results_flat.exists():
        raise FileNotFoundError(f"results_flat not found: {results_flat}")
    if not payouts_flat.exists():
        raise FileNotFoundError(f"payouts_flat not found: {payouts_flat}")

    payouts_df = pd.read_csv(payouts_flat, dtype={"race_id": str, "bet_type": str, "combo": str})
    results_map = _load_results_map_from_flat(results_flat, target_date)

    catalog = build_strategy_catalog()

    rows = []
    for rid in race_ids:
        pred_order = pred_order_by_race.get(rid, [])
        odds_map = odds_map_by_race.get(rid, {})
        venue = venue_name_from_race_id(rid)

        outcomes = actual_outcomes_from_results_map(results_map, rid)
        pay_map = payout_map_for_race(payouts_df, target_date, rid)

        for strat_name, gen in catalog:
            tickets = gen(pred_order, odds_map)
            ev = eval_one_race_tickets(
                rid=rid,
                tickets=tickets,
                outcomes=outcomes,
                pay_map=pay_map,
                stake_per_ticket=stake_per_ticket,
            )
            ev.update({
                "date": str(target_date),
                "venue": venue,
                "strategy": strat_name,
            })
            rows.append(ev)

    per_race_eval = pd.DataFrame(rows)

    # ---- サマリ作成
    def _summarize_group(df: pd.DataFrame) -> dict:
        s = summarize_eval_df(df)
        return {
            "races": s["races"],
            "tickets": s["tickets"],
            "hit_rate_race": s["hit_rate_race"],
            "hit_rate_ticket": s["hit_rate_ticket"],
            "stake": s["stake"],
            "return": s["return"],
            "roi": s["roi"],
        }

    # strategy × venue
    g1 = []
    for (venue, strategy), g in per_race_eval.groupby(["venue", "strategy"]):
        d = _summarize_group(g)
        d.update({"venue": venue, "strategy": strategy})
        g1.append(d)
    summary_by_venue = pd.DataFrame(g1)

    # strategy total
    g2 = []
    for strategy, g in per_race_eval.groupby(["strategy"]):
        d = _summarize_group(g)
        d.update({"venue": "全体", "strategy": strategy})
        g2.append(d)
    summary_total = pd.DataFrame(g2)

    # ---- 累計ログ保存（date,race_id,strategy の重複は最後勝ち）
    _ensure_parent(eval_log)
    if eval_log.exists() and eval_log.stat().st_size > 0:
        old = pd.read_csv(eval_log, dtype={"race_id": str, "strategy": str, "date": str})
    else:
        old = pd.DataFrame(columns=per_race_eval.columns)

    merged = pd.concat([old, per_race_eval], ignore_index=True)
    merged = merged.drop_duplicates(subset=["date", "race_id", "strategy"], keep="last")
    merged.to_csv(eval_log, index=False, encoding="utf-8")

    return per_race_eval, summary_by_venue, summary_total


def load_cumulative_summary(eval_log: Path | None = None) -> pd.DataFrame:
    """
    これまでの全予測（eval_log.csv）から、strategy別の累計サマリを返す
    """
    paths = default_paths()
    eval_log = (eval_log or paths["eval_log"]).resolve()
    if not eval_log.exists() or eval_log.stat().st_size == 0:
        return pd.DataFrame()

    df = pd.read_csv(eval_log, dtype={"race_id": str, "strategy": str, "date": str})

    out = []
    for strategy, g in df.groupby("strategy"):
        s = summarize_eval_df(g)
        out.append({
            "strategy": strategy,
            "races": s["races"],
            "tickets": s["tickets"],
            "hit_rate_race": s["hit_rate_race"],
            "hit_rate_ticket": s["hit_rate_ticket"],
            "stake": s["stake"],
            "return": s["return"],
            "roi": s["roi"],
        })
    return pd.DataFrame(out).sort_values("roi", ascending=False).reset_index(drop=True)


def format_summary_df_for_html(df: pd.DataFrame) -> pd.DataFrame:
    """
    表示用に % 表記に直す
    """
    if df is None or df.empty:
        return df
    x = df.copy()
    if "hit_rate_race" in x.columns:
        x["hit_rate_race"] = (x["hit_rate_race"] * 100).map(lambda v: f"{v:.1f}%")
    if "hit_rate_ticket" in x.columns:
        x["hit_rate_ticket"] = (x["hit_rate_ticket"] * 100).map(lambda v: f"{v:.2f}%")
    if "roi" in x.columns:
        x["roi"] = x["roi"].map(lambda v: f"{float(v):.1f}%")
    if "return" in x.columns:
        x["return"] = x["return"].map(lambda v: f"{float(v):.0f}")
    return x
