from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from train_strategy_selector import TrainerStrategySelector, build_prediction_cache_for_date_offline
from recommend_strategy import (
    LabelConfig,
    load_results_map,
    load_payouts_df,
    predict_best_strategy_for_race,
)
from strategies import build_strategy_catalog
from eval_bets import actual_outcomes_from_results_map, payout_map_for_race, eval_one_race_tickets


MODEL_DIR = Path("..", "data") / "03_train"
OUT_YAML = MODEL_DIR / "rec_threshold.yaml"


@dataclass
class CalibConfig:
    # キャリブレーション期間
    calib_start: str = "2024-01-01"
    calib_end: str   = "2024-12-31"

    # 参加数が少ない閾値を避ける
    min_races: int = 300

    # しきい値候補（0〜1）
    thresholds: list[float] = None

    # stake
    stake_per_ticket: int = 100

    # 選び方: "profit_max" or "min_threshold_positive"
    select_rule: str = "profit_max"


def _catalog_dict():
    return {name: gen for name, gen in build_strategy_catalog()}


def _normalize_ticket(ticket):
    if isinstance(ticket, dict):
        return str(ticket.get("bet_type", "")), str(ticket.get("combo", "")), ticket
    if isinstance(ticket, (tuple, list)) and len(ticket) >= 2:
        return str(ticket[0]), str(ticket[1]), ticket
    return "", str(ticket), ticket


def _pick_one_ticket(tickets):
    # 1位（先頭）を代表にする：戦略側の並びが「推奨順」になっている前提
    # もし違うなら、ここを「最大ticket_scoreの1点」に差し替え可能
    return tickets[0] if tickets else None


def build_df_for_period(cfg: CalibConfig) -> pd.DataFrame:
    tss = TrainerStrategySelector(
        label_cfg=LabelConfig(stake_per_ticket=cfg.stake_per_ticket, alpha_ticket_penalty=0.18, require_hit=False),
    )

    d0 = pd.to_datetime(cfg.calib_start)
    d1 = pd.to_datetime(cfg.calib_end)
    dates = [d for d in tss.trainable_dates_all if d0 <= pd.to_datetime(d) <= d1]

    cat = _catalog_dict()

    rows = []
    for date in dates:
        pred_rank_all, tri_all, pred_order_by_race, odds_map_by_race, meta = build_prediction_cache_for_date_offline(
            date=date,
            features_all=tss.features_all,
            model_tri=tss.model_tri,
            feature_cols=tss.feature_cols,
            cfg=tss.cache_cfg,
        )
        if not meta.get("ok", False) or pred_rank_all.empty:
            continue

        results_map = load_results_map(date)
        payouts_df = load_payouts_df(date)
        if payouts_df is None or payouts_df.empty:
            continue

        rids_from_payouts = set(payouts_df["race_id"].astype(str).unique())
        rids_from_results = set([rid for (rid, _) in results_map.keys()])
        rids_from_pred = set(map(str, pred_order_by_race.keys()))
        rids = sorted(rids_from_payouts & rids_from_results & rids_from_pred)

        for rid in rids:
            rid = str(rid)

            outcomes = actual_outcomes_from_results_map(results_map, rid)
            if not outcomes:
                continue
            pay_map = payout_map_for_race(payouts_df, date, rid)
            if not pay_map:
                continue

            g_rank = pred_rank_all[pred_rank_all["race_id"].astype(str) == rid].copy()
            g_tri  = tri_all[tri_all["race_id"].astype(str) == rid].copy() if ("race_id" in tri_all.columns) else pd.DataFrame()

            # 戦略推定（model_score = pred_proba）
            strat, dbg = predict_best_strategy_for_race(g_rank, g_tri)
            model_score = float(dbg.get("pred_proba", 0.0))

            if strat not in cat:
                continue

            pred_order = pred_order_by_race.get(rid, [])
            odds_map = odds_map_by_race.get(rid, {})

            tickets = cat[strat](pred_order, odds_map)
            if not tickets:
                continue

            one_ticket = _pick_one_ticket(tickets)
            if one_ticket is None:
                continue

            ev = eval_one_race_tickets(
                rid=rid,
                tickets=[one_ticket],          # 代表1点の払戻
                outcomes=outcomes,
                pay_map=pay_map,
                stake_per_ticket=cfg.stake_per_ticket,
            )
            ret = float(ev.get("return", 0.0))
            # あなたのルール：投資は「戦略点数分」
            stake = float(len(tickets) * cfg.stake_per_ticket)
            profit = ret - stake
            hit = 1 if ret > 0 else 0

            rows.append({
                "date": date,
                "race_id": rid,
                "strategy": strat,
                "model_score": model_score,
                "n_points": int(len(tickets)),
                "stake": stake,
                "return": ret,
                "profit": profit,
                "hit": hit,
            })

    return pd.DataFrame(rows)


def sweep_threshold(df: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    out = []
    for t in thresholds:
        d = df[df["model_score"] >= t].copy()
        n_races = int(d["race_id"].nunique())
        invest = float(d["stake"].sum())
        ret = float(d["return"].sum())
        profit = ret - invest
        roi = (ret / invest * 100.0) if invest > 0 else 0.0
        n_hit = int(d["hit"].sum())
        hit_rate = (n_hit / n_races * 100.0) if n_races > 0 else 0.0

        out.append({
            "threshold": float(t),
            "n_races": n_races,
            "n_hit_races": n_hit,
            "hit_rate_race_pct": float(hit_rate),
            "invest": invest,
            "return": ret,
            "profit": profit,
            "roi_pct": float(roi),
        })
    return pd.DataFrame(out)


def select_threshold(df_th: pd.DataFrame, cfg: CalibConfig) -> tuple[float, pd.DataFrame]:
    cand = df_th[df_th["n_races"] >= cfg.min_races].copy()
    if cand.empty:
        # fallback：参加数条件を無視してprofit最大
        best = df_th.sort_values("profit", ascending=False).iloc[0]
        return float(best["threshold"]), cand

    if cfg.select_rule == "min_threshold_positive":
        cand2 = cand.sort_values("threshold").query("profit > 0")
        if len(cand2):
            return float(cand2.iloc[0]["threshold"]), cand

    # default: profit_max
    best = cand.sort_values("profit", ascending=False).iloc[0]
    return float(best["threshold"]), cand


def main(
    calib_start="2024-01-01",
    calib_end="2024-12-31",
    min_races=300,
    select_rule="profit_max",
    stake_per_ticket=100,
):
    cfg = CalibConfig(
        calib_start=calib_start,
        calib_end=calib_end,
        min_races=min_races,
        thresholds=[float(x) for x in np.linspace(0.0, 1.0, 21)],
        select_rule=select_rule,
        stake_per_ticket=stake_per_ticket,
    )

    df = build_df_for_period(cfg)
    if df.empty:
        raise RuntimeError("calibration df is empty. results/payouts/features/model_tri を確認してください。")

    df_th = sweep_threshold(df, cfg.thresholds)
    best_t, cand = select_threshold(df_th, cfg)

    # 保存
    payload = {
        "rec_model_score_th": float(best_t),
        "calib_period": {"start": cfg.calib_start, "end": cfg.calib_end},
        "min_races": int(cfg.min_races),
        "select_rule": cfg.select_rule,
        "stake_per_ticket": int(cfg.stake_per_ticket),
        "summary_at_best": df_th[df_th["threshold"] == best_t].to_dict(orient="records")[0],
        "df_th": df_th.sort_values("threshold").to_dict(orient="records"),
    }

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    print("[OK] saved:", OUT_YAML)
    print("best threshold:", best_t)
    print("best summary:", payload["summary_at_best"])
    return best_t, df_th


if __name__ == "__main__":
    main()
