from __future__ import annotations

import pandas as pd
from typing import Iterable


VENUE_MAP = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

def venue_name_from_race_id(race_id: str) -> str:
    rid = str(race_id)
    code = rid[4:6] if len(rid) >= 6 else "??"
    return VENUE_MAP.get(code, code)

def normalize_combo(bet_type: str, combo_tuple: tuple[int, ...]) -> str:
    if bet_type in ["umaren", "wide", "sanrenpuku"]:
        return "-".join(map(str, sorted(combo_tuple)))
    else:
        return "-".join(map(str, combo_tuple))


def actual_outcomes_from_results_map(results_map: dict[tuple[str, int], int], rid: str):
    inv = {}
    for (race_id, um), rk in results_map.items():
        if str(race_id) == str(rid):
            try:
                inv[int(rk)] = int(um)
            except Exception:
                pass

    a1, a2, a3 = inv.get(1), inv.get(2), inv.get(3)
    if a1 is None or a2 is None or a3 is None:
        return None

    return {
        "tansho": (a1,),
        "fukusho_set": {a1, a2, a3},  # ★複勝
        "umaren": tuple(sorted((a1, a2))),
        "umatan": (a1, a2),
        "wide_12": tuple(sorted((a1, a2))),
        "wide_13": tuple(sorted((a1, a3))),
        "wide_23": tuple(sorted((a2, a3))),
        "sanrenpuku": tuple(sorted((a1, a2, a3))),
        "sanrentan": (a1, a2, a3),
    }


def payout_map_for_race(payouts_flat_df: pd.DataFrame, date: str, rid: str) -> dict[tuple[str, str], int]:
    """
    payouts_flat_df columns: date,race_id,bet_type,combo,pay100
    戻り: (bet_type, combo)->pay100
    """
    x = payouts_flat_df
    x = x[(x["date"].astype(str) == str(date)) & (x["race_id"].astype(str) == str(rid))].copy()
    out = {}
    for r in x.itertuples(index=False):
        out[(str(r.bet_type), str(r.combo))] = int(r.pay100)
    return out


def eval_one_race_tickets(
    rid: str,
    tickets: list[tuple[str, tuple[int, ...]]],
    outcomes: dict | None,
    pay_map: dict[tuple[str, str], int],
    stake_per_ticket: int = 100,
) -> dict:
    if outcomes is None:
        return {
            "race_id": rid,
            "tickets": len(tickets),
            "stake": stake_per_ticket * len(tickets),
            "return": 0.0,
            "hit_any": 0,
            "hit_tickets": 0,
        }

    stake = stake_per_ticket * len(tickets)
    ret = 0.0
    hit_tickets = 0

    for bt, cmb in tickets:
        cmb_key = normalize_combo(bt, cmb)

        hit = False
        if bt == "tansho":
            hit = (cmb == outcomes["tansho"])

        elif bt == "fukusho":
            try:
                u = int(cmb[0])
            except Exception:
                u = None
            hit = (u is not None and u in outcomes["fukusho_set"])

        elif bt == "umaren":
            hit = (cmb_key == normalize_combo("umaren", outcomes["umaren"]))

        elif bt == "umatan":
            hit = (cmb == outcomes["umatan"])

        elif bt == "wide":
            hit = (cmb_key in {
                normalize_combo("wide", outcomes["wide_12"]),
                normalize_combo("wide", outcomes["wide_13"]),
                normalize_combo("wide", outcomes["wide_23"]),
            })

        elif bt == "sanrenpuku":
            hit = (cmb_key == normalize_combo("sanrenpuku", outcomes["sanrenpuku"]))

        elif bt == "sanrentan":
            hit = (cmb == outcomes["sanrentan"])

        else:
            raise ValueError(f"unknown bet_type: {bt}")

        if hit:
            hit_tickets += 1
            pay100 = pay_map.get((bt, cmb_key), 0)
            ret += float(pay100) * (stake_per_ticket / 100.0)

    return {
        "race_id": rid,
        "tickets": len(tickets),
        "stake": int(stake),
        "return": float(ret),
        "hit_any": int(hit_tickets > 0),
        "hit_tickets": int(hit_tickets),
    }


def summarize_eval_df(eval_df: pd.DataFrame) -> dict:
    if eval_df.empty:
        return {
            "races": 0, "tickets": 0,
            "hit_rate_race": 0.0,
            "hit_rate_ticket": 0.0,
            "stake": 0, "return": 0.0,
            "roi": 0.0,
        }
    races = len(eval_df)
    tickets = int(eval_df["tickets"].sum())
    stake = int(eval_df["stake"].sum())
    ret = float(eval_df["return"].sum())
    hit_rate_race = float(eval_df["hit_any"].mean()) if races > 0 else 0.0
    hit_rate_ticket = (float(eval_df["hit_tickets"].sum()) / max(tickets, 1)) if tickets > 0 else 0.0
    roi = (ret / max(stake, 1)) * 100.0 if stake > 0 else 0.0
    return {
        "races": races,
        "tickets": tickets,
        "hit_rate_race": hit_rate_race,
        "hit_rate_ticket": hit_rate_ticket,
        "stake": stake,
        "return": ret,
        "roi": roi,
    }
