# strategies.py
from __future__ import annotations

from itertools import combinations, permutations
from typing import Dict, List, Tuple

Ticket = tuple[str, tuple[int, ...]]  # ("bet_type", (umaban,...))


def _uniq_ints(xs) -> list[int]:
    out = []
    for x in xs:
        try:
            v = int(x)
        except Exception:
            continue
        if v not in out:
            out.append(v)
    return out


# ---------------------------
# 単勝（あなた指定）
# ---------------------------
def pick_tansho_choice(pred_order: list[int], odds_map: Dict[int, float], mode: str) -> int | None:
    """
    mode:
      - "p1": 予測1位
      - "minodds_top3": 予測Top3の中で最小オッズ
      - "minodds_top5": 予測Top5の中で最小オッズ
    """
    pred_order = _uniq_ints(pred_order)
    if not pred_order:
        return None

    if mode == "p1":
        return pred_order[0]

    if mode == "minodds_top3":
        cand = pred_order[:3]
    elif mode == "minodds_top5":
        cand = pred_order[:5]
    else:
        raise ValueError(mode)

    cand2 = [u for u in cand if u in odds_map and odds_map[u] is not None]
    if not cand2:
        return None
    return min(cand2, key=lambda u: float(odds_map[u]))


def gen_tansho(pred_order: list[int], odds_map: Dict[int, float], mode: str) -> list[Ticket]:
    u = pick_tansho_choice(pred_order, odds_map, mode)
    return [("tansho", (u,))] if u is not None else []


# ---------------------------
# 2頭券：BOX / 1位固定流し
# ---------------------------
def gen_umaren_box(pred_order: list[int], n: int) -> list[Ticket]:
    top = _uniq_ints(pred_order)[:n]
    return [("umaren", tuple(sorted(x))) for x in combinations(top, 2)]

def gen_umatan_box(pred_order: list[int], n: int) -> list[Ticket]:
    top = _uniq_ints(pred_order)[:n]
    return [("umatan", x) for x in permutations(top, 2)]

def gen_wide_box(pred_order: list[int], n: int) -> list[Ticket]:
    top = _uniq_ints(pred_order)[:n]
    return [("wide", tuple(sorted(x))) for x in combinations(top, 2)]

def gen_nagashi_2(pred_order: list[int], n: int, bet_type: str) -> list[Ticket]:
    """
    1位固定の流し（2頭券）
      bet_type: "umaren" | "umatan" | "wide"
      n: 4 or 5（Top4 / Top5）
    """
    top = _uniq_ints(pred_order)[:n]
    if len(top) < 2:
        return []
    p1 = top[0]
    cands = top[1:]
    out: list[Ticket] = []
    for u in cands:
        if bet_type in ["umaren", "wide"]:
            out.append((bet_type, tuple(sorted((p1, u)))))
        elif bet_type == "umatan":
            out.append((bet_type, (p1, u)))
        else:
            raise ValueError(bet_type)
    return out


# ---------------------------
# 3頭券：三連複 / 三連単
# ---------------------------
def gen_sanrenpuku_box(pred_order: list[int], n: int) -> list[Ticket]:
    top = _uniq_ints(pred_order)[:n]
    return [("sanrenpuku", tuple(sorted(x))) for x in combinations(top, 3)]

def gen_sanrentan_box(pred_order: list[int], n: int) -> list[Ticket]:
    top = _uniq_ints(pred_order)[:n]
    return [("sanrentan", x) for x in permutations(top, 3)]

def gen_sanrentan_form_1_234_234(pred_order: list[int]) -> list[Ticket]:
    """
    1位 - 2,3,4位 - 2,3,4位
    """
    top = _uniq_ints(pred_order)[:4]
    if len(top) < 4:
        return []
    p1 = top[0]
    pool = top[1:4]  # p2,p3,p4
    out: list[Ticket] = []
    for a in pool:
        for b in pool:
            if a == b:
                continue
            out.append(("sanrentan", (p1, a, b)))
    return out

def gen_sanrentan_form_1_23_2345(pred_order: list[int]) -> list[Ticket]:
    """
    1位 - 2,3位 - 2,3,4,5位
    """
    top = _uniq_ints(pred_order)[:5]
    if len(top) < 5:
        return []
    p1 = top[0]
    pool2 = top[1:3]    # p2,p3
    pool3 = top[1:5]    # p2,p3,p4,p5
    out: list[Ticket] = []
    for a in pool2:
        for b in pool3:
            if a == b:
                continue
            out.append(("sanrentan", (p1, a, b)))
    return out


# ---------------------------
# まとめ：あなたが欲しい戦略セットを一括生成
# ---------------------------
def build_strategy_catalog() -> list[tuple[str, callable]]:
    """
    返り値: [(strategy_name, generator_fn), ...]
    generator_fn(pred_order, odds_map) -> tickets
    """
    catalog: list[tuple[str, callable]] = []

    # 単勝（3種）
    catalog.append(("tansho_p1", lambda pred, odds: gen_tansho(pred, odds, "p1")))
    catalog.append(("tansho_minodds_top3", lambda pred, odds: gen_tansho(pred, odds, "minodds_top3")))
    catalog.append(("tansho_minodds_top5", lambda pred, odds: gen_tansho(pred, odds, "minodds_top5")))

    # 馬連
    catalog.append(("umaren_box4", lambda pred, odds: gen_umaren_box(pred, 4)))
    catalog.append(("umaren_box5", lambda pred, odds: gen_umaren_box(pred, 5)))
    catalog.append(("umaren_nagashi4", lambda pred, odds: gen_nagashi_2(pred, 4, "umaren")))
    catalog.append(("umaren_nagashi5", lambda pred, odds: gen_nagashi_2(pred, 5, "umaren")))

    # 馬単
    catalog.append(("umatan_box4", lambda pred, odds: gen_umatan_box(pred, 4)))
    catalog.append(("umatan_box5", lambda pred, odds: gen_umatan_box(pred, 5)))
    catalog.append(("umatan_nagashi4", lambda pred, odds: gen_nagashi_2(pred, 4, "umatan")))
    catalog.append(("umatan_nagashi5", lambda pred, odds: gen_nagashi_2(pred, 5, "umatan")))

    # ワイド
    catalog.append(("wide_box4", lambda pred, odds: gen_wide_box(pred, 4)))
    catalog.append(("wide_box5", lambda pred, odds: gen_wide_box(pred, 5)))
    catalog.append(("wide_nagashi4", lambda pred, odds: gen_nagashi_2(pred, 4, "wide")))
    catalog.append(("wide_nagashi5", lambda pred, odds: gen_nagashi_2(pred, 5, "wide")))

    # 三連複
    catalog.append(("sanrenpuku_box4", lambda pred, odds: gen_sanrenpuku_box(pred, 4)))
    catalog.append(("sanrenpuku_box5", lambda pred, odds: gen_sanrenpuku_box(pred, 5)))

    # 三連単
    catalog.append(("sanrentan_box4", lambda pred, odds: gen_sanrentan_box(pred, 4)))
    catalog.append(("sanrentan_box5", lambda pred, odds: gen_sanrentan_box(pred, 5)))
    catalog.append(("sanrentan_1-234-234", lambda pred, odds: gen_sanrentan_form_1_234_234(pred)))
    catalog.append(("sanrentan_1-23-2345", lambda pred, odds: gen_sanrentan_form_1_23_2345(pred)))

    return catalog
