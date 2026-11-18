#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
cross_engine.py — 属性どうしをクロスする汎用エンジン

目的:
  任意の属性（年収・世帯人員・子の有無・ペット・品目・金額・時期・業態…）を
  指定して、同時出現率や条件付き分布を、実データに裏づけて出す。

2つの原則:
  1. 独立とみなして掛け算しない。
     関係（conditional / metric）が登録されていればそれを使う。
     無ければ独立を仮定し、『独立仮定を使った』と結果に必ず明示する。
  2. 裏づけの出どころを全部残す。
     どの因子が1次データで、どこが2次・推計・独立仮定かを結果に付ける。

3つの問い方:
  occurrence  … 条件をすべて満たす世帯の割合  P(A=a かつ B=b かつ …)
  conditional … ある属性を固定して別の属性の分布  P(target | 条件)
  metric      … ある属性を固定して金額・数量の推計値  E[金額_品目 | 条件]

固有名詞（店名・商品ブランド・固有の地名）は groundability で proper_noun=false。
エンジンは業態・品目・都市階級までしか返さず、固有名は『創作』と明示する。

使い方（PowerShell）:
  python cross_engine.py occurrence --where has_child=あり household_size=2 pet=犬
  python cross_engine.py conditional --target pet --where household_size=1
  python cross_engine.py metric --item 外食 --where income_quintile=V
  python cross_engine.py ground 店名
  python cross_engine.py map
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REG_PATH = HERE / "attributes.json"

C_RED, C_YEL, C_CYA, C_GRY, C_BLD, C_END = (
    "\033[91m", "\033[93m", "\033[96m", "\033[90m", "\033[1m", "\033[0m"
)
TIER_COLOR = {"1次データ": C_CYA, "2次データ": C_YEL, "推計": C_RED,
              "独立仮定": C_RED, "該当なし": C_GRY}


def load_registry():
    return json.loads(REG_PATH.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────
# 因子の取り出し
#   各クロスは「因子（factor）」の積で表す。
#   因子ごとに value（確率）と provenance（出どころ）を持たせ、
#   最後に出どころを全部並べて表示できるようにする。
# ─────────────────────────────────────────────────────────
class Factor:
    def __init__(self, desc, value, tier, source):
        self.desc = desc        # 人が読む説明
        self.value = value      # 確率（0〜1）
        self.tier = tier        # 1次データ / 2次データ / 推計 / 独立仮定
        self.source = source


def marginal_prob(reg, attr, val):
    """属性 attr が値 val を取る周辺確率を、出どころ付きで返す"""
    a = reg["attributes"].get(attr)
    if a is None:
        raise KeyError(f"未登録の属性です: {attr}")
    m = a.get("marginal")
    if not m or val not in m:
        raise KeyError(f"{attr} の周辺分布に {val} がありません")
    return Factor(f"P({attr}={val})", m[val], a["tier"], a["source"])


def conditional_table(reg, given_attr, target_attr):
    """P(target | given) の表を探す。無ければ None。"""
    for rel in reg["relationships"].values():
        if (rel.get("type") == "conditional"
                and rel.get("given") == given_attr
                and rel.get("target") == target_attr
                and rel.get("table")):
            return rel
    return None


def pet_value_prob(reg, dist, pet_val):
    """
    ペットは『犬・猫・なし』が排他でないため専用に扱う。
    犬または猫＝犬+猫-両方。両方は overlap の仮定で決める。
    """
    if pet_val in ("犬", "猫"):
        return dist[pet_val]
    if pet_val in ("ペットあり", "犬または猫", "あり"):
        dog, cat = dist.get("犬", 0.0), dist.get("猫", 0.0)
        ov = next((r for r in reg["relationships"].values()
                   if r.get("type") == "overlap"), None)
        mode = ov["assumption"] if ov else "independent"
        both = dog * cat if mode == "independent" else min(dog, cat) * 0.5
        return dog + cat - both
    if pet_val in ("なし", "ペットなし"):
        return dist.get("なし", 1 - dist.get("犬", 0) - dist.get("猫", 0))
    return dist.get(pet_val, 0.0)


# ─────────────────────────────────────────────────────────
# occurrence : 条件をすべて満たす割合
# ─────────────────────────────────────────────────────────
def occurrence(reg, where):
    """
    where は {属性: 値} の辞書。
    条件を『連鎖』で掛ける：
      P(A,B,C) = P(A) × P(B|A) × P(C|A,B)
    各段で使える条件付き関係があればそれを、無ければ周辺分布（＝独立仮定）を使う。
    どちらを使ったかは factor の tier でわかる。
    """
    factors = []
    assumed_independent = []
    resolved = []  # すでに条件に入れた属性

    for attr, val in where.items():
        rel = None
        # すでに確定した属性のどれかを given に持つ条件付き表があれば使う
        for g in resolved:
            cand = conditional_table(reg, g, attr)
            if cand:
                rel = (g, cand)
                break

        if rel:
            given_attr, table = rel
            given_val = where[given_attr]
            dist = table["table"][given_val]
            if attr == "pet":
                p = pet_value_prob(reg, dist, val)
            else:
                p = dist.get(val, 0.0)
            factors.append(Factor(
                f"P({attr}={val} | {given_attr}={given_val})",
                p, table["tier"], table["source"]))
        else:
            # 条件付き関係が無い → 周辺分布（独立仮定）
            a = reg["attributes"][attr]
            if attr == "pet":
                p = pet_value_prob(reg, a["marginal"], val)
            else:
                p = marginal_prob(reg, attr, val).value
            tier = a["tier"] if len(resolved) == 0 else "独立仮定"
            src = (a["source"] if len(resolved) == 0
                   else f"{attr}と既出条件の関係が未登録のため独立を仮定（{a['source']}）")
            factors.append(Factor(f"P({attr}={val})", p, tier, src))
            if len(resolved) > 0:
                assumed_independent.append(attr)

        resolved.append(attr)

    joint = 1.0
    for f in factors:
        joint *= f.value
    return {"joint": joint, "factors": factors,
            "assumed_independent": assumed_independent}


# ─────────────────────────────────────────────────────────
# conditional : ある条件のもとでの target の分布
# ─────────────────────────────────────────────────────────
def conditional(reg, target, where):
    """P(target | where) を返す。関係があれば使い、無ければ周辺（独立）。"""
    for g, gv in where.items():
        table = conditional_table(reg, g, target)
        if table:
            dist = dict(table["table"][gv])
            return {"dist": dist, "tier": table["tier"],
                    "source": table["source"],
                    "given": f"{g}={gv}", "assumed_independent": False}
    a = reg["attributes"].get(target)
    if a and a.get("marginal"):
        return {"dist": dict(a["marginal"]), "tier": a["tier"],
                "source": a["source"], "given": "（条件なし＝周辺分布）",
                "assumed_independent": bool(where)}
    raise KeyError(f"{target} の分布を解決できません")


# ─────────────────────────────────────────────────────────
# metric : 金額・数量の推計（品目×世帯属性）
# ─────────────────────────────────────────────────────────
def metric(reg, item, where):
    """E[item | where] を家計調査系の metric_table から返す"""
    for rel in reg["relationships"].values():
        if rel.get("type") != "metric_table":
            continue
        if item not in rel.get("items", {}):
            continue
        given = rel["given"]
        if given not in where:
            raise KeyError(
                f"{item} の金額を出すには {given} の指定が必要です "
                f"（--where {given}=... を付けてください）")
        gv = where[given]
        table = rel["items"][item]
        if gv not in table:
            raise KeyError(f"{given}={gv} は {item} の表にありません")
        return {"item": item, "value": table[gv], "unit": rel["unit"],
                "given": f"{given}={gv}", "tier": rel["tier"],
                "source": rel["source"], "as_of": rel.get("as_of", ""),
                "full_row": table}
    raise KeyError(f"{item} の金額・数量データが未登録です")


# ─────────────────────────────────────────────────────────
# groundability : その属性がどこまで裏づけられるか
# ─────────────────────────────────────────────────────────
def groundability(reg, attr):
    a = reg["attributes"].get(attr)
    if not a:
        raise KeyError(f"未登録の属性です: {attr}")
    g = a.get("groundability", {})
    return {"attr": attr, "label": a.get("label", attr),
            "tier": a["tier"], "level": g.get("level", "不明"),
            "proper_noun": g.get("proper_noun", False),
            "source": a.get("source", "")}


# ─────────────────────────────────────────────────────────
# 表示
# ─────────────────────────────────────────────────────────

def pad(s, width):
    """全角2・半角1で幅を数えて右側を空白で埋める"""
    w = sum(2 if ord(ch) > 0x2000 else 1 for ch in s)
    return s + " " * max(0, width - w)


def tcol(tier):
    return TIER_COLOR.get(tier, C_GRY)


def show_factors(factors):
    print(f"  {'因子':<34}{'確率':>9}  出どころ")
    print("  " + "-" * 76)
    for f in factors:
        c = tcol(f.tier)
        print(f"  {f.desc:<34}{f.value*100:>8.2f}%  {c}[{f.tier}]{C_END}")
        print(f"  {'':<34}{'':>9}  {C_GRY}{f.source[:60]}{C_END}")


def cmd_occurrence(reg, args):
    where = parse_where(args.where)
    r = occurrence(reg, where)
    cond = " かつ ".join(f"{k}={v}" for k, v in where.items())
    print()
    print("=" * 80)
    print(f"{C_BLD}同時出現率: {cond}{C_END}")
    print("=" * 80)
    show_factors(r["factors"])
    print("  " + "-" * 76)
    print(f"  {C_BLD}同時出現率 = {r['joint']*100:.3f}%{C_END}  "
          f"（1000世帯あたり約 {r['joint']*1000:.1f} 世帯）")
    if r["assumed_independent"]:
        print()
        print(f"  {C_RED}■ 独立仮定を使った属性: "
              f"{', '.join(r['assumed_independent'])}{C_END}")
        print(f"  {C_GRY}これらは既出条件との関係が未登録のため、独立とみなして掛けています。")
        print(f"  関係を attributes.json の relationships に足すと精度が上がります。{C_END}")
    print("=" * 80)
    print()


def cmd_conditional(reg, args):
    where = parse_where(args.where)
    r = conditional(reg, args.target, where)
    print()
    print("=" * 80)
    print(f"{C_BLD}条件付き分布: P({args.target} | {r['given']}){C_END}")
    print("=" * 80)
    items = sorted(r["dist"].items(), key=lambda kv: kv[1], reverse=True)
    for k, v in items:
        bar = "█" * max(0, round(v * 40))
        print(f"  {k:<16}{v*100:>7.2f}%  {C_CYA}{bar}{C_END}")
    print("  " + "-" * 76)
    print(f"  出どころ: {tcol(r['tier'])}[{r['tier']}]{C_END} {r['source'][:60]}")
    if r["assumed_independent"]:
        print(f"  {C_RED}※ 条件と {args.target} の関係が未登録のため、"
              f"周辺分布（＝独立）を返しています{C_END}")
    print("=" * 80)
    print()


def cmd_metric(reg, args):
    where = parse_where(args.where)
    r = metric(reg, args.item, where)
    print()
    print("=" * 80)
    print(f"{C_BLD}金額・数量の推計: {r['item']}（{r['given']}）{C_END}")
    print("=" * 80)
    print(f"  推計値: {C_BLD}{r['value']:,} {r['unit']}{C_END}")
    print()
    print(f"  {C_GRY}年収階級ごとの全体像:{C_END}")
    for k, v in r["full_row"].items():
        mark = " ◀" if f"{list(where.values())[0]}" == k or where.get(
            r['given'].split('=')[0]) == k else ""
        print(f"    {k:<5}{v:>9,} {r['unit']}{mark}")
    print("  " + "-" * 76)
    print(f"  出どころ: {tcol(r['tier'])}[{r['tier']}]{C_END} {r['source'][:66]}")
    print(f"  {C_YEL}注: これは世帯平均の月額です。個人の1回の購入額ではありません。")
    print(f"  会話で『いくら使った』と言わせるなら、この平均を中心に散らす必要があります。{C_END}")
    print("=" * 80)
    print()


def cmd_ground(reg, args):
    g = groundability(reg, args.attr)
    print()
    print("=" * 80)
    print(f"{C_BLD}裏づけの限界: {g['label']}（{g['attr']}）{C_END}")
    print("=" * 80)
    print(f"  出どころの種別 : {tcol(g['tier'])}[{g['tier']}]{C_END}")
    print(f"  裏づけられる粒度: {g['level']}")
    pn = (f"{C_RED}不可（固有名は創作のまま）{C_END}" if not g["proper_noun"]
          else f"{C_CYA}可{C_END}")
    print(f"  固有名詞の裏づけ: {pn}")
    print(f"  {C_GRY}{g['source'][:70]}{C_END}")
    print("=" * 80)
    print()


def cmd_map(reg, args):
    print()
    print("=" * 80)
    print(f"{C_BLD}属性の裏づけ地図{C_END}")
    print("=" * 80)
    print(f"  {pad('属性',16)}{pad('種別',10)}{pad('裏づけられる粒度',32)}固有名")
    print("  " + "-" * 74)
    for attr, a in reg["attributes"].items():
        g = a.get("groundability", {})
        c = tcol(a["tier"])
        pn = "不可" if not g.get("proper_noun") else "可"
        pncol = C_RED if pn == "不可" else C_CYA
        lvl = g.get("level", "")
        if len(lvl) > 30:
            lvl = lvl[:29] + "…"
        print(f"  {pad(attr,16)}{c}{pad(a['tier'],10)}{C_END}"
              f"{pad(lvl,32)}{pncol}{pn}{C_END}")
    print("  " + "-" * 74)
    print(f"  {C_GRY}会話に出す商品名・店名・固有の地名は、どれも固有名『不可』。")
    print(f"  エンジンは品目・業態・都市階級まで裏づけ、固有名は創作と切り分ける。{C_END}")
    print("=" * 80)
    print()


# ─────────────────────────────────────────────────────────
def parse_where(pairs):
    where = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"--where は 属性=値 の形で指定してください: {p}")
        k, v = p.split("=", 1)
        where[k.strip()] = v.strip()
    return where


def main():
    ap = argparse.ArgumentParser(description="属性クロス基盤")
    sub = ap.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("occurrence", help="条件をすべて満たす割合")
    o.add_argument("--where", nargs="+", required=True)
    o.set_defaults(func=cmd_occurrence)

    c = sub.add_parser("conditional", help="条件付き分布")
    c.add_argument("--target", required=True)
    c.add_argument("--where", nargs="+", default=[])
    c.set_defaults(func=cmd_conditional)

    m = sub.add_parser("metric", help="金額・数量の推計")
    m.add_argument("--item", required=True)
    m.add_argument("--where", nargs="+", required=True)
    m.set_defaults(func=cmd_metric)

    g = sub.add_parser("ground", help="属性の裏づけ限界")
    g.add_argument("attr")
    g.set_defaults(func=cmd_ground)

    mp = sub.add_parser("map", help="全属性の裏づけ地図")
    mp.set_defaults(func=cmd_map)

    args = ap.parse_args()
    reg = load_registry()
    try:
        return args.func(reg, args) or 0
    except (KeyError, ValueError) as e:
        print(f"\n{C_RED}エラー: {e}{C_END}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
