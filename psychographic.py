#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""心理情報（スクリーニング15項目）をペルソナに割り当てる。

2層：
  derived … 接地済み属性から確定的に導く障壁（信頼度:接地）。
  probabilistic … 全国統計が無い心理・行動の障壁（信頼度:方向性のみ）。
                  接地属性で条件付けした仮の出現率で割り当て、本調査で要確認。

スキーマは psych_items.json。方向性項目の出現率は仮の値で、接地と偽ってはならない。
スクリーニングで方向性項目を使うときは tier を表示し、仮説段階であることを明示する。
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 接地項目：接地済み属性から確定的に判定（evalは使わず関数で安全に）
DERIVED_PREDS = {
    "no_savings": lambda p: p.get("financial_assets") == "なし",
    "low_income": lambda p: p.get("income_quintile") in ("I", "II"),
    "uninsured":  lambda p: "なし" in (p.get("insurance") or []),
    "no_car":     lambda p: p.get("car") == "0",
    "ec_nonuser": lambda p: p.get("ec_purchase_ratio") == "非利用",
}


def load_spec():
    return json.loads((HERE / "psych_items.json").read_text(encoding="utf-8"))


def _match(cond_if, p):
    for attr, want in cond_if.items():
        v = p.get(attr)
        if isinstance(want, list):
            if v not in want:
                return False
        elif v != want:
            return False
    return True


def _prob(item, p):
    prob = item["base"]
    for c in item.get("conditions", []):
        if _match(c["if"], p):
            prob *= c["mult"]
    return min(0.95, max(0.01, prob))


def assign_psych(persona, rng, spec=None):
    spec = spec or load_spec()
    flags, tiers = {}, {}
    for d in spec["derived"]:
        flags[d["id"]] = bool(DERIVED_PREDS[d["id"]](persona))
        tiers[d["id"]] = d["tier"]
    for it in spec["probabilistic"]:
        flags[it["id"]] = rng.random() < _prob(it, persona)
        tiers[it["id"]] = it["tier"]
    persona["psych"] = flags
    persona["psych_tier"] = tiers
    return persona


def item_index(spec=None):
    spec = spec or load_spec()
    idx = {}
    for d in spec["derived"]:
        idx[d["id"]] = d
    for it in spec["probabilistic"]:
        idx[it["id"]] = it
    return idx


def screen(personas, require=(), mode="all"):
    """psych項目idのリスト require が立つ人に絞る。mode=all/any。"""
    if not require:
        return list(personas)
    out = []
    for p in personas:
        ps = p.get("psych", {})
        hits = [ps.get(i, False) for i in require]
        if (mode == "all" and all(hits)) or (mode == "any" and any(hits)):
            out.append(p)
    return out


def screen_confidence(require, idx=None):
    """スクリーニング条件の信頼度を返す。方向性のみが混じれば仮説段階。"""
    idx = idx or item_index()
    tiers = [idx[i]["tier"] for i in require if i in idx]
    if any(t == "方向性のみ" for t in tiers):
        return "方向性のみ（仮説段階・本調査で要確認）"
    return "接地（統計で裏づけ）"


def _demo(n, seed):
    import population as pop
    rng = random.Random(seed)
    spec = load_spec()
    idx = item_index(spec)
    ps = pop.generate(pop.load_src(), "全国", n, rng)["personas"]
    for p in ps:
        assign_psych(p, rng, spec)
    print(f"心理15項目の出現率（n={n}）")
    for key in ("derived", "probabilistic"):
        for it in spec[key]:
            rate = sum(p["psych"][it["id"]] for p in ps) / n
            print(f"  [{it['tier']:<6}] {it['label']:<28} {rate*100:4.1f}%")
    # スクリーニング例：基本×詳細×心理の掛け合わせ
    print("\n── スクリーニング例 ──")
    step1 = pop.apply_filter(ps, {"income_quintile": ["III", "IV", "V"]})
    print(f"  全体{n} → 年収III+ {len(step1)}人（接地）")
    step2 = screen(step1, ["cart_abandon"], "all")
    print(f"  → さらにカゴ落ち経験 {len(step2)}人  信頼度:{screen_confidence(['cart_abandon'], idx)}")
    step3 = screen(step1, ["no_savings"], "all")
    print(f"  → さらに貯蓄なし {len(step3)}人  信頼度:{screen_confidence(['no_savings'], idx)}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    _demo(a.n, a.seed)
