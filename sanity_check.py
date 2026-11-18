#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
sanity_check.py — 翼の出力が元データの含意どおりか自動検査する

「数字が現場感覚と合うか」を、使い捨ての目視でなく、繰り返せる検査にしたもの。
翼（reaction.py）の出力が、土台の1次データ（家計調査・国勢調査）から
当然そうなるはずの関係を保っているかを機械的に確かめる。

検査する関係:
  1. 教育の対象外率 ≒ 全国の子なし率（国勢調査）
  2. 同一カテゴリ内で、価格が高い案ほど価格受容が低い（単調性）
  3. 外食の年収別ターゲティングで、V層 > I層（家計調査の所得勾配）
  4. 外食の都道府県別で、高所得県 ≧ 低所得県
  5. 深掘り生成が、指定セグメントで目標人数に到達する

使い方（PowerShell）:
  python sanity_check.py
終了コード 0=全合格 / 1=要確認あり。
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import population as pop          # noqa: E402
import reaction as R             # noqa: E402

C_RED, C_CYA, C_GRY, C_BLD, C_END = (
    "\033[91m", "\033[92m", "\033[90m", "\033[1m", "\033[0m")

results = []


def check(name, ok, detail):
    results.append(ok)
    tag = f"{C_CYA}PASS{C_END}" if ok else f"{C_RED}要確認{C_END}"
    print(f"  [{tag}] {name}")
    print(f"         {C_GRY}{detail}{C_END}")


def load_props(fname):
    doc = json.loads((HERE / fname).read_text(encoding="utf-8"))
    return doc["cross_item"], doc["proposals"]


def grounding_for(item):
    table, src, as_of = R.load_item_baseline(item)
    return R.resolve_grounding(item, table, src, as_of)


def main():
    src = pop.load_src()
    rng = random.Random(7)
    print()
    print("=" * 72)
    print(f"{C_BLD}翼の整合性チェック{C_END}")
    print("=" * 72)

    personas = pop.generate(src, "全国", 3000, rng)["personas"]

    # 1. 教育の対象外率 ≒ 全国の子なし率
    item, props = load_props("proposals_education.json")
    res = R.compare(personas, props, grounding_for(item))
    childless = 1 - src["national"]["households_with_children"] / src["national"]["total_households"]
    obs = res[0]["irrelevant_share"]
    check("教育の対象外率が国勢調査の子なし率と一致",
          abs(obs - childless) <= 0.03,
          f"対象外 {obs*100:.1f}% ／ 国勢調査の子なし率 {childless*100:.1f}%（許容3pt）")

    # 2. 価格の単調性（外食）
    item, props = load_props("proposals_sample.json")
    res = R.compare(personas, props, grounding_for(item))
    by_price = sorted(res, key=lambda r: r["price_monthly"])
    accs = [r["mean_acceptance"] for r in by_price]
    mono = all(accs[i] >= accs[i + 1] - 1e-9 for i in range(len(accs) - 1))
    check("価格が高い案ほど価格受容が低い（外食）", mono,
          "／".join(f"{r['price_monthly']:,}円→{r['mean_acceptance']*100:.0f}%" for r in by_price))

    # 3. 年収別ターゲティング V>I（外食・週末ディナー）
    pr = next(p for p in props if p["id"] == "B")
    rows = R.target(personas, pr, grounding_for(item), "income_quintile")
    acc = {r["segment"]: r["mean_acceptance"] for r in rows}
    check("年収別で V層 > I層（外食16,000円）", acc.get("V", 0) > acc.get("I", 1),
          f"V層 {acc.get('V',0)*100:.0f}% ＞ I層 {acc.get('I',0)*100:.0f}%")

    # 4. 都道府県別 高所得県 ≧ 低所得県
    big = pop.generate(src, "全国", 8000, rng)["personas"]
    rows = R.target(big, pr, grounding_for(item), "prefecture")
    acc = {r["segment"]: r["mean_acceptance"] for r in rows}
    tk, ok0 = acc.get("東京都", 0), acc.get("沖縄県", 1)
    check("高所得県 ≧ 低所得県（外食・東京≧沖縄）", tk >= ok0,
          f"東京都 {tk*100:.0f}% ≧ 沖縄県 {ok0*100:.0f}%")

    # 5. 深掘り生成が目標人数に到達
    target_n = 300
    probe = pop.generate(src, "全国", 4000, rng)
    seg = pop.apply_filter(probe["personas"], {"has_child": "あり", "pet": "犬"})
    guard = 0
    while len(seg) < target_n and guard < 15:
        seg += pop.apply_filter(pop.generate(src, "全国", 20000, rng)["personas"],
                                {"has_child": "あり", "pet": "犬"})
        guard += 1
    check("深掘り生成が目標人数に到達（子あり×犬 300人）", len(seg) >= target_n,
          f"生成到達 {min(len(seg), target_n)}/{target_n}人")

    print("=" * 72)
    passed = sum(results)
    allok = passed == len(results)
    color = C_CYA if allok else C_RED
    print(f"  {color}{passed}/{len(results)} 合格{C_END}"
          + ("" if allok else "　要確認あり"))
    print("=" * 72)
    print()
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
