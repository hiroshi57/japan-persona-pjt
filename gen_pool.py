#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""スクリーニングUI用のペルソナ母集団を生成する。
接地24属性＋心理15項目＋Nemotronの語り（氏名はNemotronから抽出）を持たせてJSON出力。
使い方: python gen_pool.py <人数> <出力パス>"""
import random, json, re, sys
import population as pop, psychographic as pg, nemotron_enrich as ne

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
OUT = sys.argv[2] if len(sys.argv) > 2 else "ui/pool.json"

rng = random.Random(2026)
spec = pg.load_spec()
pool, _ = ne.load_pool()
personas = pop.generate(pop.load_src(), "全国", N, rng)["personas"]


def extract_name(text):
    if not text:
        return None
    m = re.match(r"^([^\s、。]+\s+[^\s、。]+?)(は|さん)", text)
    if m:
        return m.group(1)
    m = re.match(r"^([^\s、。]{2,8}?)(は|さん)", text)
    return m.group(1) if m else None


out = []
for i, p in enumerate(personas):
    pg.assign_psych(p, rng, spec)
    row, grain = pool.match(p, rng)
    ne.enrich(p, row, grain)
    nv = p.get("narrative", {})
    name = extract_name(nv.get("人物像")) or extract_name(nv.get("文化的背景")) or f"対象者{i+1:04d}"
    out.append({
        "id": i, "name": name,
        "basic": {"性別": p["gender"], "年齢帯": p["age_band"], "都道府県": p["prefecture"],
                  "配偶": p.get("marital_status"), "学歴": p.get("education"), "業種": p.get("industry"),
                  "職業": p.get("occupation"), "世帯人員": p["household_size"], "子": p["has_child"],
                  "年収五分位": p["income_quintile"], "年収万円": round(p["income_yen"] / 10000)},
        "detail": {"ペット": p.get("pet"), "金融資産": p.get("financial_assets"), "運転免許": p.get("drivers_license"),
                   "車": p.get("car"), "ネット頻度": p.get("internet_frequency"), "EC": p.get("ec_purchase_ratio"),
                   "スマホ": p.get("smartphone"), "SNS": p.get("sns"), "投資": p.get("investment"),
                   "保険": p.get("insurance"), "住居": p.get("housing_type"), "クレカ": p.get("credit_cards"),
                   "共働き": p.get("dual_income")},
        "psych": p["psych"], "psych_tier": p["psych_tier"],
        "narrative": {"文化的背景": nv.get("文化的背景"), "スキル": nv.get("スキル"),
                      "趣味": nv.get("趣味"), "キャリア志向": nv.get("キャリア志向"), "人物像": nv.get("人物像"),
                      "_tier": nv.get("_tier"), "_source": nv.get("_source")},
    })

items = []
for d in spec["derived"]:
    items.append({"id": d["id"], "label": d["label"], "cat": d["category"], "tier": d["tier"]})
for it in spec["probabilistic"]:
    items.append({"id": it["id"], "label": it["label"], "cat": it["category"], "tier": it["tier"]})

json.dump({"personas": out, "psych_items": items, "n": len(out)},
          open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
import os
print(f"母集団 {len(out)}名 生成 → {OUT}  {os.path.getsize(OUT)/1024/1024:.1f}MB")
print("氏名抽出:", sum(1 for o in out if not o['name'].startswith('対象者')), "/", len(out))
