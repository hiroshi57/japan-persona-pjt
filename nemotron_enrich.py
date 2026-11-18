#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nemotron-Personas-Japan の自然言語層を、当エンジンの接地ペルソナに付与する。

役割分担：
  population.py … 消費・経済属性を実統計で接地（年収・EC・金融資産・車・住居・SNS…）。
  Nemotron     … 心理・ナラティブ層（文化的背景・スキル・趣味・キャリア志向・人物像）。

付与する自然言語は全てLLM生成＝信頼度『方向性のみ』。出典はNVIDIA Nemotron-Personas-Japan
(CC-BY-4.0)。実在個人のデータではないので、人口構成は接地・語りは創作、と区別して扱う。

データ取得：この環境からは huggingface.co のデータ配信がプロキシで遮断される。
本データ（parquet 100万行）は --parquet で渡す。無ければ同梱サンプル(nemotron_sample.json)で動く。
"""
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = "NVIDIA Nemotron-Personas-Japan (CC-BY-4.0)"
TIER = "方向性のみ"

SEX_MAP = {"男性": "男", "女性": "女"}
MARITAL_MAP = {"有配偶": "既婚", "未婚": "未婚", "離別": "離別", "死別": "死別"}
EDU_BASE = ["大学院卒", "大学卒", "短大・高専卒", "専門学校卒", "高校卒", "中学卒"]


def age_band(age):
    """Nemotronの整数年齢→当エンジンの年齢帯。"""
    if age < 30:
        return "20-29"
    if age < 40:
        return "30-39"
    if age < 50:
        return "40-49"
    if age < 60:
        return "50-59"
    if age < 70:
        return "60-69"
    return "70+"


def norm_edu(text):
    """『大学卒 理系』『高校卒』等を基本区分に正規化。"""
    for base in EDU_BASE:
        if text and text.startswith(base):
            return base
    return text or ""


def norm_marital(text):
    """『既婚』『離別 (子供あり)』等を先頭区分に正規化。"""
    for key in ("既婚", "未婚", "離別", "死別"):
        if text and text.startswith(key):
            return key
    return text or ""


def _key(sex, band, pref, marital, edu):
    return (sex, band, pref, marital, edu)


class NemotronPool:
    """Nemotron行を突合キー別に索引し、ペルソナに最も近い行を返す。"""

    def __init__(self, rows):
        self.rows = rows
        self.by5, self.by4, self.by3, self.by2, self.by1 = {}, {}, {}, {}, {}
        for r in rows:
            sx, bd = r["sex"], age_band(int(r["age"]))
            pf, ma, ed = r["prefecture"], norm_marital(r["marital_status"]), norm_edu(r["education_level"])
            self.by5.setdefault(_key(sx, bd, pf, ma, ed), []).append(r)
            self.by4.setdefault((sx, bd, pf, ma), []).append(r)
            self.by3.setdefault((sx, bd, pf), []).append(r)
            self.by2.setdefault((sx, bd), []).append(r)
            self.by1.setdefault((sx,), []).append(r)

    def match(self, persona, rng):
        """段階的にキーを緩めて一致行を返す。見つかった粒度も返す。"""
        sx = SEX_MAP.get(persona.get("gender"), persona.get("gender"))
        bd = persona.get("age_band")
        pf = persona.get("prefecture")
        ma = MARITAL_MAP.get(persona.get("marital_status"), persona.get("marital_status"))
        ed = norm_edu(persona.get("education"))
        for idx, k in ((self.by5, _key(sx, bd, pf, ma, ed)), (self.by4, (sx, bd, pf, ma)),
                       (self.by3, (sx, bd, pf)), (self.by2, (sx, bd)), (self.by1, (sx,))):
            pool = idx.get(k)
            if pool:
                grain = {5: "性別+年齢+県+配偶+学歴", 4: "性別+年齢+県+配偶",
                         3: "性別+年齢+県", 2: "性別+年齢", 1: "性別のみ"}[len(k)]
                return rng.choice(pool), grain
        return (rng.choice(self.rows) if self.rows else None), "無作為"


def enrich(persona, row, grain):
    """Nemotronの自然言語層を付与。全て『方向性のみ』タグと出典つき。"""
    def lst(v):
        # 本データの _list 列は list / numpy配列 / Python形式の文字列 のいずれでも来る
        if isinstance(v, (list, tuple)):
            return list(v)
        if hasattr(v, "tolist"):
            return list(v.tolist())
        if isinstance(v, str) and v.strip():
            import ast
            try:
                return list(ast.literal_eval(v))
            except (ValueError, SyntaxError):
                return [v]
        return []
    persona["narrative"] = {
        "文化的背景": row.get("cultural_background"),
        "スキル": lst(row.get("skills_and_expertise_list")),
        "趣味": lst(row.get("hobbies_and_interests_list")),
        "キャリア志向": row.get("career_goals_and_ambitions"),
        "人物像": row.get("persona"),
        "_tier": TIER,
        "_source": SOURCE,
        "_match_grain": grain,
    }
    return persona


def load_pool(parquet=None):
    """parquet（本番）を優先、無ければ同梱サンプルを読む。
    parquet未指定でも、モジュールと同階層に nemotron_japan_sample.parquet があれば自動採用。"""
    if parquet is None:
        local = HERE / "nemotron_japan_sample.parquet"
        if local.exists():
            parquet = str(local)
    if parquet:
        import pandas as pd
        cols = ["sex", "age", "marital_status", "education_level", "occupation",
                "region", "prefecture", "cultural_background",
                "skills_and_expertise_list", "hobbies_and_interests_list",
                "career_goals_and_ambitions", "persona"]
        df = pd.read_parquet(parquet, columns=cols)
        return NemotronPool(df.to_dict("records")), f"parquet {len(df)}行"
    data = json.loads((HERE / "nemotron_sample.json").read_text(encoding="utf-8"))
    return NemotronPool(data["rows"]), f"同梱サンプル {len(data['rows'])}行"


def _demo(n, parquet, seed):
    import population as pop
    rng = random.Random(seed)
    pool, src = load_pool(parquet)
    personas = pop.generate(pop.load_src(), "全国", n, rng)["personas"]
    for p in personas:
        row, grain = pool.match(p, rng)
        if row:
            enrich(p, row, grain)
    print(f"Nemotron出典: {src}  付与={sum('narrative' in p for p in personas)}/{n}")
    ex = next(p for p in personas if "narrative" in p)
    nv = ex["narrative"]
    print("\n── 付与例（接地デモグラ＋Nemotron語り）──")
    print(f"性別{ex['gender']} 年齢帯{ex['age_band']} {ex['prefecture']} "
          f"配偶{ex.get('marital_status')} 学歴{ex.get('education')} 年収{ex['income_yen']}円")
    print(f"接地消費: 車{ex.get('car')}台 住居{ex.get('housing_type')} "
          f"SNS{ex.get('sns')} 金融資産{ex.get('financial_assets')}")
    print(f"突合粒度: {nv['_match_grain']}  信頼度: {nv['_tier']}  出典: {nv['_source']}")
    print(f"文化的背景: {nv['文化的背景']}")
    print(f"スキル: {nv['スキル']}")
    print(f"趣味: {nv['趣味']}")
    print(f"キャリア志向: {nv['キャリア志向']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--parquet", default=None, help="Nemotron parquetのパス（無ければ同梱サンプル）")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    _demo(a.n, a.parquet, a.seed)
