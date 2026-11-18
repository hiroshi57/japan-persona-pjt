#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
reaction.py — 価値創生の翼：施策案への反応を分解して比べる

生成したセグメントに複数の施策案をぶつけ、反応を
「価格受容・購入意向・確信度」に分けて返し、案を比較する。

設計の芯（claude/design/value-creation-wing.md）:
  購入意向をLLMに直接聞かない。分散過少と偏りがそのまま出るため。
  価格受容を家計調査で接地し、それを主たる説明変数にした透明な式で出す。
  すべての結果に信頼度タグ（接地／照合／方向性のみ）を付ける。

反応の4分解:
  価格受容  … 接地。家計調査の品目月額に対する施策案価格の比から出す。
  購入意向  … 価格受容（接地）×便益適合（既定1.0・方向性のみ）。
  反応理由  … 任意のLLMフック。APIキーが無くても核は動く。
  確信度    … 結果のうち接地できた割合から導く。

用途の限定:
  案の優劣をそのまま意思決定に使わない。本調査前の仮説出しと足切りに限る。

使い方（PowerShell）:
  python reaction.py run --frame 全国 --n 500 --proposals proposals_sample.json
  python reaction.py run --frame 東京都 --n 300 --where has_child=あり --proposals proposals_sample.json
  python reaction.py accept --item 外食 --price 16000   # 価格受容の五分位別の内訳だけ
"""

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import population as pop  # noqa: E402

ATTR_PATH = HERE / "attributes.json"

C_RED, C_YEL, C_CYA, C_GRY, C_BLD, C_END = (
    "\033[91m", "\033[93m", "\033[92m", "\033[90m", "\033[1m", "\033[0m"
)

# 品目の駆動要因。家計調査の年収別テーブルで接地してよいのは所得駆動の品目だけ。
# 子や年齢で駆動される品目を年収だけで接地すると、世帯構成の歪みが入る。
# （例：教育費は子ありで決まるのに、年収V層全体に高い教育費が広がってしまう）
CATEGORY_DRIVER = {
    "外食": "所得", "教養娯楽": "所得", "被服及び履物": "所得",
    "交通・通信": "所得", "こづかい": "所得", "交際費": "所得",
    "教育": "子の有無", "保健医療": "世帯主年齢", "家具・家事用品": "所得",
}


def driver_caveat(item):
    """所得以外で駆動される品目のとき、接地の限界を返す。無ければ None。"""
    d = CATEGORY_DRIVER.get(item, "所得")
    if d == "所得":
        return None
    if item in SEGMENT_GROUNDING:
        # 取得済みの年齢階級別の水準を錨に、ペルソナの属性で接地済み。注意は不要。
        return None
    if best_axis(item) == "age":
        return (f"{item}は「{d}」で決まる品目のため、年収軸では歪む。"
                f"年齢軸の接地ビュー（accept --item {item} --axis age）を使うこと。"
                f"複数案比較(run)はまだ年収軸なので、この品目では参考値。")
    return (f"{item}は主に「{d}」で決まる品目です。年収別テーブルだけの接地では"
            f"世帯構成の歪みが入ります。{d}を軸に足すまで、この品目の接地は参考値です。")


# 価格受容のバンド境界（家計調査の品目月額に対する倍率）
BAND = [
    (0.5, "安い寄り"),
    (1.5, "妥当"),
    (3.0, "高い寄り"),
    (float("inf"), "かなり高い"),
]


# 品目をどの軸で接地するか。所得で決まる品目は income、
# 世帯構成で決まる品目は年齢を代理軸にする（教育＝子育て期、保健医療＝加齢）。
AXIS_REL = {"income": "metric:金額|income_quintile", "age": "metric:金額|age_band"}
AGE_ORDER = ["-29", "30-39", "40-49", "50-59", "60-69", "70+"]
QUINTILE_ORDER = ["I", "II", "III", "IV", "V"]


def load_item_baseline(item, axis="income"):
    """家計調査の品目月額を、指定軸（income/age）で返す"""
    attr = json.loads(ATTR_PATH.read_text(encoding="utf-8"))
    rel = attr["relationships"].get(AXIS_REL[axis], {})
    table = rel.get("items", {}).get(item)
    if not table:
        raise KeyError(f"家計調査（{axis}軸）に品目 {item} の金額が未登録です")
    return table, rel.get("source", ""), rel.get("as_of", "")


def best_axis(item):
    """品目の駆動要因に合う接地軸を返す。年齢軸に表があればそれを優先。"""
    if CATEGORY_DRIVER.get(item, "所得") != "所得":
        attr = json.loads(ATTR_PATH.read_text(encoding="utf-8"))
        if item in attr["relationships"].get("metric:金額|age_band", {}).get("items", {}):
            return "age"
    return "income"


# ─────────────────────────────────────────────────────────
# 価格受容（接地）
# ─────────────────────────────────────────────────────────
def price_ratio(price_monthly, baseline_monthly):
    """施策案の想定月額が、その層の品目支出の何倍か"""
    return price_monthly / max(baseline_monthly, 1)


def ratio_to_band(ratio):
    for hi, label in BAND:
        if ratio <= hi:
            return label
    return BAND[-1][1]


def acceptance_score(ratio):
    """
    倍率から価格受容を0〜1で返す、単調減少のモデル曲線。
    曲線自体はモデルだが、入力の倍率は家計調査で接地されている。
    """
    return 1.0 / (1.0 + 0.8 * math.pow(max(ratio, 0.0), 1.2))


# 世帯構成で決まる品目の接地。年齢階級別の金額（家計調査2024）を錨にし、
# ペルソナが持つ属性（has_child）で引く。教育は子育て期の月額を子あり世帯の基準にする。
# 子なし世帯はその品目の対象外（支出ほぼ0）として扱い、ターゲットから外す。
SEGMENT_GROUNDING = {
    "教育": {
        "attr": "has_child",
        "baseline": {"あり": 20289, "なし": 0},   # 子あり=40-49と50-59の平均（22037,18541）
        "source": "家計調査2024 世帯主年齢階級別 教育費（子育て期40-50代の水準）×国勢調査 子の有無",
    },
}


def resolve_grounding(item, income_table, income_src, income_as_of):
    """品目の接地方法を決める。構成駆動品目はセグメント接地、他は年収接地。"""
    if item in SEGMENT_GROUNDING:
        g = SEGMENT_GROUNDING[item]
        return {"mode": "segment", "attr": g["attr"], "baseline": g["baseline"],
                "source": g["source"], "as_of": "2024"}
    return {"mode": "income", "table": income_table,
            "source": income_src, "as_of": income_as_of}


# ─────────────────────────────────────────────────────────
# 反応分解（1ペルソナ×1施策案）
# ─────────────────────────────────────────────────────────
def react(persona, proposal, grounding):
    if grounding["mode"] == "segment":
        seg = persona.get(grounding["attr"])
        baseline = grounding["baseline"].get(seg, 0)
        label = f"{grounding['attr']}={seg}"
    else:
        q = persona["income_quintile"]
        baseline = grounding["table"][q]
        label = q

    # 対象外（その品目をほぼ使わない層）。ターゲットから外す。
    if baseline <= 0:
        return {"segment": label, "baseline": 0, "price": proposal["price_monthly"],
                "ratio": None, "band": "対象外", "price_acceptance": 0.0,
                "benefit_fit": proposal.get("benefit_fit", 1.0),
                "purchase_intent": 0.0, "confidence": 1.0, "relevant": False}

    ratio = price_ratio(proposal["price_monthly"], baseline)
    accept = acceptance_score(ratio)
    benefit_fit = proposal.get("benefit_fit", 1.0)
    intent = max(0.0, min(1.0, accept * benefit_fit))
    confidence = max(0.3, 1.0 - abs(benefit_fit - 1.0))

    return {
        "segment": label, "baseline": baseline, "price": proposal["price_monthly"],
        "ratio": ratio, "band": ratio_to_band(ratio),
        "price_acceptance": accept, "benefit_fit": benefit_fit,
        "purchase_intent": intent, "confidence": confidence, "relevant": True,
    }


# ─────────────────────────────────────────────────────────
# 複数案比較（セグメント×複数案）
# ─────────────────────────────────────────────────────────
def compare(personas, proposals, grounding):
    results = []
    for pr in proposals:
        reacts = [react(p, pr, grounding) for p in personas]
        # 対象外（その品目を使わない層）は平均から除き、別に割合で持つ
        rel = [r for r in reacts if r["relevant"]]
        irrelevant_share = 1 - len(rel) / len(reacts)
        accepts = [r["price_acceptance"] for r in rel] or [0.0]
        intents = [r["purchase_intent"] for r in rel] or [0.0]

        band_dist = {}
        for r in rel:
            band_dist[r["band"]] = band_dist.get(r["band"], 0) + 1
        band_dist = {k: v / len(rel) for k, v in band_dist.items()} if rel else {}

        sd = statistics.pstdev(accepts) if len(accepts) > 1 else 0.0
        neutral = abs(pr.get("benefit_fit", 1.0) - 1.0) < 1e-9
        results.append({
            "id": pr["id"], "name": pr["name"], "price_monthly": pr["price_monthly"],
            "mean_acceptance": statistics.mean(accepts),
            "mean_intent": statistics.mean(intents),
            "sd_acceptance": sd, "band_dist": band_dist,
            "benefit_neutral": neutral, "n": len(reacts),
            "irrelevant_share": irrelevant_share,
        })

    results.sort(key=lambda r: r["mean_intent"], reverse=True)
    return results


# ─────────────────────────────────────────────────────────
# 接地したターゲティング（1施策案を誰に当てるか）
#
# 便益の仮定を一切使わず、家計調査で接地された価格受容だけで、
# セグメント間の当てやすさを比べる。これは完全に接地なので、
# 「どの層を狙うか」の答えに方向性の濁りが入らない。
# ─────────────────────────────────────────────────────────
def bucket_key(persona, by):
    if by == "prefecture":
        return persona["prefecture"]
    return str(persona.get(by))


def target(personas, proposal, grounding, by):
    """施策案1案について、by で分けたセグメントごとの価格受容を出して並べる"""
    buckets = {}
    for p in personas:
        k = bucket_key(p, by)
        buckets.setdefault(k, []).append(react(p, proposal, grounding))

    rows = []
    for k, rs in buckets.items():
        rel = [r for r in rs if r["relevant"]]
        accepts = [r["price_acceptance"] for r in rel] or [0.0]
        easy = (sum(1 for r in rel if r["band"] in ("安い寄り", "妥当")) / len(rs))
        rows.append({
            "segment": k, "n": len(rs), "share": len(rs) / len(personas),
            "mean_acceptance": statistics.mean(accepts),
            "easy_share": easy,
        })
    rows.sort(key=lambda r: r["mean_acceptance"], reverse=True)
    return rows


# ─────────────────────────────────────────────────────────
# 表示
# ─────────────────────────────────────────────────────────
def band_color(label):
    return {"安い寄り": C_CYA, "妥当": C_CYA, "高い寄り": C_YEL,
            "かなり高い": C_RED}.get(label, C_GRY)


def cmd_accept(args):
    axis = args.axis or best_axis(args.item)
    table, src, as_of = load_item_baseline(args.item, axis)
    order = AGE_ORDER if axis == "age" else QUINTILE_ORDER
    axis_label = "世帯主年齢" if axis == "age" else "年収五分位"
    print()
    print("=" * 74)
    print(f"{C_BLD}価格受容：{args.item} / 想定月額 {args.price:,}円 / {axis_label}軸{C_END}")
    print("=" * 74)
    print(f"  {axis_label:<10}{'家計調査の月額':>13}{'倍率':>8}{'判定':>10}{'受容スコア':>10}")
    for k in order:
        base = table.get(k)
        if base is None:
            continue
        if base <= 0:
            print(f"  {k:<10}{base:>12,}円{'—':>8}{C_GRY}{'対象外':>10}{C_END}{'—':>9}")
            continue
        ratio = price_ratio(args.price, base)
        band = ratio_to_band(ratio)
        acc = acceptance_score(ratio)
        c = band_color(band)
        print(f"  {k:<10}{base:>12,}円{ratio:>7.1f}倍{c}{band:>10}{C_END}{acc*100:>9.0f}%")
    print("  " + "-" * 70)
    print(f"  {C_CYA}[接地]{C_END} 倍率と判定は家計調査由来。{C_GRY}{src[:52]}（{as_of}）{C_END}")
    if axis == "age":
        print(f"  {C_GRY}この品目は所得でなく世帯主年齢で決まるため、年齢軸で接地している。{C_END}")
    print("=" * 74)
    print()


def cmd_run(args):
    proposals_doc = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
    item = proposals_doc.get("cross_item", "外食")
    proposals = proposals_doc["proposals"]
    table, src, as_of = load_item_baseline(item)
    grounding = resolve_grounding(item, table, src, as_of)

    # セグメント生成（population.py を使う）
    src_data = pop.load_src()
    where = {}
    for p in args.where or []:
        k, v = p.split("=", 1)
        where[k.strip()] = v.strip()

    rng = random.Random(args.seed)
    if where:
        probe = pop.generate(src_data, args.frame, 3000, rng)
        hit = len(pop.apply_filter(probe["personas"], where))
        yr = max(hit / 3000, 0.001)
        personas = pop.apply_filter(probe["personas"], where)
        guard = 0
        while len(personas) < args.n and guard < 12:
            batch = min(60000, int((args.n - len(personas)) / yr * 1.3) + 1000)
            extra = pop.generate(src_data, args.frame, batch, rng)
            personas += pop.apply_filter(extra["personas"], where)
            guard += 1
        personas = personas[:args.n]
    else:
        personas = pop.generate(src_data, args.frame, args.n, rng)["personas"]

    results = compare(personas, proposals, grounding)

    # 表示
    cond = (" / " + " ".join(f"{k}={v}" for k, v in where.items())) if where else ""
    seg_note = "（子あり世帯のみ対象）" if grounding["mode"] == "segment" else ""
    print()
    print("=" * 76)
    print(f"{C_BLD}複数案比較：{item}{seg_note} / {args.frame}{cond} / {len(personas)}人{C_END}")
    print("=" * 76)

    print(f"  {'順位':<4}{'案':<18}{'想定月額':>9}{'平均購入意向':>12}{'平均価格受容':>12}{'ばらつき':>9}")
    for i, r in enumerate(results, 1):
        print(f"  {i:<4}{r['id']+' '+r['name']:<18}{r['price_monthly']:>8,}円"
              f"{r['mean_intent']*100:>11.0f}%{r['mean_acceptance']*100:>11.0f}%"
              f"{r['sd_acceptance']:>9.3f}")
    if grounding["mode"] == "segment" and results:
        print(f"  {C_GRY}対象外（この品目を使わない層）: "
              f"{results[0]['irrelevant_share']*100:.0f}%（平均から除外）{C_END}")
    print()

    # バンド分布（勝ち案の中身を分布で見せる）
    top = results[0]
    print(f"{C_BLD}■ 1位「{top['name']}」の価格受容の分布{C_END}")
    for band in ["安い寄り", "妥当", "高い寄り", "かなり高い"]:
        share = top["band_dist"].get(band, 0.0)
        bar = "█" * round(share * 36)
        c = band_color(band)
        print(f"  {c}{band:<8}{C_END}{share*100:>6.0f}%  {c}{bar}{C_END}")
    print()

    # 分散過少の自己点検（年収接地のときだけ。セグメント接地は基準額が一定で0が正常）
    print(f"{C_BLD}■ 自己点検{C_END}")
    if grounding["mode"] == "segment":
        print(f"  {C_GRY}セグメント接地のため、対象層内の基準額は一定（ばらつき0は設計どおり）。{C_END}")
    else:
        for r in results:
            if r["sd_acceptance"] < 0.02:
                print(f"  {C_RED}警告{C_END} {r['name']}：価格受容のばらつきがほぼ無い。"
                      f"年収差が消えている疑い。")
        if all(r["sd_acceptance"] >= 0.02 for r in results):
            print(f"  {C_CYA}OK{C_END} 全案で価格受容が分散している（分散過少なし）。")
    print()

    # 信頼度タグ
    print(f"{C_BLD}■ 信頼度{C_END}")
    cav = driver_caveat(item)
    if cav:
        print(f"  {C_RED}注意{C_END} {cav}")
        print()
    if grounding["mode"] == "segment":
        print(f"  価格受容   {C_CYA}[接地]{C_END} {grounding['source']}（{grounding['as_of']}）")
    else:
        print(f"  価格受容   {C_CYA}[接地]{C_END} 家計調査 {item} の五分位別月額（{as_of}）")
    neutral = all(r["benefit_neutral"] for r in results)
    if neutral:
        print(f"  購入意向   {C_CYA}[接地]{C_END} 便益適合を中立(1.0)にしたため価格受容と一致")
        print(f"  案の順位   {C_YEL}[方向性のみ]{C_END} 価格だけで並べた順位。"
              f"便益の差はまだ入っていない")
    else:
        print(f"  購入意向   {C_YEL}[接地＋方向性]{C_END} 便益適合を動かした分は方向性")
        print(f"  案の順位   {C_YEL}[方向性のみ]{C_END} 便益適合の仮定に依存")
    print(f"  {C_GRY}態度の外的妥当性（実人間との一致）は外部調査の照合が済むまで未検証。{C_END}")
    print(f"  {C_GRY}用途は本調査前の仮説出しと足切りに限る。優劣をそのまま決定に使わない。{C_END}")
    print("=" * 76)
    print()

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"item": item, "frame": args.frame, "where": where,
             "n": len(personas), "results": results},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"保存: {Path(args.out).resolve()}\n")


def cmd_target(args):
    doc = json.loads(Path(args.proposals).read_text(encoding="utf-8"))
    item = doc.get("cross_item", "外食")
    table, src, as_of = load_item_baseline(item)
    grounding = resolve_grounding(item, table, src, as_of)
    props = doc["proposals"]
    pr = next((p for p in props if p["id"] == args.id), props[0]) if args.id else props[0]

    src_data = pop.load_src()
    personas = pop.generate(src_data, args.frame, args.n, random.Random(args.seed))["personas"]
    rows = target(personas, pr, grounding, args.by)

    print()
    print("=" * 76)
    print(f"{C_BLD}接地ターゲティング：{item}「{pr['name']}」{pr['price_monthly']:,}円/月{C_END}")
    print(f"  枠 {args.frame} / {len(personas)}人 / セグメント軸 {args.by}")
    print("=" * 76)
    print(f"  {'当てやすさ順':<4}{'セグメント':<12}{'構成比':>8}{'平均価格受容':>12}{'無理ない層':>10}")
    for i, r in enumerate(rows, 1):
        c = C_CYA if r["mean_acceptance"] >= 0.5 else C_YEL if r["mean_acceptance"] >= 0.35 else C_RED
        print(f"  {i:<6}{str(r['segment']):<12}{r['share']*100:>7.0f}%"
              f"{c}{r['mean_acceptance']*100:>11.0f}%{C_END}{r['easy_share']*100:>9.0f}%")
    print("  " + "-" * 72)
    if grounding["mode"] == "segment":
        print(f"  {C_CYA}[接地]{C_END} {grounding['source']}（{grounding['as_of']}）。便益の仮定は不使用。")
    else:
        print(f"  {C_CYA}[接地]{C_END} すべて家計調査 {item} の五分位別月額由来（{as_of}）。便益の仮定は不使用。")
    cav = driver_caveat(item)
    if cav:
        print(f"  {C_RED}注意{C_END} {cav}")
    print(f"  {C_GRY}『無理ない層』＝価格が『妥当』以下に収まる人の割合。狙う層の目安。{C_END}")
    print("=" * 76)
    print()


def main():
    ap = argparse.ArgumentParser(description="価値創生の翼：施策案への反応比較")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="セグメントに複数案をぶつけて比較")
    r.add_argument("--frame", default="全国")
    r.add_argument("--n", type=int, default=500)
    r.add_argument("--where", nargs="*")
    r.add_argument("--proposals", default=str(HERE / "proposals_sample.json"))
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("accept", help="価格受容の五分位別内訳だけを見る")
    a.add_argument("--item", default="外食")
    a.add_argument("--price", type=int, required=True)
    a.add_argument("--axis", choices=["income", "age"],
                   help="接地軸。省略時は品目の駆動要因から自動選択")
    a.set_defaults(func=cmd_accept)

    t = sub.add_parser("target", help="1施策案を誰に当てるか（接地のみ）")
    t.add_argument("--frame", default="全国")
    t.add_argument("--n", type=int, default=2000)
    t.add_argument("--proposals", default=str(HERE / "proposals_sample.json"))
    t.add_argument("--id", help="対象の施策案ID（省略時は先頭）")
    t.add_argument("--by", default="income_quintile",
                   choices=["income_quintile", "has_child", "pet", "prefecture",
                            "dual_income", "household_size",
                            "industry", "occupation", "education",
                            "car", "housing_type", "financial_assets"])
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_target)

    args = ap.parse_args()
    try:
        return args.func(args) or 0
    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f"\n{C_RED}エラー: {e}{C_END}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
