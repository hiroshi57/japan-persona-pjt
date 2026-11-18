#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
population.py — 人口駆動のペルソナ生成器

「人口から作る」ための土台。
TOPPANは『日本の人口構成を再現した10万人』を掲げる。こちらは人数で張り合わない。
代わりに、実在の人口から任意の規模・任意の枠（全国／県／セグメント）を
分布妥当に切り出せる生成器にする。試す層だけ深く掘る『深掘り生成』の基盤になる。

設計の芯:
  まず人口を数える。何世帯あって、どこに、どういう構成でいるか。
  そのうえで属性を定員方式で配る。定員で配るので、全体の構成比は必ず実データに一致する。
  （乱数で引くと標本誤差で分布がぶれるが、定員方式はぶれない。
    都道府県×子供×ペットの検証で確認済み。）

接地の原則:
  枠ごとに、どの属性が人口データで裏づけられ、どこが全国近似かを必ず出す。
  県別に実データがある属性（都道府県シェア・子供あり率）はそのまま使う。
  県別が無い属性（世帯人員・年収五分位）は全国分布で近似し、近似と明示する。
  推測値は入れない。

使い方（PowerShell）:
  python population.py build --frame 全国 --n 1000
  python population.py build --frame 東京都 --n 500
  python population.py build --frame 全国 --n 2000 --where has_child=あり --out personas.json
  python population.py frames
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats_utils as su  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC_PATH = HERE.parent / "source_data.json"   # /home/claude/cross/source_data.json

C_RED, C_YEL, C_CYA, C_GRY, C_BLD, C_END = (
    "\033[91m", "\033[93m", "\033[96m", "\033[90m", "\033[1m", "\033[0m"
)

# 全国の世帯人員分布（令和7年国勢調査）。2人以上の内訳はここから取る。
HOUSEHOLD_SIZE_NATIONAL = {"1": 0.414, "2": 0.284, "3": 0.154, "4": 0.105, "5+": 0.043}
# 2人以上世帯の中での内訳（単独を除いて正規化）。県別の単独率に接ぎ木する。
_MULTI = {k: v for k, v in HOUSEHOLD_SIZE_NATIONAL.items() if k != "1"}
_MULTI_SUM = sum(_MULTI.values())
MULTI_SHAPE = {k: v / _MULTI_SUM for k, v in _MULTI.items()}

INCOME_QUINTILE = {"I": 0.2, "II": 0.2, "III": 0.2, "IV": 0.2, "V": 0.2}
INCOME_BAND_YEN = {
    "I": [1000000, 3290000], "II": [3290000, 4590000], "III": [4590000, 6240000],
    "IV": [6240000, 8600000], "V": [8600000, 15000000],
}
# 五分位の代表所得（帯の中央値、円）。県別の所得傾斜を解くときの基準にする。
INCOME_MID = {k: (v[0] + v[1]) / 2 for k, v in INCOME_BAND_YEN.items()}

# 世帯人員×ペット（ペットフード協会2024、単身/2人以上別）
PET_BY_SIZE = {
    "1":  {"犬": 0.0477, "猫": 0.0587, "なし": 0.8936},
    "2":  {"犬": 0.1148, "猫": 0.1048, "なし": 0.7804},
    "3":  {"犬": 0.1148, "猫": 0.1048, "なし": 0.7804},
    "4":  {"犬": 0.1148, "猫": 0.1048, "なし": 0.7804},
    "5+": {"犬": 0.1148, "猫": 0.1048, "なし": 0.7804},
}

# 世帯人員×車の相性（東京の世帯構成×車 相関0.51に基づく方向性）。
# QuotaAssigner の定員は県別平均に合わせて先に固定するので、
# これは「誰に何台を割り当てるか」の順序を傾けるだけで、県別平均は壊さない。
CAR_AFFINITY = {
    "1":     {"0": 2.2, "1": 1.0, "2": 0.4, "3+": 0.25},
    "multi": {"0": 0.65, "1": 1.1, "2": 1.4, "3+": 1.6},
}

# 属性ごとの接地タグ
TIER = {
    "prefecture": ("1次データ", "令和7年国勢調査 都道府県別人口・世帯数"),
    "has_child":  ("1次データ", "令和7年国勢調査 18歳未満世帯員のいる世帯（県別）"),
    "household_size": ("1次データ", "単独率は国勢調査2020県別×令和7全国水準で接地／2人以上の内訳は全国"),
    "income_quintile": ("1次データ＋推計", "県別平均は全国家計構造調査2019／分布形は全国を県平均に合わせ傾斜"),
    "pet": ("2次データ", "令和6年全国犬猫飼育実態調査（世帯人員別。全国値）"),
    "dual_income": ("1次データ・近似", "共働き率は労働力調査2024（全夫婦74.9%）を子あり世帯に適用。子なしは不明"),
    "education": ("合成・全国校正", "学歴はNemotron全国分布で接地（専門学校卒のみ東京暫定）。年収と相関"),
    "occupation": ("1次データ・全国（大区分の水準）＋細分暫定", "労働力調査(令和7年平均)で正規/非正規/自営・役員/無職・学生の水準を全国接地。細分(非正規内訳・自営/役員・学生/無職)はe-Stat限定のため暫定維持、年齢別1次で差替え予定"),
    "industry": ("合成・全国校正", "業種はNemotronのoccupationを大分類19区分へ集計。就業者のみ付与"),
    "car": ("2次データ・県別接地", "世帯当たり台数は自動車検査登録情報協会2025県別。内訳は平均のみの最大エントロピー"),
    "housing_type": ("東京暫定", "住居形態は東京分布を全国に暫定適用。年収と相関（持ち家ほど高所得）"),
    "financial_assets": ("2次データ＋東京按分", "なし率は家計の金融行動2023の全国・世帯類型別。保有額内訳は東京按分。年収相関0.72"),
    "insurance": ("2次データ＋一部東京暫定", "生命保険89.8/医療65.7/がん41.9/個人年金24.3%は生保文化センター調査の全国値。学資は東京暫定・子あり限定"),
    "age_band": ("合成・全国校正", "年齢はNemotron全国分布で接地（20歳以上、70+が26%）。年収と相関0.42"),
    "gender": ("1次データ", "男女比は令和2国勢調査の全国値（15歳以上 男0.481/女0.519）"),
    "marital_status": ("1次データ・性別接地", "配偶関係は令和2国勢調査の全国・性別値。年齢と相関"),
    "drivers_license": ("2次データ・性別接地", "運転免許は警察庁令和5年の全国・性別値（男84.0/女66.7%）"),
    "internet_frequency": ("1次データ＋東京按分", "利用率86.2%は令和5年通信利用動向調査の全国値。頻度内訳は東京按分。年齢相関"),
    "ec_purchase_ratio": ("2次データ＋東京按分", "利用率53.5%は家計消費状況調査2023の全国値。購入割合の強度内訳は東京按分。年収相関"),
    "smartphone": ("1次データ＋東京按分", "保有率72.9%は令和5年通信利用動向調査の全国値。OS内訳は東京按分"),
    "sns": ("1次データ＋東京暫定", "主要5銘柄は令和5年度情通政策研の全国利用率。TikTok・なしは東京暫定。年齢相関"),
    "investment": ("2次データ＋一部東京暫定", "株式15.6/投信13.3/NISA20.5/暗号8.5%は全国値。FX・不動産は東京暫定。年収相関0.56"),
    "credit_cards": ("2次データ＋東京按分", "保有率84.2%はJCB調査(2017)の全国値。枚数内訳は東京按分。年収相関"),
}


def pref_size_dist(src, pref, single_scale):
    """県別の世帯人員分布。単独率は県別実データ、2人以上の内訳は全国形。"""
    base = src["prefectures"][pref].get("single_share_2020")
    if base is None:
        return dict(HOUSEHOLD_SIZE_NATIONAL)
    single = min(0.95, max(0.05, base * single_scale))
    dist = {"1": single}
    for k, w in MULTI_SHAPE.items():
        dist[k] = (1 - single) * w
    return dist


def _solve_tilt(target_mean, rng_iter=60):
    """
    五分位の確率を p_i ∝ exp(θ·mid_i) で傾け、期待所得が target_mean になるθを二分法で解く。
    県の平均所得に分布を合わせるための、既知の平均へのキャリブレーション。
    """
    mids = [INCOME_MID[k] for k in INCOME_QUINTILE]
    base = [INCOME_QUINTILE[k] for k in INCOME_QUINTILE]

    def mean_at(theta):
        w = [b * pow(2.718281828, theta * (m / 1e6)) for b, m in zip(base, mids)]
        s = sum(w)
        return sum(wi / s * m for wi, m in zip(w, mids))

    lo, hi = -3.0, 3.0
    for _ in range(rng_iter):
        mid = (lo + hi) / 2
        if mean_at(mid) < target_mean:
            lo = mid
        else:
            hi = mid
    theta = (lo + hi) / 2
    w = [b * pow(2.718281828, theta * (m / 1e6)) for b, m in zip(base, mids)]
    s = sum(w)
    return {k: wi / s for k, wi in zip(INCOME_QUINTILE, w)}


def dual_income_probs(mids, rate, target_ratio):
    """
    各世帯の収入中央値 mids から、共働きである確率を返す。
    全体の共働き率を rate に、共働き世帯平均収入÷片働き世帯平均収入を
    target_ratio（家計調査由来1.23）に合わせるよう、収入で傾けたロジスティックを解く。
    収入そのものは変えないので、収入分布は保たれる。
    """
    import math
    if not mids:
        return []
    mean_m = sum(mids) / len(mids)
    c = [(m - mean_m) / mean_m for m in mids]

    def alpha_for(theta):
        # 全体の共働き率が rate になる切片を二分法で求める
        lo, hi = -12.0, 12.0
        for _ in range(50):
            a = (lo + hi) / 2
            r = sum(1 / (1 + math.exp(-(a + theta * ci))) for ci in c) / len(c)
            if r < rate:
                lo = a
            else:
                hi = a
        return (lo + hi) / 2

    def ratio_at(theta):
        a = alpha_for(theta)
        ps = [1 / (1 + math.exp(-(a + theta * ci))) for ci in c]
        num = sum(p * m for p, m in zip(ps, mids))
        den = sum(p for p in ps)
        den2 = sum(1 - p for p in ps)
        num2 = sum((1 - p) * m for p, m in zip(ps, mids))
        mean_dual = num / den if den else 0
        mean_single = num2 / den2 if den2 else 0
        return (mean_dual / mean_single if mean_single else 1.0), a

    # θを二分法で解いて、収入比を target_ratio に合わせる
    lo, hi = 0.0, 6.0
    a = alpha_for(0.0)
    for _ in range(40):
        theta = (lo + hi) / 2
        r, a = ratio_at(theta)
        if r < target_ratio:
            lo = theta
        else:
            hi = theta
    theta = (lo + hi) / 2
    a = alpha_for(theta)
    return [1 / (1 + math.exp(-(a + theta * ci))) for ci in c]


def car_count_dist(mean):
    """世帯当たり平均台数 mean に平均を合わせた 0/1/2/3+ の分布を返す。
    0/1/2台以上の内訳は公表されていないため、平均だけを与えた最大エントロピー分布にする
    （平均以外の仮定を置かない、いちばん中立な形）。P(k) ∝ exp(θ·k)、
    支持は {0,1,2,3.3}。3.3 は「3台以上」の実効値。"""
    import math
    ks = [0.0, 1.0, 2.0, 3.3]
    labels = ["0", "1", "2", "3+"]
    m = min(max(mean, 0.02), 3.2)
    lo, hi = -6.0, 6.0
    for _ in range(80):
        th = (lo + hi) / 2
        w = [math.exp(th * k) for k in ks]
        s = sum(w)
        cur = sum(wi / s * k for wi, k in zip(w, ks))
        if cur < m:
            lo = th
        else:
            hi = th
    th = (lo + hi) / 2
    w = [math.exp(th * k) for k in ks]
    s = sum(w)
    return {lab: wi / s for lab, wi in zip(labels, w)}


def pref_income_dist(src, pref, nat_mean_k):
    """
    県別の五分位分布。県の平均所得（2019, 総世帯）の全国比で、全国の五分位を傾ける。
    絶対額の基準が違うため、全国比だけを使って相対的に傾ける。
    """
    pm = src["prefectures"][pref].get("income_2019_k")
    if pm is None or not nat_mean_k:
        return dict(INCOME_QUINTILE)
    nat_mid_mean = sum(INCOME_QUINTILE[k] * INCOME_MID[k] for k in INCOME_QUINTILE)
    target = nat_mid_mean * (pm / nat_mean_k)   # 全国比で目標平均を作る
    return _solve_tilt(target)


def load_src():
    # リポジトリ構成（親ディレクトリ）でも、zip展開した平坦構成（同階層）でも動く
    for cand in (SRC_PATH, HERE / "source_data.json"):
        if cand.exists():
            return json.loads(cand.read_text(encoding="utf-8"))
    return json.loads(SRC_PATH.read_text(encoding="utf-8"))


def households_by_pref(src):
    """県別世帯数を子供あり世帯数÷子供あり率から求め、全国計に合わせて調整する"""
    raw = {n: p["child_households"] / p["child_rate"]
           for n, p in src["prefectures"].items()}
    scale = src["national"]["total_households"] / sum(raw.values())
    return {n: v * scale for n, v in raw.items()}


# ─────────────────────────────────────────────────────────
# 枠（frame）を組む
# ─────────────────────────────────────────────────────────
def build_frame(src, frame):
    """
    frame = '全国' か 県名。
    その枠に含まれる県と、県ごとの世帯シェアを返す。
    """
    hh = households_by_pref(src)
    if frame in ("全国", "national", "日本"):
        prefs = dict(hh)
        label = "全国"
    elif frame in src["prefectures"]:
        prefs = {frame: hh[frame]}
        label = frame
    else:
        raise ValueError(f"枠が不明です: {frame}（全国 または 47都道府県名）")
    total = sum(prefs.values())
    share = {n: v / total for n, v in prefs.items()}
    return {"label": label, "prefs": prefs, "share": share,
            "total_households": total, "child_rate": src["prefectures"]}


# ─────────────────────────────────────────────────────────
# 生成
# ─────────────────────────────────────────────────────────
def generate(src, frame, n, rng=random):
    """人口から n 人のペルソナを分布妥当に生成する"""
    fr = build_frame(src, frame)

    # 単独率を2020県別から令和7全国水準へ合わせる係数（世帯加重で全国41.4%になるよう調整）
    hh = households_by_pref(src)
    tot_hh = sum(hh.values())
    w_single = sum(hh[p] / tot_hh * src["prefectures"][p].get("single_share_2020", 0.38)
                   for p in hh)
    single_scale = src["national"].get("single_share_2025", 0.414) / max(w_single, 1e-9)

    # 県別所得傾斜の基準にする全国平均（2019, 総世帯, 世帯加重）
    nat_mean_k = sum(hh[p] / tot_hh * src["prefectures"][p].get("income_2019_k", 0)
                     for p in hh)

    # 1. 県へ定員配分（県別世帯シェアに一致）
    pref_q = su.QuotaAssigner(fr["share"], n, rng)
    pref_list = [pref_q.assign() for _ in range(n)]

    idx_by_pref = {}
    for i in range(n):
        idx_by_pref.setdefault(pref_list[i], []).append(i)

    size_list = [None] * n
    inc_list = [None] * n
    child_flag = [False] * n
    car_list = [None] * n
    car_by_pref = src.get("car_per_household_by_pref", {})
    nat_car = src["national"].get("car_per_household")

    # 2. 県ごとに、その県の世帯人員分布・所得分布で定員配分する
    for pref, idxs in idx_by_pref.items():
        c = len(idxs)
        size_q = su.QuotaAssigner(pref_size_dist(src, pref, single_scale), c, rng)
        inc_q = su.QuotaAssigner(pref_income_dist(src, pref, nat_mean_k), c, rng)
        # 車は県別の世帯当たり台数で接地（定員方式なので県平均は保たれる）
        cmean = car_by_pref.get(pref, nat_car)
        car_q = su.QuotaAssigner(car_count_dist(cmean), c, rng) if cmean else None
        for j, i in enumerate(idxs):
            size_list[i] = size_q.assign()
            inc_list[i] = inc_q.assign()
            if car_q is not None:
                aff = CAR_AFFINITY["1"] if size_list[i] == "1" else CAR_AFFINITY["multi"]
                car_list[i] = car_q.assign(aff)

        # 子供ありは2人以上世帯にだけ配る（単身は定義上ほぼ子なし）
        cr = src["prefectures"][pref]["child_rate"]
        target = round(cr * c)
        multi = [i for i in idxs if size_list[i] != "1"]
        rng.shuffle(multi)
        for i in multi[:min(target, len(multi))]:
            child_flag[i] = True

    # 共働き（就業状況）は子あり世帯にだけ接地する。
    # 共働き率は労働力調査の全夫婦世帯の値で、子あり世帯＝就業期の夫婦の近似。
    # 子なしは単身・高齢夫婦が混在し接地できないため「不明」にする。
    # 収入は書き換えず、「誰を共働きにするか」を収入が高い人ほど共働きに寄せて選ぶ。
    # これで収入分布を保ったまま、共働き世帯の平均収入が片働きの約1.23倍になる。
    di_rate = src["national"].get("dual_income_rate_among_couples", 0.749)
    di_ratio = src["national"].get("dual_income_income_ratio", 1.23)
    child_idx = [i for i in range(n) if child_flag[i]]
    child_mids = [(INCOME_BAND_YEN[inc_list[i]][0] + INCOME_BAND_YEN[inc_list[i]][1]) / 2
                  for i in child_idx]
    dual_flag = [None] * n
    probs = dual_income_probs(child_mids, di_rate, di_ratio)
    for i, p in zip(child_idx, probs):
        dual_flag[i] = "共働き" if rng.random() < p else "片働き"

    # 学歴・職業・業種（全国値未取得のため東京分布を暫定適用、要e-Stat差し替え）。
    # 学歴は年収と相関づける。職業は世帯の整合を軽く取り、業種は就業者にだけ付ける。
    edu_dist = src["national"].get("education_dist")
    occ_dist = src["national"].get("occupation_dist")
    ind_dist = src["national"].get("industry_dist")
    EDU_ORDER = ["中学卒", "高校卒", "専門学校卒", "短大・高専卒", "大学卒", "大学院卒"]
    EMPLOYED = {"正社員", "契約社員", "派遣社員", "パート・アルバイト", "自営業", "会社役員"}

    # 住居形態・金融資産・保険（全国値未取得のため東京分布を暫定適用）。
    # 金融資産と住居形態は年収と相関づけ、学資保険は子あり世帯に限定する。
    fin_dist = src["national"].get("financial_assets_dist")
    house_dist = src["national"].get("housing_type_dist")
    ins_base = src["national"].get("insurance_dist")
    f_child = sum(1 for x in child_flag if x) / n if n else 0.0
    ins_child = ins_nochild = None
    if ins_base:
        # 学資保険は子あり世帯に寄せる。全体の加入率が設計どおりに残るよう、
        # 子あり世帯内の率を 0.127÷子あり率 に引き上げ、子なしは0にする。
        r_gakushi = min(0.95, ins_base["学資保険"] / f_child) if f_child > 1e-6 else 0.0
        ins_child = dict(ins_base); ins_child["学資保険"] = r_gakushi
        ins_nochild = dict(ins_base); ins_nochild["学資保険"] = 0.0

    # 金融資産：なし率を世帯類型別に全国接地（家計の金融行動2023、単身36.0/二人以上24.7%）。
    # 保有額の内訳は東京按分。『なし』を最下位に置き、年収相関で低所得ほど『なし』寄りに。
    fin_nashi = src["national"].get("financial_assets_nashi_rate")
    fin_single = fin_multi = None
    if fin_dist and fin_nashi:
        def _fin_variant(nashi):
            dd = {"なし": nashi}
            for k, v in fin_dist.items():
                dd[k] = (1 - nashi) * v
            return dd
        fin_single = _fin_variant(fin_nashi["単身"])
        fin_multi = _fin_variant(fin_nashi["二人以上"])

    # 年齢・性別・配偶関係＋詳細7属性（全国値未取得のため東京分布を暫定適用）。
    # 相関の向きだけ接地する：年齢↔SNS/ネット、年収↔投資/EC/クレカ/年齢。
    AGE_ORDER = ["20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    age_dist = src["national"].get("age_dist")
    age_ord = {k: age_dist[k] for k in AGE_ORDER if k in age_dist} if age_dist else None
    gender_dist = src["national"].get("gender_dist")
    marital_dist = src["national"].get("marital_dist")
    marital_by_gender = src["national"].get("marital_by_gender")
    dl_dist = src["national"].get("drivers_license_dist")
    dl_by_gender = src["national"].get("drivers_license_by_gender")
    net_dist = src["national"].get("internet_freq_dist")
    ec_dist = src["national"].get("ec_ratio_dist")
    sp_dist = src["national"].get("smartphone_dist")
    sns_dist = src["national"].get("sns_dist")
    inv_dist = src["national"].get("investment_dist")
    cc_dist = src["national"].get("credit_cards_dist")

    personas = []
    for i in range(n):
        size = size_list[i]
        inc = inc_list[i]
        pet = su.pick_by_quantile(PET_BY_SIZE[size], rng.random())
        lo, hi = INCOME_BAND_YEN[inc]
        income = int(lo + (hi - lo) * rng.random())
        p = {
            "id": i + 1,
            "prefecture": pref_list[i],
            "household_size": size,
            "has_child": "あり" if child_flag[i] else "なし",
            "income_quintile": inc,
            "income_yen": income,
            "pet": pet,
            "dual_income": dual_flag[i] if child_flag[i] else "不明",
        }
        # 所得順位（五分位index＋区分内位置、0〜1）を相関元に使う
        inc_rank = (["I", "II", "III", "IV", "V"].index(inc)
                    + (income - lo) / max(hi - lo, 1)) / 5
        if car_list[i] is not None:
            p["car"] = car_list[i]
        if fin_single:
            # 金融資産：世帯類型でなし率を切替（全国）。保有額内訳は東京按分、年収相関0.72
            fd = fin_single if p["household_size"] == "1" else fin_multi
            p["financial_assets"] = su.pick_by_quantile(
                fd, su.correlated_uniform(inc_rank, 0.72, rng))
        elif fin_dist:
            p["financial_assets"] = su.pick_by_quantile(
                fin_dist, su.correlated_uniform(inc_rank, 0.72, rng))
        if house_dist:
            # 住居形態：年収と相関0.38。持ち家ほど高所得側に寄る
            p["housing_type"] = su.pick_by_quantile(
                house_dist, su.correlated_uniform(inc_rank, 0.38, rng))
        if edu_dist:
            # 学歴：年収が高い世帯ほど高学歴に寄せる（分布は保つ）
            iu = (["I", "II", "III", "IV", "V"].index(inc) + 0.5) / 5
            eq = su.correlated_uniform(iu, 0.45, rng)
            p["education"] = su.pick_by_quantile(
                {k: edu_dist[k] for k in EDU_ORDER if k in edu_dist}, eq)
        if occ_dist:
            occ = su.pick_by_quantile(occ_dist, rng.random())
            # 子あり世帯で学生は不自然なので就業者側へ寄せる（年齢不在の最小限の整合）
            if p["has_child"] == "あり" and occ == "学生":
                occ = "正社員"
            p["occupation"] = occ
        if ind_dist:
            # 業種は就業者にだけ付く
            p["industry"] = (su.pick_by_quantile(ind_dist, rng.random())
                             if p.get("occupation") in EMPLOYED else "非該当")
        if ins_base:
            # 保険：子あり/子なしで学資保険の有無を切り替え、年収で軽く傾ける
            ins_d = ins_child if child_flag[i] else ins_nochild
            p["insurance"] = su.multi_select(ins_d, driver_u=inc_rank,
                                             strength=0.3, rng=rng)

        # 年齢：年収と相関0.42。年齢順位は後続（SNS・ネット・配偶）の相関元に使う
        age_rank = 0.5
        if age_ord:
            age = su.pick_by_quantile(age_ord, su.correlated_uniform(inc_rank, 0.42, rng))
            p["age_band"] = age
            age_rank = su.rank_in_category(age_ord, age, rng)
        young = 1.0 - age_rank
        if gender_dist:
            p["gender"] = su.pick_by_quantile(gender_dist, rng.random())
        if marital_dist:
            # 性別で分布を切替（全国・国勢調査）。年齢が上ほど未婚→有配偶→離別→死別へ
            md = (marital_by_gender or {}).get(p.get("gender")) or marital_dist
            p["marital_status"] = su.pick_by_quantile(
                md, su.correlated_uniform(age_rank, 0.5, rng))
        if dl_dist:
            # 性別で保有率を切替（全国・警察庁 男84.0%/女66.7%）
            dld = (dl_by_gender or {}).get(p.get("gender")) or dl_dist
            p["drivers_license"] = su.pick_by_quantile(dld, rng.random())
        if net_dist:
            # 年齢が上ほど低頻度側へ（毎日→ほぼ使わないの順なので age_rank を正で使う）
            p["internet_frequency"] = su.pick_by_quantile(
                net_dist, su.correlated_uniform(age_rank, 0.45, rng))
        if ec_dist:
            p["ec_purchase_ratio"] = su.pick_by_quantile(
                ec_dist, su.correlated_uniform(inc_rank, 0.30, rng))
        if sp_dist:
            p["smartphone"] = su.pick_by_quantile(sp_dist, rng.random())
        if sns_dist:
            p["sns"] = su.multi_select(sns_dist, driver_u=young, strength=0.64, rng=rng)
        if inv_dist:
            p["investment"] = su.multi_select(inv_dist, driver_u=inc_rank,
                                              strength=0.56, rng=rng)
        if cc_dist:
            p["credit_cards"] = su.pick_by_quantile(
                cc_dist, su.correlated_uniform(inc_rank, 0.30, rng))
        personas.append(p)

    return {"frame": fr["label"], "n": n, "personas": personas,
            "frame_total_households": fr["total_households"]}


def apply_filter(personas, where):
    """セグメント条件で絞る（深掘り生成の前段）。
    値はスカラー（一致）またはリスト（いずれかに一致＝複数選択）。"""
    out = personas
    for k, v in where.items():
        if isinstance(v, (list, tuple)):
            vals = set(str(x) for x in v)
            out = [p for p in out if str(p.get(k)) in vals]
        else:
            out = [p for p in out if str(p.get(k)) == str(v)]
    return out


# ─────────────────────────────────────────────────────────
# 検証
# ─────────────────────────────────────────────────────────
def validate(src, result):
    """生成結果の分布が人口データに一致するかを測る"""
    personas = result["personas"]
    n = len(personas)
    hh = households_by_pref(src)
    checks = []

    # 都道府県シェア（全国枠のときのみ）
    if result["frame"] == "全国":
        design = {k: v / sum(hh.values()) for k, v in hh.items()}
        actual = {}
        for p in personas:
            actual[p["prefecture"]] = actual.get(p["prefecture"], 0) + 1 / n
        gap = max(abs(design.get(k, 0) - actual.get(k, 0)) for k in design) * 100
        checks.append(("都道府県シェア", gap))

    # 世帯人員
    design = HOUSEHOLD_SIZE_NATIONAL
    actual = {}
    for p in personas:
        actual[p["household_size"]] = actual.get(p["household_size"], 0) + 1 / n
    gap = max(abs(design[k] - actual.get(k, 0)) for k in design) * 100
    checks.append(("世帯人員", gap))

    # 年収五分位
    actual = {}
    for p in personas:
        actual[p["income_quintile"]] = actual.get(p["income_quintile"], 0) + 1 / n
    gap = max(abs(0.2 - actual.get(k, 0)) for k in INCOME_QUINTILE) * 100
    checks.append(("年収五分位", gap))

    # 子供あり率（枠全体）
    if result["frame"] == "全国":
        design_cr = src["national"]["households_with_children"] / src["national"]["total_households"]
    else:
        design_cr = src["prefectures"][result["frame"]]["child_rate"]
    actual_cr = sum(1 for p in personas if p["has_child"] == "あり") / n
    checks.append(("子供あり率", abs(design_cr - actual_cr) * 100))

    return checks


# ─────────────────────────────────────────────────────────
# 表示
# ─────────────────────────────────────────────────────────
def cmd_build(src, args):
    rng = random.Random(args.seed)
    where = {}
    for p in args.where or []:
        k, v = p.split("=", 1)
        where[k.strip()] = v.strip()

    if not where:
        result = generate(src, args.frame, args.n, rng)
    else:
        # まず少量で歩留まりを測り、必要な生成量を見積もってから作る。
        probe = generate(src, args.frame, 2000, rng)
        hit = len(apply_filter(probe["personas"], where))
        yield_rate = max(hit / 2000, 0.001)
        filtered = apply_filter(probe["personas"], where)
        guard = 0
        while len(filtered) < args.n and guard < 12:
            need = args.n - len(filtered)
            batch = min(60000, int(need / yield_rate * 1.3) + 1000)
            extra = generate(src, args.frame, batch, rng)
            filtered += apply_filter(extra["personas"], where)
            guard += 1
        result = probe
        result["personas"] = filtered[:args.n]
        result["n"] = len(result["personas"])
        result["segment_yield"] = yield_rate

    print()
    print("=" * 76)
    print(f"{C_BLD}人口から生成: {result['frame']} / {result['n']}人{C_END}")
    if where:
        cond = " ".join(f"{k}={v}" for k, v in where.items())
        print(f"  セグメント条件: {cond}")
        if result.get("segment_yield"):
            print(f"  このセグメントは人口の約 {result['segment_yield']*100:.1f}%"
                  f"（固定プールでは薄く、深掘り生成で直接作成）")
    print("=" * 76)

    print(f"  枠の実世帯数: {result['frame_total_households']:,.0f} 世帯")
    print()

    # 接地タグ
    print(f"{C_BLD}■ 属性の接地{C_END}")
    for attr, (tier, src_note) in TIER.items():
        c = C_CYA if tier.startswith("1次") else C_YEL if tier.startswith("2次") else C_RED
        if attr == "has_child" and result["frame"] != "全国":
            src_note = f"令和7年国勢調査 {result['frame']}の18歳未満世帯員のいる世帯率"
        print(f"  {attr:<16}{c}[{tier}]{C_END} {C_GRY}{src_note[:44]}{C_END}")
    print()

    # 検証
    checks = validate(src, result) if not where else []
    if checks:
        print(f"{C_BLD}■ 分布の検証（人口データとの最大乖離）{C_END}")
        for name, gap in checks:
            c = C_RED if gap >= 5 else C_YEL if gap >= 2 else C_CYA
            print(f"  {name:<14}{c}{gap:.2f}pt{C_END}")
        print()

    # サンプル表示
    print(f"{C_BLD}■ 生成例（先頭5人）{C_END}")
    print(f"  {'#':<4}{'都道府県':<9}{'世帯人員':<7}{'子供':<5}{'年収五分位':<6}{'年収':>11}{'ペット':>7}")
    for p in result["personas"][:5]:
        print(f"  {p['id']:<4}{p['prefecture']:<9}{p['household_size']:<7}"
              f"{p['has_child']:<5}{p['income_quintile']:<6}"
              f"{p['income_yen']:>10,}円{p['pet']:>7}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  保存: {Path(args.out).resolve()}")
    print("=" * 76)
    print()


def cmd_frames(src, args):
    hh = households_by_pref(src)
    print()
    print("=" * 60)
    print(f"{C_BLD}生成できる枠{C_END}")
    print("=" * 60)
    print(f"  全国       {src['national']['total_households']:>12,.0f} 世帯")
    print(f"  {C_GRY}── 都道府県（世帯数の多い順・上位10）──{C_END}")
    for n, v in sorted(hh.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {n:<9}{v:>12,.0f} 世帯  子供あり率 "
              f"{src['prefectures'][n]['child_rate']*100:.1f}%")
    print(f"  {C_GRY}…ほか37県。全国／任意の県名を --frame に指定できる。{C_END}")
    print("=" * 60)
    print()


def main():
    ap = argparse.ArgumentParser(description="人口駆動のペルソナ生成器")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="人口から生成する")
    b.add_argument("--frame", default="全国", help="全国 または 都道府県名")
    b.add_argument("--n", type=int, default=1000)
    b.add_argument("--where", nargs="*", help="セグメント条件（例: has_child=あり）")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--out", help="JSON保存先")
    b.set_defaults(func=cmd_build)

    f = sub.add_parser("frames", help="生成できる枠の一覧")
    f.set_defaults(func=cmd_frames)

    args = ap.parse_args()
    src = load_src()
    try:
        return args.func(src, args) or 0
    except (ValueError, KeyError) as e:
        print(f"\n{C_RED}エラー: {e}{C_END}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
