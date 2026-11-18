#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_utils.py — 分布を壊さずに相関をつけるための道具

これまでの生成コードは、設計分布から引いたあとに補正を掛けっぱなしにしていた。
そのため補正が強い項目ほど分布が崩れ、金融資産では設計18.3%の区分が0.1%になった。

ここでは「相関はつけるが、構成比は必ず設計どおりに保つ」3つの方法を用意する。

  1. 順序のある項目（金額・頻度・程度）
     → ガウス・コピュラで順位を相関させる。構成比は数学的に保たれる。

  2. 順序のない項目（職業・情報源）
     → 定員を先に決めて、相性の良い人から埋める。構成比は定員どおりになる。

  3. 複数選択の項目（SNS・投資・保険）
     → 「なし」を先に排他で決め、残りを条件付き確率で引く。
       補正係数は平均が1になるよう正規化するので、全体の選択率は変わらない。
"""

import math
import random


# ─────────────────────────────────────────────────────────
# 正規分布の累積分布関数とその逆関数
# scipy を使わずに済ませるため自前で持つ
# ─────────────────────────────────────────────────────────
def norm_cdf(x):
    """標準正規分布の累積分布関数"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p):
    """
    標準正規分布の逆累積分布関数（パーセント点）
    Acklam の有理関数近似。絶対誤差は約1.15e-9で、本用途には十分。
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low, p_high = 0.02425, 1 - 0.02425

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


# ─────────────────────────────────────────────────────────
# 1. 順序のある項目: ガウス・コピュラ
# ─────────────────────────────────────────────────────────
def correlated_uniform(driver_u, rho, rng=random):
    """
    0〜1の値 driver_u と相関 rho を持つ、別の0〜1の値を返す。

    肝心な点は、返り値が必ず0〜1の一様分布になること。
    一様分布であれば、それを累積分布に通すだけで
    設計した構成比をそのまま再現できる。

    driver_u : 相関元の順位（0〜1）。たとえば年収の上位何%か。
    rho      : 相関係数（-1〜1）。0なら無相関。
    """
    u = min(max(driver_u, 1e-6), 1 - 1e-6)
    rho = max(-0.999, min(0.999, rho))

    z_driver = norm_ppf(u)
    z_noise = norm_ppf(rng.random())
    z = rho * z_driver + math.sqrt(1 - rho * rho) * z_noise
    return norm_cdf(z)


def pick_by_quantile(ordered_dist, q):
    """
    順序つきの構成比から、累積確率 q の位置にある区分を返す。

    ordered_dist : 小さい順に並んだ {区分名: 割合}。並び順が意味を持つ。
    q            : 0〜1の値。

    q が一様分布であれば、返る区分の構成比は ordered_dist そのものになる。
    """
    total = sum(ordered_dist.values())
    cum = 0.0
    items = list(ordered_dist.items())
    for name, w in items:
        cum += w / total
        if q <= cum:
            return name
    return items[-1][0]


def correlated_choice(ordered_dist, driver_u, rho, rng=random):
    """
    順序つき項目を、driver_u と相関 rho を保ったまま選ぶ。
    構成比は ordered_dist どおりに保たれる。
    """
    q = correlated_uniform(driver_u, rho, rng)
    return pick_by_quantile(ordered_dist, q)


def value_in_band(band_name, band_ranges, q_within=None, rng=random):
    """
    区分名から実数値を作る。
    q_within を渡すと区分内の位置も指定でき、相関を区分内まで反映できる。
    """
    lo, hi = band_ranges[band_name]
    if hi <= lo:
        return lo
    if q_within is None:
        q_within = rng.random()
    return int(lo + (hi - lo) * q_within)


# ─────────────────────────────────────────────────────────
# 2. 順序のない項目: 定員つき割り当て
# ─────────────────────────────────────────────────────────
class QuotaAssigner:
    """
    構成比どおりの定員を先に用意し、相性の良い人から埋めていく。

    これまでは「65歳以上なら正社員の重みを0.1倍」のような補正を
    掛けていたが、元の構成比はすでに全年齢込みの値なので、
    そこへ補正を重ねると二重計上になり、全体の構成比が崩れた。

    この方式なら定員が尽きるまで配るだけなので、
    全体の構成比は設計どおりになり、かつ相性も反映される。
    """

    def __init__(self, dist, n, rng=random):
        self.rng = rng
        self.remaining = self._allocate(dist, n)

    @staticmethod
    def _allocate(dist, n):
        """割合から整数の定員を作る。端数は大きい順に配る。"""
        total = sum(dist.values())
        raw = {k: v / total * n for k, v in dist.items()}
        quota = {k: int(v) for k, v in raw.items()}
        short = n - sum(quota.values())
        # 小数部の大きい区分から1ずつ足す
        order = sorted(raw, key=lambda k: raw[k] - int(raw[k]), reverse=True)
        for k in order[:short]:
            quota[k] += 1
        return quota

    def assign(self, affinity=None):
        """
        1人分を割り当てる。

        affinity : {区分名: 相性の強さ} を渡すと、相性に応じて選ばれやすくなる。
                   省略すると残り定員に比例して選ばれる。
        """
        available = {k: c for k, c in self.remaining.items() if c > 0}
        if not available:
            return None

        if affinity:
            weights = [available[k] * max(affinity.get(k, 1.0), 1e-6)
                       for k in available]
        else:
            weights = [available[k] for k in available]

        if sum(weights) <= 0:
            weights = [available[k] for k in available]

        picked = self.rng.choices(list(available), weights=weights)[0]
        self.remaining[picked] -= 1
        return picked

    def left(self):
        return sum(self.remaining.values())


# ─────────────────────────────────────────────────────────
# 3. 複数選択の項目: 「なし」を排他にする
# ─────────────────────────────────────────────────────────
def balanced_factor(driver_u, strength):
    """
    平均がちょうど1になる補正係数を返す。

    これまでの補正は平均が1になっていなかった。
    たとえばSNSの年齢補正は平均が約0.72だったため、
    全選択肢が一律に3割近く減り、LINEが87.2%から62.8%に落ちた。

    driver_u : 0〜1の順位
    strength : 傾きの強さ（0で補正なし、0.5なら0.5〜1.5の範囲で動く）
    """
    return 1.0 + strength * (2.0 * driver_u - 1.0)


def multi_select(dist, driver_u=0.5, strength=0.0, none_key="なし", rng=random):
    """
    複数選択の項目を、全体の選択率を保ったまま選ぶ。

    これまでは各項目を独立に引いていたため、
    保険6種がすべて外れる確率は0.3%にしかならず、
    設計の「なし16.2%」と数学的に両立しなかった。

    ここでは「なし」を先に排他で決める。
    「なし」でなかった人の中で各項目を p/(1-p_none) の確率で引けば、
    全体で見た選択率は設計どおり p に戻る。

    dist     : {項目名: 選択率}。none_key を含む。
    driver_u : 相関元の順位（0〜1）。年齢や年収など。
    strength : 傾きの強さ。平均1の係数になるので全体の選択率は変わらない。
    """
    p_none_base = dist.get(none_key, 0.0)

    # 「なし」かどうか自体にも相関をつける。
    # これを固定にすると、たとえば投資経験が年収でほとんど変わらなくなる。
    # 係数の平均は1なので「なし」の割合は設計どおりに保たれる。
    # 項目側と同じ理由で、ここでも上限1.0での頭打ちを避ける。
    # p_none * (1 + s) が1を超えない範囲に傾きを抑える。
    if p_none_base > 1e-9:
        s_none = min(strength, 1.0 / p_none_base - 1.0)
    else:
        s_none = 0.0
    p_none = min(1.0, max(0.0, p_none_base * centered(1 - driver_u, max(0.0, s_none))))

    if rng.random() < p_none:
        return [none_key]

    # 条件付き確率の分母には、その人自身の「なし」確率を使う。
    # こうすると、選択率の期待値は p_i のまま変わらない。
    denom = max(1.0 - p_none, 1e-9)

    chosen = []
    for key, p in dist.items():
        if key == none_key:
            continue
        p_base = min(1.0, p / denom)

        # 選択率がもともと高い項目に強い補正を掛けると、確率が1.0で頭打ちになり、
        # 上振れだけが切り捨てられて全体の選択率が下がる。
        # LINEのように9割以上が選ぶ項目で起きるため、
        # 上限までの余裕に応じて傾きを弱める。
        damped = strength * (1.0 - p_base)
        p_cond = min(1.0, p_base * balanced_factor(driver_u, damped))

        if rng.random() < p_cond:
            chosen.append(key)

    if chosen:
        return chosen

    # 1つも選ばれなかった場合は、最も選択率の高い項目を1つ入れる。
    # ここで「なし」に落とすと、「なし」の割合が設計より増えてしまう。
    candidates = {k: v for k, v in dist.items() if k != none_key}
    if not candidates:
        return [none_key]
    best = max(candidates, key=candidates.get)
    return [best]


# ─────────────────────────────────────────────────────────
# 順位の計算
# ─────────────────────────────────────────────────────────
def percentile_of(value, band_ranges, dist):
    """
    実数値が、設計分布の中で下から何%の位置にあるかを返す。
    相関の元（driver_u）として使う。
    """
    total = sum(dist.values())
    cum = 0.0
    for name, w in dist.items():
        lo, hi = band_ranges.get(name, (0, 0))
        share = w / total
        if value < hi or hi <= lo:
            within = 0.5 if hi <= lo else (value - lo) / max(hi - lo, 1)
            within = min(max(within, 0.0), 1.0)
            return min(max(cum + share * within, 0.0), 1.0)
        cum += share
    return 0.99


def age_percentile(age, lo=20, hi=85):
    """年齢を0〜1に直す。若いほど0に近い。"""
    return min(max((age - lo) / (hi - lo), 0.0), 1.0)


def rank_in_category(dist, chosen_key, rng=random):
    """
    選ばれた区分が、分布全体の中で占める範囲のどこかを一様に返す。

    なぜ必要か:
      correlated_uniform は「相関元が0〜1の一様分布である」ことを前提にする。
      前提が崩れると出力も偏り、構成比が保てない。

      たとえば学歴を順位に直すとき、「大学卒=0.82」のように固定値を割り当てると、
      全体で見た分布は6つの点に集中する。これを相関元に使うと、
      年収の下位区分がほとんど選ばれなくなる。実際、最初の修正では
      個人年収「100万円未満」が設計11.2%に対し2.2%しか出なかった。

      区分の持ち分の中で一様に散らせば、全体として一様分布になり、前提が満たされる。
    """
    total = sum(dist.values())
    cum = 0.0
    for name, w in dist.items():
        share = w / total
        if name == chosen_key:
            return cum + share * rng.random()
        cum += share
    return rng.random()


def combine_drivers(parts, rho_between=0.0):
    """
    複数の相関元を1つにまとめる。返り値は0〜1の一様分布になる。

    単純に重み付き平均を取ると、一様分布どうしを足しても一様にはならず、
    真ん中に寄ってしまう（2つの平均なら三角分布になる）。
    そのまま相関元に使うと、分布の両端が痩せる。

    正規分布の世界に移してから足し、分散で割り戻してから戻せば、
    結果は正しく一様分布になる。

    parts       : [(0〜1の値, 重み), ...]
    rho_between : 相関元どうしの相関。0のままにすると、
                  互いに関係のある値（年収と年齢など）を混ぜたときに
                  ばらつきを小さく見積もってしまい、結果が
                  過剰に散らばって分布の両端が太る。
                  実際、金融資産の最上位区分が設計10.4%に対し
                  16%前後まで膨らむ原因になっていた。
    """
    zs = []
    for u, w in parts:
        u = min(max(u, 1e-6), 1 - 1e-6)
        zs.append((norm_ppf(u), w))

    num = sum(z * w for z, w in zs)
    var = sum(w * w for _, w in zs)
    # 相関がある分だけ分散は大きくなる。その分を足してから割る。
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            var += 2 * zs[i][1] * zs[j][1] * rho_between
    if var <= 0:
        return 0.5
    return norm_cdf(num / math.sqrt(var))


def centered(u, strength):
    """
    平均が0.5でない値から、平均1の補正係数を作る。

    balanced_factor は相関元が一様（平均0.5）である前提なので、
    偏った値をそのまま渡すと係数の平均が1からずれ、全体の割合が動く。
    実際、運転免許でこれが起き、保有率が設計より7ポイント高く出た。
    """
    return 1.0 + strength * (2.0 * min(max(u, 0.0), 1.0) - 1.0)
