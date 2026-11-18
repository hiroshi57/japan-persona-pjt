# 国内ペルソナ化プロジェクト（Persona Screening）

全国統計で接地した日本人の仮想ペルソナを都度生成し、基本×詳細×心理の3層でスクリーニングして、調査手法（アンケート／インタビュー／観察・行動／エスノグラフィー）で壁打ちするマーケティング調査ツール。本調査（実在者への調査）の前の仮説出し・案の足切りを目的とする。

## 構成

```
ui/        フロントエンド（単一HTMLのスクリーニング画面）
  template.html          雛形。__GENSPEC__ と __GENJS__ を差し込む
  genspec.json           全属性の分布・心理規則・価格基準（接地データ）
  gen.js                 クライアント側のペルソナ生成器（ガウシアンコピュラで相関を付与し周辺分布を保持）
  persona_screening.html ビルド済み単体HTML（成果物）
  build.py               template + genspec + gen.js → persona_screening.html を生成

engine/    Python生成エンジン（本体。厳密な結合分布を担保）
  population.py          人口駆動のペルソナ生成（接地24属性、属性ごとにTIER付与）
  source_data.json       全属性の出典とTIER
  psych_items.json       心理15項目（接地5＋方向性10）の定義
  psychographic.py       心理フラグの割り当て
  nemotron_enrich.py     Nemotronの語り（文化背景・趣味等）の付与
  reaction.py            施策案の反応分解（価格受容の接地）
  cross_engine.py / stats_utils.py / gen_pool.py / sanity_check.py  補助
  attributes.json / industry_matrix.json / proposals_*.json        定義・入力

data/      nemotron_japan_sample.parquet（NVIDIA Nemotron-Personas-Japan 5万行サンプル, CC-BY-4.0）

docs/      設計ドキュメント
  value-creation-wing.md        信頼度で追い越す設計（第2の翼）
  research-methods-gradient.md  調査手法の信頼度勾配（第3の翼）
  psychographic-screening.md    心理15項目の設計
  nemotron-personas-japan.md    Nemotron評価と統合
  usage-guidelines.md           使用ガイドラインと免責
  mealkit.md                    実案件ワークケース
  toppan-ai-persona.md          競合（TOPPAN）の整理
```

## 使い方（UI）

```powershell
cd ui
python build.py          # persona_screening.html を生成
```

ブラウザで persona_screening.html を開くと画面は動くが、対話・生成（アンケート回答／観察／エスノ）は claude.ai のアーティファクトで公開したときのみ動作する（Claudeに問い合わせる `sample` 能力を使うため）。ローカルの file:// では画面構成の確認のみ。

## 使い方（エンジン）

```powershell
cd engine
python gen_pool.py 2000 pool.json                 # 母集団2000人を生成
python reaction.py run --frame 全国 --n 500 --proposals proposals_mealkit.json
python sanity_check.py                            # 接地が崩れていないか点検
```

## データ出典（主）

国勢調査・労働力調査（令和7年平均）・家計調査・全国家計構造調査2024・家計の金融行動2023・生保文化センター・通信利用動向調査・情報通信メディア調査・自動車検査登録情報協会・警察庁 ほか。詳細は engine/source_data.json と docs/ を参照。

Nemotronの語りは **NVIDIA Nemotron-Personas-Japan（CC-BY-4.0）**。製品表示時は帰属表示が必須。

## 大容量ファイルの注意

data/nemotron_japan_sample.parquet は約22MB。GitHubの単一ファイル上限（100MB）内だが、Git LFS の利用を推奨。不要なら .gitignore で除外し、docs/nemotron-personas-japan.md の手順で再取得できる。

## 位置づけ・免責

接地／方向性の信頼度タグを無視して案の優劣をそのまま意思決定に使わないこと。購入意向などの態度の数値は実在者との一致を保証しない。詳細は docs/usage-guidelines.md。
