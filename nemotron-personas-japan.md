# NVIDIA Nemotron-Personas-Japan 評価と統合設計

確認日：2026-11-11 ／ 検証日：2026-11-12 ／ 出典：https://huggingface.co/datasets/nvidia/Nemotron-Personas-Japan

## 何か

NVIDIAのDataDesignerが、実世界の人口統計・地理・性格分布に基づいて生成した日本語の合成ペルソナ100万行。ライセンスは CC-BY-4.0（表示すれば商用可）。22カラム。

- 構造化属性：sex・age(整数)・marital_status（「離別 (子供あり)」等、子の有無を含む）・education_level（「大学卒 理系」等）・occupation（「介護福祉業 中堅」等、産業＋役職の自由記述）・region・area・prefecture・country。
- 自然言語フィールド：cultural_background、skills_and_expertise（＋_list）、hobbies_and_interests（＋_list）、career_goals_and_ambitions、persona要約、professional/sports/arts/travel/culinary の5観点ナラティブ。

## このプロジェクトでの位置づけ

競合ではなく補完。population.py は消費・経済属性を実統計で接地する。Nemotronはそれが持たない心理・ナラティブ層（性格・スキル・趣味・キャリア・文化背景）と基本デモグラを提供する。自然言語フィールドは全てLLM生成＝信頼度「方向性のみ」。人口構成は接地、語りの中身は創作、と扱う。

## 統合方式（実装済み：engine/nemotron_enrich.py）

当エンジンの接地ペルソナに Nemotron の語りを付与する（Nemotronを土台にしない）。突合キーは性別・年齢帯・都道府県・配偶関係・学歴で、一致が無ければ段階的に緩める。付与時に _tier="方向性のみ"・_source・突合粒度を明記。

## 本データ検証（2026-11-12、5万行サンプル）

ユーザーがColab経由で書き出した5万行サンプル（nemotron_japan_sample.parquet、22MB、engine同梱・自動検出）で検証。

突合品質：全国500〜800人を生成して突合したところ、最細粒度（性別＋年齢＋県＋配偶＋学歴）で72%、県レベルまで含めれば100%一致。5行サンプル時の「性別のみ」転落は解消。付与は整合的（例：大阪20代未婚中学卒→大阪出身の文化背景・たこ焼き作り・堅実な金銭感覚）。

分布の相互検証（Nemotron実測 対 当方の全国接地）：
- 性別：Nemotron 男0.478/女0.522 ≒ 当方 男0.481/女0.519（一致）。
- 配偶関係：Nemotron 有配偶約59.8%・未婚約25.7% ≒ 当方 有配偶55.6%・未婚29.5%（近い）。
- 年齢：Nemotron 70歳以上25.6%・中央値53歳。当方の東京暫定（70+が11.1%）より明らかに全国実態に近い。
- 学歴：Nemotron 高校卒38.6%・大学卒24%。当方の東京暫定（大学卒偏重）より全国的。

## 追加の好機：年齢・学歴の接地

当方が政府PDFの壁で接地できずにいた年齢・学歴を、Nemotronの全国校正済み分布で代替できる。ただしNemotronは合成データ（NVIDIAの校正）であり1次ではない。信頼度は「合成・全国校正」＝東京暫定より実態に近いが1次接地には劣る、という中間タグが妥当。年齢は東京偏りが大きく代替の価値が高い。採否は信頼度方針に関わる判断のためユーザー確認事項。

## データ取得の制約

当実行環境は huggingface.co のデータ配信をプロキシが遮断。本データはColabノート（colab_nemotron.ipynb）で軽量サンプルを書き出し→チャット添付→当方統合、の運用。全件100万行に拡張する場合も同手順でサンプル行数を増やす。

## 残論点

- occupation自由記述（産業＋規模）は当エンジンの業種に通じる。業種の全国化にも転用余地。
- 表示時は CC-BY-4.0 の帰属表示（NVIDIA Nemotron-Personas-Japan）を製品UIに必須。
- 語りは全て「方向性のみ」。実在個人の事実として提示しない。
