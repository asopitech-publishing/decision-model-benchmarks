# Decision Model Benchmarks

Strands Decider 2B、Clef-flash、Jevの判定品質を比較するベンチマークコードです。日本語20問のルーティング評価と、24件×3設問の問い合わせ分類を収録しています。layaの測定コードと生結果は再現用の記録として残し、現在のモデル比較から外しています。記事本文は別の非公開リポジトリ [`substack-manuscripts`](https://github.com/asopitech-publishing/substack-manuscripts) で管理します。

10月7日のバッチ試験は短文24件だけを使い、長い入力や長短混合を測っていません。公開済みの生データは保持しますが、件数/秒の順位やバッチ数の推奨には使いません。[試験範囲と撤回した解釈](BATCH_RESULTS.md)を先に確認してください。

## 内容

- `benchmarks/compare_four.py` — 4モデル共通のリクエスト、品質・応答時間・並列処理量の測定。HTTPの生レスポンス、usage、SHA-256を結果JSONへ保存します。
- `benchmarks/test_compare_four.py` — API課金なしで実行できる単体テスト。
- `benchmarks/kiro_article_cases.json` — [クラスメソッドの検証記事](https://dev.classmethod.jp/articles/strands-decider-2b-m4-mac-kiro-crew/)の日本語20問と、共通の`B_desc`判定文。`prepare_kiro_article.py`で元記事から再生成できます。
- `benchmarks/clef_vs_laya_cases.json` — 今回作成した日本語サポート問い合わせ24件、担当部署・緊急度・返金要求の3設問。
- [`RESULTS.md`](RESULTS.md) — 2026年10月6日の品質・並列性能の集計と評価上の注意点。
- [`BATCH_RESULTS.md`](BATCH_RESULTS.md) — 短文バッチ試験で確認できた範囲と、性能比較から外した理由。
- [`REPLICA_RESULTS.md`](REPLICA_RESULTS.md) — StrandsとClefを1・2・4プロセス常駐させた短文負荷のHTTP応答記録。
- [`PRECISION_RESULTS.md`](PRECISION_RESULTS.md) — laya FP16/q8の判定記録と、性能試験の生結果の保存先。
- [`results/2026-10-06/`](results/2026-10-06/) — 同日の公開用生結果12ファイル。HTTP応答本文、usage、各リクエストの計測値を収録。
- [`results/2026-10-07/`](results/2026-10-07/) — バッチ推論・複数インスタンス・FP16/q8比較の公開用生結果JSON19ファイルと適用範囲の説明。
- `benchmarks/export_public_results.py` — 元の測定結果から公開用ファイルを作り、応答本文のハッシュと機密情報を検査。
- `benchmarks/bench_model_batch.py`、`benchmarks/*_batch.py` — 独立した要求を、各モデル1インスタンスで実際に1回の推論へまとめる測定。
- `benchmarks/batch_server.py` — 受付キューと2 msの待機窓を持つ、1モデル・1推論ワーカーのHTTPサーバー。`X-Inference-Batch-Size`で実バッチサイズを返します。
- `benchmarks/export_batch_results.py` — 10月7日の6ファイルを公開用に検査・出力。
- `benchmarks/audit_strands_versions.py` — 保存済みのStrands v19/v21応答について、版・要求ハッシュ・応答ハッシュ・判定を再検証。
- `benchmarks/compare_laya_precisions.py`、`benchmarks/audit_laya_precisions.py` — FP16/q8を`head_max_len=512`で直接測定し、全文入力・重みの識別・生応答・採点を監査。
- `benchmarks/export_laya_performance.py` — 両精度のバッチ・複数インスタンス・HTTP実バッチ計6ファイルを、モデル識別・判定・全応答ハッシュ・機密情報まで監査して公開用に変換。
- `benchmarks/bench_replicas.py`、`benchmarks/export_replica_results.py` — ローカルモデルを別プロセスで1・2・4個動かし、応答時間を測定・公開用に監査。

## 準備

Python 3.11以降を使います。測定コードと単体テストは標準ライブラリだけで動きます。元記事から設問を再抽出するときだけ `beautifulsoup4` が必要です。

ローカルモデルは各プロジェクトの手順で導入し、測定前に別ターミナルでサーバーを起動します。以下のポートは `compare_four.py` の既定値です。

```bash
# Strands Decider 2B（公式のmlx追加依存を導入済みの環境）
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 \
  --device mlx --host 127.0.0.1 --port 8012

# Clef-flash 4bit（mlx-community/clef-flash-4bit の配布スクリプト）
python <clef-checkout>/clef_mlx.py serve --port 8013

# laya multilingual q8（laya-mlx のローカルモデル）
laya-mlx-server --model <laya-model-directory> \
  --dtype int8_mixed --host 127.0.0.1 --port 8014
```

Jevには `TYPESAFE_API_KEY` 環境変数、または `benchmarks/.env.local` の `TYPESAFE_API_KEY=...` を使います。後者は `chmod 600 benchmarks/.env.local` が必要です。認証情報はコミットしません。Jevへの呼び出しは課金対象で、測定コードはJevに追加のウォームアップを送りません。

## 実行

```bash
python -m unittest discover -s benchmarks -p 'test_*.py'

# 20問の逐次測定（--parallel の後に同時数を指定しない）
python benchmarks/compare_four.py --model strands \
  --fixture benchmarks/kiro_article_cases.json \
  --out results/strands-v21-kiro.json --parallel

# 24件の逐次測定と同時数2・4・8
python benchmarks/compare_four.py --model strands \
  --fixture benchmarks/clef_vs_laya_cases.json \
  --out results/strands-v21-support.json --parallel 2 4 8

# v19の判定を再実行する場合は、サーバー側もv19で起動する
python benchmarks/compare_four.py --model strands --strands-version v19 \
  --fixture benchmarks/kiro_article_cases.json \
  --out results/strands-v19-kiro.json --parallel

# 公開済みのv19/v21結果をAPI呼び出しなしで再監査する
python benchmarks/audit_strands_versions.py
```

`--model`を`clef`、`laya`、`jev`に変えると同じ設問を送れます。Strandsの既定はv21で、v19を使うときはクライアントの`--strands-version v19`とサーバーのモデル指定を揃えます。既存結果へ並列測定を追記する際も版の一致を確認します。MPSを測る場合はStrandsサーバーを`--device mps`で起動し、別の結果ファイルを指定します。測定前のローカルモデル用ウォームアップを省く場合は`--no-warmup`を付けます。

## layaの保存済み記録

FP16とq8の判定差、重みの識別、公開済みの生結果は[判定記録](PRECISION_RESULTS.md)にあります。短文バッチ性能の再測定は実施しません。

## 複数インスタンスでの応答時間を測る

同じMacにモデルを1・2・4プロセス常駐させ、同時数1・2・4・8のHTTP要求を各3回測ります。プロセスごとの推論バッチは1件に固定します。測定プログラムが起動と終了を管理し、ループバック上の各プロセスへ要求を振り分けます。Jevには送信しません。

```bash
python benchmarks/bench_replicas.py --model-kind strands \
  --python path/to/strands-env/bin/python \
  --source path/to/strands-decider/src \
  --model StrandsAgents/strands-decider-2B-hobson-v21 \
  --fixture benchmarks/clef_vs_laya_cases.json \
  --out results/strands-replicas.json --port-base 8120
```

Strandsでは`--model-kind strands`とMLX依存を導入したPython、`--source`に`strands-decider/src`、`--model`にローカルキャッシュ済みの`StrandsAgents/strands-decider-2B-hobson-v21`を使います。必要なら`HF_HOME`をそのキャッシュに向けてください。Clefでは`--model-kind clef`とMLX用Python、`--source`に`clef_mlx.py`のあるディレクトリ、`--model`に4bitモデルのローカルディレクトリを使います。測定はオフラインで実行し、結果ファイルの上書きを拒否します。`--python`には仮想環境の`bin/python`を指定してください。

結果には各要求の応答本文、p50・p95、処理量、振り分け先インスタンス番号、実推論バッチサイズを残します。元結果から公開用の3ファイルを作るときは`python benchmarks/export_replica_results.py results results/2026-10-07`を実行します。元結果にはローカルの応答ヘッダーも残り、公開用では取り除かれます。

## バッチ試験のアーカイブ

`benchmarks/bench_model_batch.py`と`benchmarks/batch_server.py`は公開済みの生結果を再現するために残しています。短文24件の試験からモデル間のバッチ性能や実運用のバッチ数を評価しない方針です。[結果の読み方](BATCH_RESULTS.md)を確認してください。

通常の測定出力にはHTTPの生レスポンスが含まれ、モード600で保存されます。`results/`は原則Git管理対象外で、監査済みの公開用ファイルだけを例外として収録します。元の測定データに含まれていたPCの絶対パスを相対パスに置き換え、応答ヘッダーを除いています。HTTP応答本文はBase64で全件保持し、SHA-256で照合しています。記事の集計と照合するときは、同一の設問、モデル版、実行環境、サーバー設定を記録してください。

10月7日のバッチ用6ファイルと複数インスタンス用3ファイルも同じ方針で監査して収録しました。バッチ用は`python benchmarks/export_batch_results.py results results/2026-10-07`で再生成できます。公開用エクスポートでは保存済みの確率から最終判定を再集計します。公開前に認証情報とPC固有のパスを確認してください。

FP16/q8の追加性能測定6ファイルは`python benchmarks/export_laya_performance.py results results/2026-10-07`で監査しました。既存の公開ファイルは上書きしません。
