# Decision Model Benchmarks

Strands Decider 2B、Clef-flash、laya、Jevを同じ設問形式で比較するベンチマークコードです。日本語20問のルーティング評価と、24件×3設問の問い合わせ分類を収録しています。記事本文は別の非公開リポジトリ [`substack-manuscripts`](https://github.com/asopitech-publishing/substack-manuscripts) で管理します。

## 内容

- `benchmarks/compare_four.py` — 4モデル共通のリクエスト、品質・応答時間・並列処理量の測定。HTTPの生レスポンス、usage、SHA-256を結果JSONへ保存します。
- `benchmarks/test_compare_four.py` — API課金なしで実行できる単体テスト。
- `benchmarks/kiro_article_cases.json` — [クラスメソッドの検証記事](https://dev.classmethod.jp/articles/strands-decider-2b-m4-mac-kiro-crew/)の日本語20問と、共通の`B_desc`判定文。`prepare_kiro_article.py`で元記事から再生成できます。
- `benchmarks/clef_vs_laya_cases.json` — 今回作成した日本語サポート問い合わせ24件、担当部署・緊急度・返金要求の3設問。
- [`RESULTS.md`](RESULTS.md) — 2026年10月6日の品質・並列性能の集計と評価上の注意点。
- [`BATCH_RESULTS.md`](BATCH_RESULTS.md) — 10月7日に各ローカルモデルを1インスタンスで動かしたバッチ推論・HTTP同時要求の測定。
- [`REPLICA_RESULTS.md`](REPLICA_RESULTS.md) — 同じMacで各モデルを1・2・4プロセス常駐させたときのHTTP応答時間。
- [`results/2026-10-06/`](results/2026-10-06/) — 同日の公開用生結果12ファイル。HTTP応答本文、usage、各リクエストの計測値を収録。
- [`results/2026-10-07/`](results/2026-10-07/) — バッチ推論と複数インスタンス試験の公開用生結果9ファイル。
- `benchmarks/export_public_results.py` — 元の測定結果から公開用ファイルを作り、応答本文のハッシュと機密情報を検査。
- `benchmarks/bench_model_batch.py`、`benchmarks/*_batch.py` — 独立した要求を、各モデル1インスタンスで実際に1回の推論へまとめる測定。
- `benchmarks/batch_server.py` — 受付キューと2 msの待機窓を持つ、1モデル・1推論ワーカーのHTTPサーバー。`X-Inference-Batch-Size`で実バッチサイズを返します。
- `benchmarks/export_batch_results.py` — 10月7日の6ファイルを公開用に検査・出力。
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
```

`--model`を`clef`、`laya`、`jev`に変えると同じ設問を送れます。MPSを測る場合はStrandsサーバーを`--device mps`で起動し、別の結果ファイルを指定します。測定前のローカルモデル用ウォームアップを省く場合は`--no-warmup`を付けます。

## 複数インスタンスでの応答時間を測る

同じMacにモデルを1・2・4プロセス常駐させ、同時数1・2・4・8のHTTP要求を各3回測ります。プロセスごとの推論バッチは1件に固定します。測定プログラムが起動と終了を管理し、ループバック上の各プロセスへ要求を振り分けます。Jevには送信しません。

```bash
python benchmarks/bench_replicas.py --model-kind laya \
  --python path/to/laya-env/bin/python \
  --source path/to/laya-mlx \
  --model path/to/laya-multilingual-q8 \
  --fixture benchmarks/clef_vs_laya_cases.json \
  --out results/laya-replicas.json --port-base 8120
```

Strandsでは`--model-kind strands`とMLX依存を導入したPython、`--source`に`strands-decider/src`、`--model`にローカルキャッシュ済みの`StrandsAgents/strands-decider-2B-hobson-v21`を使います。必要なら`HF_HOME`をそのキャッシュに向けてください。Clefでは`--model-kind clef`とMLX用Python、`--source`に`clef_mlx.py`のあるディレクトリ、`--model`に4bitモデルのローカルディレクトリを使います。測定はオフラインで実行し、結果ファイルの上書きを拒否します。`--python`には仮想環境の`bin/python`を指定してください。

結果には各要求の応答本文、p50・p95、処理量、振り分け先インスタンス番号、実推論バッチサイズを残します。元結果から公開用の3ファイルを作るときは`python benchmarks/export_replica_results.py results results/2026-10-07`を実行します。元結果にはローカルの応答ヘッダーも残り、公開用では取り除かれます。

## 単一モデル内のバッチ推論を測る

10月7日の試験では、サーバーの同時受付とモデルの推論バッチを分けて測りました。各モデルに対応するPython環境で以下を実行します。`path/to/...`は自分の環境のソースとモデルの位置へ置き換えてください。Jev APIは使いません。

```bash
# laya-mlxのパッケージが読み込める環境
PYTHONPATH="benchmarks:path/to/laya-mlx" python benchmarks/bench_model_batch.py \
  --model-kind laya --model path/to/laya-multilingual-q8 \
  --fixture benchmarks/clef_vs_laya_cases.json --out results/laya-batch.json

# StrandsのMLX追加依存を導入済みの環境
PYTHONPATH=benchmarks python benchmarks/bench_model_batch.py \
  --model-kind strands --model StrandsAgents/strands-decider-2B-hobson-v21 \
  --fixture benchmarks/clef_vs_laya_cases.json --out results/strands-batch.json

# ClefのMLX移植版が読み込める環境
PYTHONPATH="benchmarks:path/to/clef-mlx" python benchmarks/bench_model_batch.py \
  --model-kind clef --model mlx-community/clef-flash-4bit \
  --fixture benchmarks/clef_vs_laya_cases.json --out results/clef-batch.json
```

既定でバッチ件数1・2・4・8を各5回測ります。各バッチ形状をローカルで1周してから計測し、プロンプトキャッシュは無効にします。出力には逐次基準の回答、各試行の全回答、判定の一致、確率差、推論時間、Metalメモリを保存します。

HTTP経由の同時要求は、元のサーバーの代わりに以下のバッチサーバーを別ターミナルで起動して測ります。例はlaya用です。StrandsとClefも`--model-kind`、モデルID、ポートを変えます。各プロセスが保持するモデルは1個です。

```bash
PYTHONPATH="benchmarks:path/to/laya-mlx" python benchmarks/batch_server.py \
  --model-kind laya --model path/to/laya-multilingual-q8 \
  --port 8014 --batch-requests 8 --batch-wait-ms 2

python benchmarks/compare_four.py --model laya \
  --fixture benchmarks/clef_vs_laya_cases.json \
  --out results/laya-http-batch.json --parallel 2 4 8
```

サーバーはループバックのみに公開し、既定では1要求当たり最大3問を受け付けます。設問数が違う場合は`--max-questions-per-request`を指定してください。各HTTP応答の`inference_batch_size`を確認すれば、同時要求が実際に何件の推論バッチになったか分かります。

通常の測定出力にはHTTPの生レスポンスが含まれ、モード600で保存されます。`results/`は原則Git管理対象外です。2026年10月6日の公開用に監査した12ファイルのみ、例外として収録しました。元の測定データに含まれていたPCの絶対パスを相対パスに置き換え、応答ヘッダーを除いています。HTTP応答本文はBase64で全件保持し、SHA-256で照合しています。記事の集計と照合するときは、同一の設問、モデル版、実行環境、サーバー設定を記録してください。

10月7日のバッチ用6ファイルと複数インスタンス用3ファイルも同じ方針で監査して収録しました。バッチ用は`python benchmarks/export_batch_results.py results results/2026-10-07`で再生成できます。公開前に認証情報とPC固有のパスを確認してください。
