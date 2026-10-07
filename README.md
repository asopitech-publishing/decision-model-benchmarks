# Decision Model Benchmarks

Strands Decider 2B、Clef-flash、laya、Jevを同じ設問形式で比較するベンチマークコードです。日本語20問のルーティング評価と、24件×3設問の問い合わせ分類を収録しています。記事本文は別の非公開リポジトリ [`substack-manuscripts`](https://github.com/asopitech-publishing/substack-manuscripts) で管理します。

## 内容

- `benchmarks/compare_four.py` — 4モデル共通のリクエスト、品質・応答時間・並列処理量の測定。HTTPの生レスポンス、usage、SHA-256を結果JSONへ保存します。
- `benchmarks/test_compare_four.py` — API課金なしで実行できる単体テスト。
- `benchmarks/kiro_article_cases.json` — [クラスメソッドの検証記事](https://dev.classmethod.jp/articles/strands-decider-2b-m4-mac-kiro-crew/)の日本語20問と、共通の`B_desc`判定文。`prepare_kiro_article.py`で元記事から再生成できます。
- `benchmarks/clef_vs_laya_cases.json` — 今回作成した日本語サポート問い合わせ24件、担当部署・緊急度・返金要求の3設問。
- [`RESULTS.md`](RESULTS.md) — 2026年10月6日の品質・並列性能の集計と評価上の注意点。
- [`results/2026-10-06/`](results/2026-10-06/) — 同日の公開用生結果12ファイル。HTTP応答本文、usage、各リクエストの計測値を収録。
- `benchmarks/export_public_results.py` — 元の測定結果から公開用ファイルを作り、応答本文のハッシュと機密情報を検査。

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

通常の測定出力にはHTTPの生レスポンスが含まれ、モード600で保存されます。`results/`は原則Git管理対象外です。2026年10月6日の公開用に監査した12ファイルのみ、例外として収録しました。元の測定データに含まれていたPCの絶対パスを相対パスに置き換え、応答ヘッダーを除いています。HTTP応答本文はBase64で全件保持し、SHA-256で照合しています。記事の集計と照合するときは、同一の設問、モデル版、実行環境、サーバー設定を記録してください。
