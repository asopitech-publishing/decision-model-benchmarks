# laya FP16・q8 比較 — 2026年10月7日

Kiro型の20問では、FP16とq8の判定は20問すべて同じでした。両者とも`simple`を20回選び、先行記事の許容ラベルには9/20問、第一候補には6/20問が一致しました。したがって、今回のKiro試験の低成績はq8への量子化では説明できません。サポート問い合わせ72判定ではq8が58/72、FP16が56/72でした。

| モデル | Kiro許容内 | Kiro第一候補一致 | サポート一致 | Kiro最長入力 |
| --- | ---: | ---: | ---: | ---: |
| laya-multilingual FP16 | 9/20 | 6/20 | 56/72 | 938トークン |
| laya-multilingual q8 | 9/20 | 6/20 | 58/72 | 938トークン |

## 量子化で判定が変わったのは3件

サポート問い合わせの72判定中、差が出たのは次の3件です。Kiroの20問には差がありません。

| 設問 | 判定項目 | 正解 | FP16 | q8 |
| --- | --- | --- | --- | --- |
| a03 | 緊急度 | 1 | 2 | 1 |
| a05 | 担当部署 | account | billing | account |
| s03 | 緊急度 | 0 | 1 | 2 |

q8はa03・a05で正解に変わり、s03では両方とも不正解でした。この72判定ではq8が2件上回ります。件数が少ないため、他の用途までq8の精度が高いとは評価しません。

## 両モデルに同じ全文を渡した

元のチェックポイント設定は`max_len=1024`、`head_max_len=256`です。Kiroの判定文と選択肢は合計294トークンで、特殊トークン込みでは297トークン必要です。256のままでは判定文の末尾が38トークン欠けます。今回は両モデルとも実行時の`head_max_len`を512にし、判定文・選択肢・ケース本文が切れていないことを全件検査しました。チェックポイントの設定ファイルと重みは変更していません。

モデルを1つずつMLX GPUに読み込み、各条件でローカルのウォームアップを1回実行した後、同じ順序のケースを1回ずつ判定しました。モデルへの直接呼び出しで、HTTPサーバーやJev APIは使っていません。FP16は配布元の重み、q8はそのローカル8bit変換版です。重みファイルのSHA-256はそれぞれ`9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`と`4efc8723f42950d07fd79d32d032e5ef06bd3e4b3a7fbcc58c4b5a125ed85b86`です。

モデルファイルはFP16が614.0 MiB、q8が515.4 MiBでした。今回の逐次推論p50はKiroでFP16 7.392 ms、q8 7.787 ms、サポートでFP16 6.944 ms、q8 8.514 msです。各条件1周なので、速度の優劣を決める測定としては扱いません。既存のHTTP応答時間とも測定経路が違います。

## 再現と生結果

`laya-mlx`を読み込める環境で、次の形で各モデルと各設問集を実行します。`--precision`と`--model`には対応する組み合わせを指定します。`--head-max-len`の既定値は512で、設定ファイルは変更しません。

```bash
PYTHONPATH=<laya-mlx-source> <laya-mlx-python> benchmarks/compare_laya_precisions.py \
  --precision fp16 --model <laya-multilingual-directory> \
  --fixture benchmarks/kiro_article_cases.json \
  --out <private-result-file>

python benchmarks/audit_laya_precisions.py
```

公開用の生結果には各ケースの判定・確率・入力トークン内訳・応答本文・Base64・SHA-256・計測時間を保存しました。監査コードが生応答のハッシュ、設問ハッシュ、採点、機密情報の混入を再検査します。

- [FP16・Kiro](results/2026-10-07/laya-fp16-kiro-head512.json) / [q8・Kiro](results/2026-10-07/laya-q8-kiro-head512.json)
- [FP16・サポート](results/2026-10-07/laya-fp16-support-head512.json) / [q8・サポート](results/2026-10-07/laya-q8-support-head512.json)
