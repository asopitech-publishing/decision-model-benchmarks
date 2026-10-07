# laya FP16・q8 の判定記録（2026年10月7日）

配布元のlaya multilingual FP16重みと、そのローカル8bit変換版を同じ入力で測りました。Kiro Crew型の20問では両版とも全問を`simple`と判定し、許容ラベルに入ったのは9/20問、第一候補との一致は6/20問です。サポート問い合わせ24件×3問の72判定ではFP16が56/72、q8が58/72でした。Kiro型ルーターの候補からは両版を外します。

| モデル | Kiro許容内 | Kiro第一候補一致 | サポート一致 |
| --- | ---: | ---: | ---: |
| laya-multilingual FP16 | 9/20 | 6/20 | 56/72 |
| laya-multilingual q8 | 9/20 | 6/20 | 58/72 |

## 判定差は72件中3件

| 設問 | 判定項目 | 正解 | FP16 | q8 |
| --- | --- | --- | --- | --- |
| a03 | 緊急度 | 1 | 2 | 1 |
| a05 | 担当部署 | account | billing | account |
| s03 | 緊急度 | 0 | 1 | 2 |

q8はa03・a05で正解に変わり、s03では両版とも外しました。この72判定ではq8が2件多く正解しました。

## 入力と重みを揃えた

元のチェックポイント設定は`max_len=1024`、`head_max_len=256`です。Kiroの判定文と選択肢には特殊トークン込みで297トークン必要で、既定値では末尾38トークンが欠けます。両版とも実行時の`head_max_len`を512に設定し、判定文・選択肢・ケース本文が全件入ったことを検査しました。最長の入力は938トークンです。チェックポイントの設定ファイルと重みは変更していません。

FP16は配布元の重み、q8はそのローカル8bit変換版です。重みファイルのSHA-256はそれぞれ`9d628fd971b700382ac6f65920a86f149777b2e748e0c955fb3b19695aa8f204`と`4efc8723f42950d07fd79d32d032e5ef06bd3e4b3a7fbcc58c4b5a125ed85b86`です。モデルファイルはFP16が614.0 MiB、q8が515.4 MiBでした。

## 性能測定の扱い

両版について、24件の短いサポート問い合わせによるモデル内バッチ、HTTP実バッチ、1・2・4インスタンスの生結果も保存しています。この入力では1設問あたり最大108トークンで、layaの1024トークン上限から遠く離れていました。長い入力や長短混合を含む運用のバッチ性能を測ったことにはならないため、これらの速度順位・倍率・配置推奨を比較結果から外しました。詳細は[短文バッチ試験の記録](BATCH_RESULTS.md)を参照してください。

## 再現と生結果

`laya-mlx`を読み込める環境で、次の形で品質試験を実行します。モデルディレクトリの設定ファイルは変更しません。

```bash
PYTHONPATH=<laya-mlx-source> <laya-mlx-python> benchmarks/compare_laya_precisions.py \
  --precision fp16 --model <laya-multilingual-directory> \
  --fixture benchmarks/kiro_article_cases.json \
  --out <private-result-file>

python benchmarks/audit_laya_precisions.py
```

公開用の生結果には各ケースの判定・確率・入力トークン内訳・応答本文・Base64・SHA-256・計測時間を残しました。監査コードが生応答のハッシュ、設問ハッシュ、採点、機密情報の混入を再検査します。

- [FP16・Kiro](results/2026-10-07/laya-fp16-kiro-head512.json) / [q8・Kiro](results/2026-10-07/laya-q8-kiro-head512.json)
- [FP16・サポート](results/2026-10-07/laya-fp16-support-head512.json) / [q8・サポート](results/2026-10-07/laya-q8-support-head512.json)
- [FP16・モデル内バッチ](results/2026-10-07/laya-fp16-batch-head512-20261007.json) / [q8・モデル内バッチ](results/2026-10-07/laya-q8-batch-head512-20261007.json)
- [FP16・複数インスタンス](results/2026-10-07/laya-fp16-replicas-head512-20261007.json) / [q8・複数インスタンス](results/2026-10-07/laya-q8-replicas-head512-20261007.json)
- [FP16・HTTP実バッチ](results/2026-10-07/laya-fp16-http-batch-head512-20261007.json) / [q8・HTTP実バッチ](results/2026-10-07/laya-q8-http-batch-head512-20261007.json)
