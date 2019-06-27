# Iris
アヤメの分類実験を機械学習により行う

---

## 必要なモジュール
- pytorch 0.4以上
- pyyaml
- sklearn
- visdom

## 実行方法
1. 最上階層にcorpusフォルダとsaveフォルダを作成
2. make_data.pyを実行
3. python -m visdom.serverでvisdomサーバをたてる
4. train.pyを実行しvisdomによる損失グラフを観察しながらCtrl+Cで終了
5. test.pyで分類テスト
