# Face_Predict_Keras

Kerasで顔識別。Local PC上で動作するサンプル。
(2022/9/26)

Qiita記事：[Colaboratoryで学習したモデル+CPU環境でハーフタレントの顔識別したときのメモ](https://qiita.com/kiwsdiv/items/2f72b20c345cbfeee34e)

# Requirement

動作確認環境
- Windows10 Home 21H1
- Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz 1.80 GHz
- Python 3.8.6
- tensorflow 2.8.2
- keras 2.8.0
- opencv-python 4.6.0.66
- numpy 1.23.3
- etc...

その他、必要なライブラリがあればインストールしてください。

# Usage

コンソール上で `predict.py` 以下を実行。
```bash
$ python predict.py
~~~ Warning ~~~
```

テストする対象の画像がdataフォルダからランダムに選択され、ウィンドウが表示される。

![images](images/face_wnd.png)

何かキーを押すとウィンドウが閉じて推定に進む。

```
Test filename : data\Yuji\3_face_145.png
Yuji:98.70%
['Other', 'Joy', 'Harry', 'Uentsu', 'Raul', 'Yuji']
[ 0.1598472   0.42168     0.00062809  0.55710834  0.15593176 98.70481   ]
Erapsed time  : 0.2063 [s]
```
推定結果と処理時間が表示される。

# Note
- バージョン依存によって動作しない可能性があります。

# Author
- iwatadive28