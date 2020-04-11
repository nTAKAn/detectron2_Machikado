# detectron2 for まちカドまぞく

<img src=https://user-images.githubusercontent.com/33882378/79041969-d4473e00-7c2e-11ea-9072-b24d55bb4762.jpg>

AI学習モチベーション維持のために、まちカドまぞくが好きすぎるので detectron2 用のデータセット
なんかを作って試してみました。

## 1.使い方
* ます当然ですが、detectron2 は動く状態にしてください。
* detectron2 ディレクトリの下に clone してください。（別に変えてもいいけど、その場合ディレクトリ関係のパスは修正してください）
* https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md から、以下のファイルをダウンロードしてください。

COCO Object Detection Baselines ってとこの X101-FPN の model

画像で言うとこれです。

<img src=https://user-images.githubusercontent.com/33882378/79042089-c2b26600-7c2f-11ea-9630-69cef399b497.jpg>

* で coco_models の下にコピーしてください。

ですが、ファイル名が確実に違うので・・・例えば、ダウンロードされたファイル名が model_final_2d9806.pkl だとしたら、</br>
ノート中盤の

```
cfg.MODEL.WEIGHTS = './coco_models/model_final_2d9806.pkl'
```
の部分を
```
'./coco_models/model_final_2d9806.pkl'
```

へ書き換えて貰えばOKです。