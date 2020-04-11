# detectron2 for まちカドまぞく

AI学習モチベーション維持のために、まちカドまぞくが好きすぎるので detectron2 用のデータセット
なんかを作って試してみました。

## 1.使い方
* ます当然ですが、detectron2 は動く状態にしてください。
* detectron2 ディレクトリの下に clone してください。（別に変えてもいいけど、その場合ディレクトリ関係のパスは修正してください）
* https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md から、以下のファイルをダウンロードしてください。

COCO Object Detection Baselines ってとこの X101-FPN の model

直リンは以下

https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_2d9806.pkl

* で coco_models の下に model_final_2d9806.pkl をコピーしてください。

ファイル名が確実に違うので・・・ノート中盤の

```
cfg.MODEL.WEIGHTS = './coco_models/model_final_2d9806.pkl'
```
の部分を
```
'./coco_models/model_final_2d9806.pkl'
```
へ書き換えて貰えばOKです。