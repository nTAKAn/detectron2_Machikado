# detectron2 for まちカドまぞく  

## ＜現在テスト継続中です＞

データセットのアノテーションが中途半端なんで仮の状態です！
ですが、30枚程度の訓練画像にしてはテスト結果が良いですね。
後50枚程度アノテーションが残っていますので、もう少し結果は良くなりそうです・・・生暖かく見守ってください。

---

<img src=https://user-images.githubusercontent.com/33882378/79041969-d4473e00-7c2e-11ea-9072-b24d55bb4762.jpg>

AI学習モチベーション維持のために、まちカドまぞくが好きすぎるので detectron2 用のデータセットを作って試してみました。

> 製作に当たっては
> https://demura.net/deeplearning/16807.html
> を参考にさせていただいています。（感謝です！！）

## 1.使い方

### (1) 当然ですが、detectron2 は動く状態にしてください。

### (2) detectron2 ディレクトリの下に clone してください。
別に変えてもいいけど、その場合ディレクトリ関係のパスは修正してください。

### (3) weight ファイルをダウンロードします。

https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md から、以下のファイルをダウンロードしてください。

画面の中段くらいの、「COCO Object Detection Baselines」ってとこの「X101-FPN」の欄の「model」ってとこです。

ちなみに、画像で言うとこれです。
 
<img src=https://user-images.githubusercontent.com/33882378/79042089-c2b26600-7c2f-11ea-9630-69cef399b497.jpg>

### (4) ダウンロードしたファイルを coco_models の下にコピーしてください。

ですが、ファイル名が確実に違うので・・・例えば、ダウンロードされたファイル名が model_final_xxxxxx.pkl だとしたら、ノート中盤の・・・

```
cfg.MODEL.WEIGHTS = './coco_models/model_final_2d9806.pkl'
```
の部分を・・・
```
cfg.MODEL.WEIGHTS = './coco_models/model_final_xxxxxx.pkl'
```

へ書き換えて貰えばOKです。