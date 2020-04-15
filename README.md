# detectron2 for まちカドまぞく
---

上手くいくとこんな感じでインスタンスセグメンテーションされます。
<img src=https://user-images.githubusercontent.com/33882378/79189949-02f72b80-7e5e-11ea-81e4-cdc58a3d33c9.jpg>

「インスタンスセグメンテーションがしたい！！」けど難しそうだ・・・

というわけで、学習モチベーション維持のために、まちカドまぞくが好きすぎるので detectron2 用のデータセットを　VoTT で作って、detectron2 を試してみました。

> 製作に当たっては https://demura.net/deeplearning/16807.html を参考にさせていただいています。（感謝です！！）
>
> データセットは 100% ネタですが、detectron2 を自作のデータセットで試したい方の参考になれば何よりです。

## 1.使い方

### (1) 当然ですが、detectron2 は動く状態にしてください。
本家 https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md の通りで動くようになるはずですが、私も初め動きませんでしたのでポイントを・・・

> Requirements
>
>    * Linux or macOS with Python ≥ 3.6
>    * PyTorch ≥ 1.3
>    * torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this.
>    * OpenCV, optional, needed by demo and visualization pycocotools: pip install cython; pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

以上の要件が確実に満たされているかを確認してください。

で、

> Build Detectron2 from Source
> 
> ```
> # Or, to install it from a local clone:
> git clone https://github.com/facebookresearch/detectron2.git
> cd detectron2 && python -m pip install -e .
> ```

の手順でセットアップできました。

### (2) detectron2_Machikado は detectron2 ディレクトリの下に clone してください。
detectron2 を git clone して出来た detectron2 ディレクトリの直下で clone してください。別に変えてもいいけど、その場合ディレクトリ関係のパスは修正してください。

### (3) weight ファイルをダウンロードします。

https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md から、以下のファイルをダウンロードしてください。

画面の中段くらいの、「COCO Object Detection Baselines」ってとこの「COCO Instance Segmentation Baselines with Mask R-CNN」の「X101-FPN」の欄の「model」ってとこです。

ちなみに、画像で言うとこれです。
 
<img src=https://user-images.githubusercontent.com/33882378/79058377-3a23dc00-7ca8-11ea-9622-a8e4c8ea53f8.jpg>

### (4) ダウンロードしたファイルを coco_models ディレクトリにコピーしてください。

ですが、ファイル名が確実に違うので・・・例えば、ダウンロードされたファイル名が model_final_xxxxxx.pkl だとしたら、ノート中盤の・・・

```python
cfg.MODEL.WEIGHTS = './coco_models/model_final_2d9806.pkl'
```

の部分を・・・

```python
cfg.MODEL.WEIGHTS = './coco_models/model_final_xxxxxx.pkl'
```

へ書き換えて貰えばOKです。

### (5) machikado データセットをダウンロード

* release から v1.x.x_vott-json-export.zip (v1.x.xは色々変わるので新しいのを選んでください)
* zip を展開後 detectron2_Machikado/vott-json-export ディレクトリにコピーしてください。

> データセットは Git リリースでアップしています。

## 2. 学習

[Machikado_training.ipynb](https://github.com/nTAKAn/detectron2_Machikado/blob/master/Machikado_training.ipynb) を実行すればOK！

> VoTT でアノテーションした違うデータセットを使いたい場合は、VoTT でエクスポートした vott-json-export をそのまま上書きしてしまえば良いです。
> ただし、Machikado_evaluate.ipynb の変更が必要です(推論時は日本語名を使ってますので・・・)。

## 3. 推論

[Machikado_predict.ipynb](https://github.com/nTAKAn/detectron2_Machikado/blob/master/Machikado_predict.ipynb)

> 違うデータセットの場合は、以下の部分の「VoTT のカテゴリはこっち」を使ってください。
> 
> ```python
> # 日本語名はこっち
> MetadataCatalog.get('train').set(thing_classes=CAT_NAME_JP)
> MetadataCatalog.get('test').set(thing_classes=CAT_NAME_JP)
> 
> # VoTT のカテゴリ名はこっち
> #MetadataCatalog.get('train').set(thing_classes=list(CAT_NAME2ID.keys()))
> #MetadataCatalog.get('test').set(thing_classes=list(CAT_NAME2ID.keys()))
> ```

## 4. 評価

AP, mAP を計算してみます。

[Machikado_evalute.ipynb](https://github.com/nTAKAn/detectron2_Machikado/blob/master/Machikado_evalute.ipynb)