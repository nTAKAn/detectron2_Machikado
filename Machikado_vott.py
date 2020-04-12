import os
import json
import random
from collections import OrderedDict
from PIL import Image

from detectron2.structures import BoxMode

# #############################################################################
# エクスポートファイルからカテゴリ名を調べる
def get_cat_names(export_filename):
    with open(export_filename, 'r') as f:
        json_data = json.load(f)
        
    cat_name2id = OrderedDict()
    cat_id2name = OrderedDict()

    for i, node in enumerate(json_data['tags']):
        cat_name2id[node['name']] = i
        cat_id2name[i] = node['name']
    
    return cat_name2id, cat_id2name


# #############################################################################
# machikado用にアレンジした読み込み関数
def get_machikado_dicts(export_filename, image_dirname, cat_name2id):
    with open(export_filename, 'r') as f:
        json_data = json.load(f, object_pairs_hook=OrderedDict) # データ順を固定しておく
    
    assets = json_data['assets']

    dataset_dicts = []
    for item in assets.values():
        asset = item['asset']
        regions = item['regions']

        if len(regions) == 0:
            print('警告: name: {} - 領域データが空だったのでスキップ'.format(asset['name']))
            continue
            
        # 画像サイズを取得し確認する
        # （VoTT でアノテーション中画像を差し替えると画像のサイズが古い画像のままになるので修正する）
        im = Image.open(image_dirname + asset['name'])
        w, h = im.size
        
        if asset['size']['height'] != h or asset['size']['width'] != w:
            print('警告: name: {} - 画像サイズの不整合 image_size:({}, {}), {}: ({}, {})'.format(
                asset['name'], asset['size']['width'], asset['size']['height'], export_filename, w, h))
        
        record = {}
        record['file_name'] = image_dirname + asset['name']
        record['height'] = h
        record['width'] = w

        objs = []
        for region in regions:
            points = region['points']
            assert len(points), '座標データが無い！'

            if len(region['tags']) > 1:
                print('警告: name: {} - 複数のタグを確認！ tags: {}'.format(asset['name'], region['tags']))

            poly = []
            for pt in points:
                poly += [pt['x'], pt['y']]

            bbox = region['boundingBox']

            obj = {
                'bbox': [bbox['left'], bbox['top'], bbox['left'] + bbox['width'], bbox['top'] + bbox['height']],
                'bbox_mode': BoxMode.XYWH_ABS, # XYWH_REL はまだサポートされていないらしい
                'segmentation': [poly],
                'category_id': cat_name2id[region['tags'][0]],
                'iscrowd': 0
            }
            objs.append(obj)

        record['annotations'] = objs
        dataset_dicts.append(record)
        
    return dataset_dicts