import numpy as np
import pandas as pd
import cv2

from detectron2.data import DatasetCatalog, MetadataCatalog


def append_masks(dataset_dicts):
    """
    annotations の segmentation から、マスク画像を生成して、新たに masks として追加する
    """
    for asset in dataset_dicts:
        
        asset['masks'] = []

        for anno in asset['annotations']:
            assert len(anno['segmentation'])

            p = anno['segmentation'][0]
            pts = np.vstack([p[::2], p[1::2]]).T
            pts = pts.astype(np.int)

            ih, iw = asset['height'], asset['width']
            t_im = np.zeros((ih, iw), dtype=np.uint8)
            mask = cv2.fillPoly(t_im, [pts], 1).astype(np.bool)

            asset['masks'].append(mask)
            

def get_true_datas(catalog_name):
    """
    データセットから評価用にマスクデータとクラスを取り出す
    """
    true_dicts = []
    
    for i, asset in enumerate(DatasetCatalog.get(catalog_name)):
        true_classes = []
        
        for anno in asset['annotations']:
            true_classes.append(anno['category_id'])

        true_classes, true_masks = np.asarray(true_classes), np.asarray(asset['masks'])
        assert (len(true_classes) == len(true_masks)), '全ての要素数は等しいはず'

        true_dict = {'classes': true_classes, 'masks': true_masks, 'file_name': asset['file_name']}
        true_dicts.append(true_dict)
        
    return true_dicts


def predict_datas(predictor, catalog_name, verbose=False):
    """
    データセットを一括で評価する
    """
    pred_dicts = []
    
    for i, asset in enumerate(DatasetCatalog.get(catalog_name)):
        w, h = asset['width'], asset['height']
        img = cv2.imread(asset['file_name'])

        if verbose:
            print('predict ({:4d}/{:4d}): {}'.format(i + 1, len(DatasetCatalog.get(catalog_name)), asset["file_name"]))
        output = predictor(img)

        pred = output['instances'].get_fields()

        pred_masks = pred['pred_masks'].cpu().numpy()
        pred_classes = pred['pred_classes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        
        assert (len(pred_classes) == len(pred_masks) == len(scores)), '全ての要素数は等しいはず'
        assert np.allclose(np.arange(len(scores)), scores.argsort()[::-1]), \
            'スコアがソートされていない scores: {}'.format(scores) # ソートされている様なのですがチェック

        pred_dict = {'classes': pred_classes, 'masks': pred_masks, 'scores': scores, 'file_name': asset['file_name']}
        pred_dicts.append(pred_dict)
    
    return pred_dicts


def make_info_dict(true_dicts, pred_dicts, classes, th):
    """
    AP の計算に必要なデータを生成する
    """
    assert len(true_dicts) == len(pred_dicts), '要素数は等しいはず'

    columns = ['file_i', 'pred_i', 'score','correct', 'pre', 'rec', 'iou']
    tmp_dicts = {_cls: {col: [] for col in columns} for _cls in classes}
    
    num_true_cls_count = {_cls: 0 for _cls in classes}  # TP の数をカウント

    for file_i, (true_dict, pred_dict) in enumerate(zip(true_dicts, pred_dicts)):
        iou_list = [[] for _ in true_dict['classes']]
        p_scores = []

        # 予想された物に対して、それぞれ教師データとの IoU を計算する
        for pred_i, (p_cls, p_mask, p_score) in enumerate(zip(pred_dict['classes'], pred_dict['masks'], pred_dict['scores'])):
            p_scores.append(p_score)

            for i, (t_cls, t_mask) in enumerate(zip(true_dict['classes'], true_dict['masks'])):
                iou = (t_mask & p_mask).sum() / (p_mask | t_mask).sum()
                iou_list[i].append(iou)

        iou_list = np.asarray(iou_list).T

        # 教師データに含まれる各クラスについてそれぞれ correct? を計算する
        u_classes = np.unique(true_dict['classes'])

        for u_cls in u_classes:
            # 該当クラス u_cls の iou を取得する
            t_indices = np.where(true_dict['classes'] == u_cls)[0]
            p_indices = np.where(pred_dict['classes'] == u_cls)[0]
            u_iou_list = iou_list[:, t_indices][p_indices, :]
            
            # IoU のしきい値で正解ラベルを処理
            correct_list = u_iou_list > th
            
            # ダブルカウントされた場合例外
            if (correct_list.sum(axis=0) > 1).sum() > 1:
                i = np.where(correct_list.sum(axis=0) > 1)[0]
                print(correct_list)
                assert False, 'ダブルカウント'

            num_true_cls_count[u_cls] += len(t_indices)  # true ラベルをカウント
            
            # 値を格納
            tmp_dict = tmp_dicts[u_cls]
            tmp_dict['file_i'] += [file_i] * len(p_indices)
            tmp_dict['pred_i'] += list(range(len(p_indices)))
            tmp_dict['score'] += [p_scores[i] for i in p_indices]
            tmp_dict['correct'] += list(correct_list.sum(axis=1) > 0)
            tmp_dict['pre'] += [None] * len(p_indices)
            tmp_dict['rec'] += [None] * len(p_indices)
            tmp_dict['iou'] += list(u_iou_list.max(axis=1))
    
    # pre, rec を計算する
    df_dict = {_cls: pd.DataFrame(tmp_dicts[_cls], columns=columns) for _cls in tmp_dicts.keys()}
    
    for _cls in df_dict.keys():
        df = df_dict[_cls]
        df.sort_values('score', ascending=False, inplace=True)
        
        TP_count = 0
        pres = []
        recs = []

        for i in range(len(df)):
            se = df.iloc[i, :]

            if se['correct']:
                TP_count += 1

            pres.append(TP_count / (i + 1))
            recs.append(TP_count / num_true_cls_count[_cls])

        df['pre'] = pres
        df['rec'] = recs

    return df_dict


import pandas as pd

def make_info_dict(true_dicts, pred_dicts, classes, th):
    """
    AP の計算に必要なデータを生成する
    """
    assert len(true_dicts) == len(pred_dicts), '要素数は等しいはず'

    columns = ['file_i', 'pred_i', 'score','correct', 'pre', 'rec', 'iou']
    tmp_dicts = {_cls: {col: [] for col in columns} for _cls in classes}
    
    num_true_cls_count = {_cls: 0 for _cls in classes}  # TP の数をカウント

    for file_i, (true_dict, pred_dict) in enumerate(zip(true_dicts, pred_dicts)):
        iou_list = [[] for _ in true_dict['classes']]
        p_scores = []

        # 予想された物に対して、それぞれ教師データとの IoU を計算する
        for pred_i, (p_cls, p_mask, p_score) in enumerate(zip(pred_dict['classes'], pred_dict['masks'], pred_dict['scores'])):
            p_scores.append(p_score)

            for i, (t_cls, t_mask) in enumerate(zip(true_dict['classes'], true_dict['masks'])):
                iou = (t_mask & p_mask).sum() / (p_mask | t_mask).sum()
                iou_list[i].append(iou)

        iou_list = np.asarray(iou_list).T

        # 教師データに含まれる各クラスについてそれぞれ correct? を計算する
        u_classes = np.unique(true_dict['classes'])

        for u_cls in u_classes:
            # 該当クラス u_cls の iou を取得する
            t_indices = np.where(true_dict['classes'] == u_cls)[0]
            p_indices = np.where(pred_dict['classes'] == u_cls)[0]
            u_iou_list = iou_list[:, t_indices][p_indices, :]
            
            # IoU のしきい値で正解ラベルを処理
            correct_list = u_iou_list > th
            
            # ダブルカウントされた場合例外
            if (correct_list.sum(axis=0) > 1).sum() > 1:
                i = np.where(correct_list.sum(axis=0) > 1)[0]
                print(correct_list)
                assert False, 'ダブルカウント'

            num_true_cls_count[u_cls] += len(t_indices)  # true ラベルをカウント
            
            # 値を格納
            tmp_dict = tmp_dicts[u_cls]
            tmp_dict['file_i'] += [file_i] * len(p_indices)
            tmp_dict['pred_i'] += list(range(len(p_indices)))
            tmp_dict['score'] += [p_scores[i] for i in p_indices]
            tmp_dict['correct'] += list(correct_list.sum(axis=1) > 0)
            tmp_dict['pre'] += [None] * len(p_indices)
            tmp_dict['rec'] += [None] * len(p_indices)
            tmp_dict['iou'] += list(u_iou_list.max(axis=1))
    
    # pre, rec を計算する
    df_dict = {_cls: pd.DataFrame(tmp_dicts[_cls], columns=columns) for _cls in tmp_dicts.keys()}
    
    for _cls in df_dict.keys():
        df = df_dict[_cls]
        df.sort_values('score', ascending=False, inplace=True)
        
        TP_count = 0
        pres = []
        recs = []

        for i in range(len(df)):
            se = df.iloc[i, :]

            if se['correct']:
                TP_count += 1

            pres.append(TP_count / (i + 1))
            recs.append(TP_count / num_true_cls_count[_cls])

        df['pre'] = pres
        df['rec'] = recs

    return df_dict


def calc_AP(df_dict, cat_names):
    """
    AP を計算する
    """
    AP = []

    for _cls in df_dict.keys():
        df = df_dict[_cls]
        rec, pre = df['rec'].values, df['pre'].values

        # 1つしかない場合は 0
        if len(rec) == 1:
            AP.append(0)
            continue

        # 台形として面積を求めます
        S = 0
        for i in range(len(rec) - 1):
            h = rec[i + 1] - rec[i]
            p1, p2 = pre[i], pre[i + 1]

            S += (p1 + p2) * h / 2

        AP.append(S)
        
    df = pd.DataFrame(AP + [np.mean(AP)], columns=['AP'], index=(CAT_NAME_JP + ['mAP']))
        
    return df