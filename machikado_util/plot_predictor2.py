import numpy as np
import random
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog


def make_masked_image(src_img, masks, classes, mask_colors, alpha=0.5):
    tmp_src = src_img.astype(np.float).copy()

    dst_img = tmp_src.copy()
    
    if len(classes) > 0:  # 何も検出できない可能性があるので分類結果の個数はチェックする
        # 交差領域以外のマスク合成
        for i in range(len(masks)):
            mask = masks[i]
            dst_img[mask] = dst_img[mask] * (1 - alpha) + mask_colors[classes[i]][::-1] * alpha
        
        # 輪郭を生成する
        for i in range(len(masks)):
            mask = masks[i]
            pad = np.min(src_img.shape[:2]) // 150
            lw = pad * 2
            
            if pad == 0:
                pad, lw = 1, 2
            
            tmp_mask = np.pad(mask, ((0, 0), (pad, pad)), 'edge')
            b0 = tmp_mask[:, :-lw] ^ tmp_mask[:, lw:]
            tmp_mask = np.pad(mask, ((pad, pad), (0, 0)), 'edge')
            b1 = tmp_mask[:-lw, :] ^ tmp_mask[lw:, :]
            tmp_mask = np.pad(mask, ((pad, pad), (pad, pad)), 'edge')
            b2 = tmp_mask[:-lw, :-lw] ^ tmp_mask[lw:, lw:]
            
            dst_img[b0 | b1 | b2] = mask_colors[classes[i]][::-1]

    # 背景の合成（グレースケールへ変換）
    union_mask = np.zeros(masks.shape[1:], dtype=np.bool)

    for i in range(len(masks)):
        union_mask |= masks[i]
        
    bk = tmp_src
    bk = bk[:, :, 0] * 0.3 + bk[:, :, 1] * 0.6 + bk[:, :, 2] * 0.1
    bk = np.stack([bk, bk, bk], axis=2)

    if (~union_mask).sum() > 0:
        dst_img[~union_mask] = bk[~union_mask]
    
    return dst_img.astype(np.uint8)


def plot_output(ax, src_img, masks, bboxes, scores, classes, mask_colors, cat_names, alpha):
    dst_img = make_masked_image(src_img=src_img, masks=masks, classes=classes, mask_colors=mask_colors, alpha=alpha)

    ih, iw = dst_img.shape[:2]

    ax.imshow(dst_img[:, :, ::-1])

    for i in range(len(masks)):
        _bboxes = (bboxes[i].tensor).cpu().numpy().astype(np.int)
        
        pred_cls = classes[i]

        for bbox in _bboxes:
            c = mask_colors[pred_cls] / 255
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            r = patches.Rectangle(xy=(x1, y1), width=(x2 - x1), height=(y2 - y1), ec=c, linewidth=3, fill=False, alpha=alpha)
            ax.add_patch(r)

            boxdic = {
                'facecolor': np.clip((c * 2), 0, 1),
                'edgecolor': c / 2,
                'boxstyle': 'Round',
                'linewidth': 2
            }
            ax.text(x1, y1, '{} {:.0f}%'.format(cat_names[pred_cls], scores[i] * 100),
                    size=12, color='black', bbox=boxdic, fontweight='semibold', alpha=alpha)


def plot_outputs(inputs, outputs, titles, mask_colors, cat_names, alpha=0.5, figsize=(16, 24)):
    plt.figure(figsize=figsize)
    
    for i in range(len(outputs)):
        pred = outputs[i]['instances'].get_fields()

        masks = pred['pred_masks'].cpu().numpy()
        classes = pred['pred_classes'].cpu().numpy()
        bboxes = pred['pred_boxes']
        scores = pred['scores'].cpu().numpy()
        
        ax = plt.subplot(5, 2, i + 1)
    
        plot_output(ax=ax, src_img=inputs[i],
                    masks=masks, bboxes=bboxes, scores=scores, classes=classes,
                    mask_colors=mask_colors, cat_names=cat_names, alpha=alpha)
        
        ax.set_title(titles[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def plot_predictor2(predictor, catalog_name, mask_colors, cat_names, figsize=(16, 24), random_state=None):
    """
    予想と表示を行う
    """
    if random_state is not None:
        random.seed(random_state)
        
    inputs = []
    outputs = []
    titles = []
    
    for i, asset in enumerate(random.sample(DatasetCatalog.get(catalog_name), 10)):
        img = cv2.imread(asset["file_name"])
        
        inputs.append(img)
        outputs.append(predictor(img))
        titles.append(asset['file_name'])
    
    plot_outputs(inputs=inputs, outputs=outputs, titles=titles, mask_colors=mask_colors, cat_names=cat_names, alpha=0.5, figsize=(16, 24))