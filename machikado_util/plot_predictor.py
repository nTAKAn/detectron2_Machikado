import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

def plot_predictor(predictor, catalog_name, figsize=(16, 24), random_state=None):
    """
    予想と表示を行う
    """
    if random_state is not None:
        random.seed(random_state)
    
    plt.figure(figsize=figsize)

    for i, asset in enumerate(random.sample(DatasetCatalog.get(catalog_name), 10)):
        w, h = asset['width'], asset['height']
        img = cv2.imread(asset["file_name"])
        outputs = predictor(img)

        v = Visualizer(img[:, :, ::-1],
                       metadata=MetadataCatalog.get(catalog_name), 
                       scale=max([300 / w, 300 / h]),  # 画像サイズに合わせてスケールを変更させた（見やすくなる）
                       instance_mode=ColorMode.IMAGE_BW)   # remove the colors of unsegmented pixels

        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        ax = plt.subplot(5, 2, i + 1)
        ax.imshow(v.get_image())
        ax.set_title(asset['file_name'])
        ax.axis('off')

    plt.tight_layout()
    plt.show()