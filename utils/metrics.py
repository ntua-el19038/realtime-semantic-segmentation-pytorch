from torchmetrics import JaccardIndex
from torchmetrics import Dice
# from torchmetrics.segmentation import GeneralizedDiceScore
def get_seg_metrics(config, task='multiclass', reduction='none'):
    # if config.dataset=='prostate':
    #     metrics =Dice(num_classes=config.num_class)
    # else:
    metrics = JaccardIndex(task=task, num_classes=config.num_class,
                            ignore_index=config.ignore_index, average=reduction)
    return metrics