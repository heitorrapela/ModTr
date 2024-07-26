import torch
import torchmetrics
import torchmetrics.detection # Lazy import

# https://github.com/Lightning-AI/metrics/issues/1024 
try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection import MAP
    MeanAveragePrecision = MAP


class Detection():

    def __init__(self, box_format='xyxy', device='cpu', class_metrics=False):

        self.device = device
        self.map = self.metric_map(class_metrics=class_metrics)
        
    # MAP Metric
    def metric_map(self, class_metrics=False):
        return MeanAveragePrecision(class_metrics=class_metrics).to(self.device) #(box_format=box_format).to(self.device)