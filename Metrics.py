from torchmetrics import ConfusionMatrix,Accuracy,Precision,Recall
from torchmetrics.classification import MulticlassF1Score
import seaborn as sns

class ComputeMetrics:
    def __init__(self,c):
        self.num_classes= c
        self.f1 = MulticlassF1Score( num_classes=self.num_classes,average=None)
        self.acc = Accuracy(task="multiclass", num_classes=self.num_classes,average=None)
        self.pre = Precision(task="multiclass", num_classes=self.num_classes,average=None)
        self.rec = Recall(task="multiclass", num_classes=self.num_classes,average=None)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

    def __call__(self,preds,targets):
        metrics_dict = {"f1":self.f1,"accuracy":self.acc,"precesion":self.pre,"recall":self.rec,"confmat":self.confmat}
        class_metric_values = {}
        macro_metric_values = {}
        for name, metric in metrics_dict.items():
            class_metric_values[name] = metric(preds,targets)
            if name !="confmat":
                macro_metric_values[name] = class_metric_values[name].mean()
        return class_metric_values,macro_metric_values

    def print_metrics(self,preds,targets):
        class_metric_values,macro_metric_values = self(preds,targets)
        for name,value in macro_metric_values.items():
            print(f"{name}: ** Avg {round(value.item(),3)} **  : class : |  {'  |  '.join( [ format(x , '0.3f')  for x in class_metric_values[name]])}")
        sns.heatmap(class_metric_values["confmat"],annot=True,cmap="crest",cbar=False,xticklabels=list(range(self.num_classes)),yticklabels=list(range(self.num_classes)),fmt='g')
