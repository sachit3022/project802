from torchmetrics import ConfusionMatrix,Accuracy,Precision,Recall
from torchmetrics.classification import MulticlassF1Score
import seaborn as sns

num_classes=10

f1 = MulticlassF1Score( num_classes=num_classes,average=None)
acc = Accuracy(task="multiclass", num_classes=num_classes,average=None)
pre = Precision(task="multiclass", num_classes=num_classes,average=None)
rec = Recall(task="multiclass", num_classes=num_classes,average=None)
confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

def compute_metrics(preds,targets):
    metrics_dict = {"f1       ":f1,"accuracy ":acc,"precesion":pre,"recall   ":rec,"confmat":confmat}
    class_metric_values = {}
    macro_metric_values = {}
    for name, metric in metrics_dict.items():

        class_metric_values[name] = metric(preds,targets)
        if name !="confmat":
            macro_metric_values[name] = class_metric_values[name].mean()
    return class_metric_values,macro_metric_values


def print_metrics(preds,targets):
    class_metric_values,macro_metric_values = compute_metrics(preds,targets)
    for name,value in macro_metric_values.items():
        print(f"{name}: ** Avg {round(value.item(),3)} **  : class : |  {'  |  '.join( [ format(x , '0.3f')  for x in class_metric_values[name]])}")
    sns.heatmap(class_metric_values["confmat"],annot=True,cmap="crest",cbar=False,xticklabels=list(range(num_classes)),yticklabels=list(range(num_classes)),fmt='g')
