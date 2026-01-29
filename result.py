from evaluation.evaluate import *
from evaluation.roc_auc import plot_roc
from evaluation.confusion import plot_confusion

metrics, y_true, y_pred, y_prob = evaluate(model, val_loader)

plot_roc(y_true, y_prob)
plot_confusion(y_true, y_pred)
