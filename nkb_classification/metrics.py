import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(epoch_results):
    running_loss = epoch_results['running_loss']
    confidences = epoch_results['confidences']
    predictions = epoch_results['predictions']
    ground_truth = epoch_results['ground_truth']

    epoch_acc = accuracy_score(ground_truth, predictions)
    epoch_precision = precision_score(ground_truth, np.array(confidences).argmax(axis=1))
    epoch_recall = recall_score(ground_truth, np.array(confidences).argmax(axis=1))
    epoch_roc_auc = roc_auc_score(ground_truth, np.array(confidences)[:, 1])
    epoch_loss = np.mean(running_loss)
    metrics = {
        'epoch_acc': epoch_acc,
        'epoch_precision': epoch_precision,
        'epoch_recall': epoch_recall,
        'epoch_roc_auc': epoch_roc_auc,
        'epoch_loss': epoch_loss,
    }
    return metrics


def log_metrics(experiment,
                epoch,
                metrics,
                fold='Train'):
    acc = metrics['epoch_acc']
    precision = metrics['epoch_precision']
    recall = metrics['epoch_recall']
    roc_auc = metrics['epoch_roc_auc']
    epoch_loss = metrics['epoch_loss']
    # print(f'Epoch {epoch} {fold.lower()} roc_auc {roc_auc:.4f}')
    # print(f'Epoch {epoch} {fold.lower()} balanced accuracy {acc:.4f}')
    experiment.log_metric(f'{fold} accuracy', acc, epoch=epoch, step=epoch)
    experiment.log_metric(f'{fold} precision', precision, epoch=epoch, step=epoch)
    experiment.log_metric(f'{fold} recall', recall, epoch=epoch, step=epoch)
    experiment.log_metric(f'{fold} ROC AUC', roc_auc, epoch=epoch, step=epoch)
    experiment.log_metric(f'{fold} loss', epoch_loss, epoch=epoch, step=epoch)


def log_confusion_matrix(experiment,
                         label_names,
                         epoch,
                         results,
                         fold='Validation'):
    experiment.log_confusion_matrix(results['ground_truth'],
                                    results['predictions'],
                                    labels=label_names, 
                                    title=f'{fold} confusion matrix',
                                    file_name=f'{fold}-confusion-matrix.json',
                                    epoch=epoch)
