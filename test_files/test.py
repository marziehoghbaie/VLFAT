import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn.functional as F
from pretty_confusion_matrix import pp_matrix
from sklearn import metrics
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import roc_auc_score, roc_curve


def test(loader, model, loss_fn, logger, phase, device='cuda'):
    model.eval()
    running_corrects = 0
    running_loss = 0
    trues = []
    y_prob = []
    y_pred = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            prob, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            trues.extend(labels.cpu())
            y_pred.extend(preds.cpu())
            y_prob.extend(prob.cpu())
            running_corrects += preds.eq(labels.view_as(preds)).sum().item()

            loss = loss_fn(logits, labels)
            running_loss += loss.item() * images.size(0)

    accuracy = running_corrects / len(loader.dataset)
    'y_true, y_pred'
    balanced_acc = balanced_accuracy_score(y_true=trues, y_pred=y_pred)
    loss = running_loss / len(loader.dataset)
    logger.info('[INFO] {} acc, balanced accuracy,  and loss: {}, {}, {}'.format(phase, accuracy, balanced_acc, loss))

    return accuracy, loss, balanced_acc


def test_complete(loader, model, loss_fn, logger, phase, save_path, device='cuda', n_test=0):
    model.eval()
    auc = 0
    running_corrects = 0
    running_loss = 0
    trues = []
    y_prob = []
    y_pred = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)  # accelerator handles it
            logits = model(images)

            prob, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            trues.extend(labels.cpu())
            y_pred.extend(preds.cpu())
            y_prob.extend(prob.cpu())
            running_corrects += preds.eq(labels.view_as(preds)).sum().item()

            loss = loss_fn(logits, labels)
            running_loss += loss.item() * images.size(0)

    accuracy = running_corrects / len(loader.dataset)
    balanced_acc = balanced_accuracy_score(y_true=trues, y_pred=y_pred)
    loss = running_loss / len(loader.dataset)
    logger.info(
        '[INFO] running corrects:{}, {} acc, balanced accuracy,  and loss, n_test: {}, {}, {}, {}'.format(running_corrects, phase, accuracy, balanced_acc,
                                                                                     loss, n_test))
    "get the classification report"
    classification_report = metrics.classification_report(y_true=trues, y_pred=y_pred,
                                                          target_names=loader.dataset.categories)
    logger.info('[INFO] classification report \n')
    logger.info(classification_report)
    #
    n_classes = len(loader.dataset.categories)
    categories = loader.dataset.categories
    draw_conMatrix(trues, y_pred, n_test, save_path, n_classes, categories)

    if len(loader.dataset.categories) == 2:
        "cal auc for binary classification"
        auc = draw_roc(trues, y_prob, save_path, logger)

    return balanced_acc, loss, auc


def test_complete_Robustness(loader, model, device='cuda'):
    model.eval()
    trues = []
    y_prob = []
    y_pred = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)  # accelerator handles it
            logits = model(images)
            prob, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            trues.extend(labels.cpu())
            y_pred.extend(preds.cpu())
            y_prob.extend(prob.cpu())

    balanced_acc = balanced_accuracy_score(y_true=trues, y_pred=y_pred)

    return balanced_acc


def draw_conMatrix(trues, y_pred, n_test, save_path, n_classes, categories):
    # # draw the confusion matrix
    conf_matrix = metrics.confusion_matrix(trues, y_pred)
    df_cm = pd.DataFrame(conf_matrix, index=range(1, n_classes + 1),
                         columns=categories)

    cmap = 'OrRd'
    pp_matrix(df_cm, cmap=cmap)
    plt.savefig('{}/conf_mtrx_{}.png'.format(save_path, str(n_test)))

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                                display_labels=categories)

    cm_display.plot(cmap="OrRd", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig('{}/conf_mtrx_simple_{}.png'.format(save_path, str(n_test)))


def draw_roc(y_true, y_prob, save_path, logger):
    # calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob, pos_label=1)
    AP = average_precision_score(y_true, y_prob, average="macro")
    display = PrecisionRecallDisplay(
        recall=recall,
        precision=precision,
        average_precision=AP,
    )
    display.plot()
    _ = display.ax_.set_title(f"Precision Recall curve(full volumes)")
    plt.savefig('{}/PrecisionRecall_curve.png'.format(save_path))

    RocCurveDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_prob,
        name=f"ROC Curve",
        color="darkorange")

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves")
    plt.legend()
    plt.savefig('{}/ROC.png'.format(save_path))
    auc = roc_auc_score(y_true, y_prob)
    logger.info(f'[INFO] AUC score: {auc}')
    return auc