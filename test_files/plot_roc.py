import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer


def binary_auc(trues, probs, configs_list, results_path):
    class_of_interest_index = 1
    model_idx = 0  # iterate over models
    for y_true, y_prob in zip(trues, probs):
        color = configs_list[model_idx][0]
        model_type = configs_list[model_idx][1]
        RocCurveDisplay.from_predictions(
            y_true,
            np.array(y_prob)[:, class_of_interest_index],
            name=f"{model_type}",
            color=color,
            ax=plt.gca()
        )
        model_idx += 1

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-averaged One-vs-Rest")
    plt.savefig(
        '{}/ROC_FULL_{}.png'.format(results_path + '/test_avg_auc', 'MicroAVG'))


def multiclass_auc(trues, probs, configs_list, results_path):
    model_idx = 0  # iterate over models
    for y_true, y_prob in zip(trues, probs):
        color = configs_list[model_idx][0]
        model_type = configs_list[model_idx][1]
        label_binarizer = LabelBinarizer().fit(y_true)
        y_onehot_test = label_binarizer.transform(y_true)
        RocCurveDisplay.from_predictions(
            y_onehot_test.ravel(),
            np.array(y_prob).ravel(),
            name=f"{model_type}",
            color=color,
            ax=plt.gca()
        )
        model_idx += 1

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-averaged One-vs-Rest")
    plt.legend(prop={'size': 9})
    plt.savefig('{}/ROC_FULL_{}_9_fat_vs_vlfat_clinical.png'.format(results_path, 'MicroAVG'))

    plt.clf()

    model_idx = 0  # iterate over models
    for y_true, y_prob in zip(trues, probs):
        color = configs_list[model_idx][0]
        model_type = configs_list[model_idx][1]
        label_binarizer = LabelBinarizer().fit(y_true)
        y_onehot_test = label_binarizer.transform(y_true)
        RocCurveDisplay.from_predictions(
            y_onehot_test.ravel(),
            np.array(y_prob).ravel(),
            name=f"{model_type}",
            color=color,
            ax=plt.gca()
        )
        model_idx += 1

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Micro-averaged One-vs-Rest")
    plt.legend()
    plt.savefig('{}/ROC_FULL_{}_fat_vs_vlfat_clinical.png'.format(results_path+'/test_avg_auc', 'MicroAVG'))