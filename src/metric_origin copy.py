import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torchmetrics import AUROC
from torchmetrics import Accuracy
from torchmetrics.classification import MulticlassF1Score

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve
from torchmetrics import ConfusionMatrix


def accuracy(y_predict, y_correct):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = Accuracy(num_classes=2, multiclass=True).to(device)
    predict_softmax = F.softmax(y_predict, dim=1)
    correct_flatten = torch.flatten(y_correct)
    return metric(predict_softmax, correct_flatten).item()


def auroc(y_predict, y_correct):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric = AUROC(num_classes=2).to(device)

    predict_softmax = F.softmax(y_predict, dim=1)
    correct_flatten = torch.flatten(y_correct)

    return metric(predict_softmax, correct_flatten).item()


def f1(y_predict, y_correct):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predict_softmax = F.softmax(y_predict, dim=1)
    correct_flatten = torch.flatten(y_correct)
    metric = MulticlassF1Score(num_classes=2).to(device)
    return metric(predict_softmax, correct_flatten).item()


def final_test_metric(y_predict, y_correct) -> dict:
    result_dict = {}
    print("before thresh")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm = ConfusionMatrix(num_classes=2).to(device)
    print(cm(y_predict, y_correct))

    y_pre_np = y_predict[:, 1].squeeze().to("cpu")
    y_pre_np = y_pre_np.numpy()
    y_corr_np = y_correct.squeeze().to("cpu")
    y_corr_np = y_corr_np.numpy()
    fpr, tpr, threshholds = roc_curve(y_corr_np, y_pre_np)
    draw(fpr, tpr)

    f = lambda x: 1.0 - x - interp1d(fpr, tpr)(x)
    eer = brentq(f, 0.0, 1.0)
    thresh = interp1d(fpr, threshholds)(eer).tolist()

    y_pre_discrete = (y_pre_np > thresh).astype("int")
    matrix = confusion_matrix(y_corr_np, y_pre_discrete)
    result_dict["f1"] = f1_score(y_corr_np, y_pre_discrete)
    result_dict["accuracy"] = accuracy_score(y_corr_np, y_pre_discrete)
    result_dict["auroc"] = auroc(y_predict, y_correct)

    y_pre_act = F.sigmoid(y_predict[:, 1])
    thresh_act = F.sigmoid(torch.Tensor([thresh]).to(device))

    draw_histo(y_pre_act * 200, y_pre_discrete.astype("int"), thresh_act * 200)
    print("after thresh")
    print(matrix)
    return result_dict


def draw(x, y):
    plt.plot(x, y, color="red")
    plt.plot([0.0, 1.0], [0.0, 1.0], color="black")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("plot/ROC_CURVE.png")


def draw_histo(pat_score):
    df = pd.read_excel("sig200.xlsx")
    thresh = 32.23144
    fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    sns.histplot(data=df, x="x_percent", ax=ax, hue="df_class", kde=True)

    # threshold_line
    ax.axvline(thresh, color="black", linewidth=3)
    str_thresh = "Threshold: " + str(thresh)
    ax.text(thresh + 1, 250, str_thresh)  # threshold text

    pat_text = "Patient:" + str(pat_score)
    ax.axvline(50, color="red", linewidth=3)
    ax.text(pat_score + 1, 250, pat_text, color="red")

    btw = "Between point:" + str(50 - thresh[0]) + "\n Warning: Danger!"

    # if 점수가 넘어가면~ 안넘어가면~
    ax.text(((thresh[0] + 50) / 2) - 10, 250, btw, color="red")

    # 지금까지 그림들 다 이렇게 저장됨.
    fig.savefig("HISTO.png")
