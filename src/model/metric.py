import torch
from sklearn import metrics
from sklearn.metrics import f1_score


def accuracy(y_predict, y_correct):
    with torch.no_grad():
        assert y_predict.shape[0] == len(y_correct)
        correct = 0
        correct += torch.sum(y_predict == y_correct).item()
    return correct / len(y_correct)


# def top_k_acc(y_predict, y_correct, k=3):
#     with torch.no_grad():
#         pred = torch.topk(y_predict, k, dim=1)[1]
#         assert pred.shape[0] == len(y_correct)
#         correct = 0
#         for i in range(k):
#             correct += torch.sum(pred[:, i] == y_correct).item()
#     return correct / len(y_correct)


def F1_score(y_predict, y_correct):
    with torch.no_grad():
        y_predict = y_predict.cpu()
        y_predict.detach().numpy()
        y_correct = y_correct.cpu()
        y_correct.detach().numpy()
        y_correct = y_correct.reshape(-1, 1)
    result = f1_score(y_correct, y_predict)

    return result


def auc_score(y_predict, y_correct):
    y_predict = y_predict.cpu().detach().numpy()

    y_correct = y_correct.cpu().detach().numpy()

    fpr, tpr, thresholds = metrics.roc_curve(y_correct, y_predict)
    result = metrics.auc(fpr, tpr)

    return result


# def roc_auc_score(y_predict,y_correct):
