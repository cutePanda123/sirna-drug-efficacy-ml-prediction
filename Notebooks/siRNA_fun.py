from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.metrics import mean_squared_error
import numpy as np
# calculate_validation_score for GridSearchCV
def calculate_validation_score(y_true, y_pred, threshold=30):
    # y_pred = preds
    # y_true = data.get_label()
    mae = np.mean(np.abs(y_true - y_pred))
    # if mae < 0: mae = 0
    # elif mae >100: mae = 100

    y_true_binary = ((y_true <= threshold) & (y_true >= 0)).astype(int)
    y_pred_binary = ((y_pred <= threshold) & (y_pred >= 0)).astype(int)

    mask = (y_pred >= 0) & (y_pred <= threshold)
    range_mae = (
        mean_absolute_error(y_true[mask], y_pred[mask]) if np.sum(mask) > 0 else 100
    )
    # if range_mae < 0: range_mae = 0
    # elif range_mae >100: range_mae = 100

    # precision = precision_score(y_true_binary, y_pred_binary, average="binary")
    # recall = recall_score(y_true_binary, y_pred_binary, average="binary")

    if np.sum(y_pred_binary) > 0:
        precision = (np.array(y_pred_binary) & y_true_binary).sum()/np.sum(y_pred_binary)
    else:
        precision = 0
    if np.sum(y_true_binary) > 0:
        recall = (np.array(y_pred_binary) & y_true_binary).sum()/np.sum(y_true_binary)
    else:
        recall = 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    score = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
    return score

custom_scorer = make_scorer(calculate_validation_score, greater_is_better=True)


# calculate_metrics for lightgbm training
def calculate_validation_score_for_training(preds, data, threshold=30):
    y_pred = preds
    y_true = data.get_label()
    mae = np.mean(np.abs(y_true - y_pred))
    # if mae < 0: mae = 0
    # elif mae >100: mae = 100

    y_true_binary = ((y_true <= threshold) & (y_true >= 0)).astype(int)
    y_pred_binary = ((y_pred <= threshold) & (y_pred >= 0)).astype(int)

    mask = (y_pred >= 0) & (y_pred <= threshold)
    range_mae = (
        mean_absolute_error(y_true[mask], y_pred[mask]) if np.sum(mask) > 0 else 100
    )
    # if range_mae < 0: range_mae = 0
    # elif range_mae >100: range_mae = 100

    # precision = precision_score(y_true_binary, y_pred_binary, average="binary")
    # recall = recall_score(y_true_binary, y_pred_binary, average="binary")

    if np.sum(y_pred_binary) > 0:
        precision = (np.array(y_pred_binary) & y_true_binary).sum()/np.sum(y_pred_binary)
    else:
        precision = 0
    if np.sum(y_true_binary) > 0:
        recall = (np.array(y_pred_binary) & y_true_binary).sum()/np.sum(y_true_binary)
    else:
        recall = 0

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    score = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
    return "custom_score", score, True


# For the training data, double the size of the observations with y < 30:
def Get_sample_weight(y, threshold=30):
    weights = list(map(lambda x: 2 if (x<=threshold and x>=0) else 1, y))
    weights = np.array(weights)
    return weights


def custom_rmse(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return 'custom_rmse', rmse, False

def custom_rmse_for_xgb(y_true, y_pred):
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Return the metric name, value, and whether higher values are better (False for RMSE)
    return 'custom_rmse', rmse


