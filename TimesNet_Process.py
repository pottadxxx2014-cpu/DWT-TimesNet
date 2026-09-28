# V2_2 新增：写法对应 CNN_Process.py；只做 Direct TimesNet 的输入准备和输出整理。
import numpy as np
import pandas as pd
import torch

from function_package import interpolate_occupancy_zero, interpolate_occupancy_linear, interpolate_occupancy_spline
from DWT_CNN_TimesNet import TimesNet_for_DWT, timesnet_pred_generate


def TimesNet_model(low_file_list, high_file_list, week_number, train_test_index,
                  model_params, train_params, method, log_dir=None, save_weights=False):
    X_list = []
    Y_list = []

    for week in range(week_number):
        low_file = low_file_list[week]
        high_file = high_file_list[week]

        # 输入只能使用该周的小时观测，不使用未知15分钟真值。
        if method == 'Zero':
            input_values = interpolate_occupancy_zero(low_file)
        elif method == 'Linear':
            input_values = interpolate_occupancy_linear(low_file)
        else:
            input_values = interpolate_occupancy_spline(low_file)

        X_list.append(np.asarray(input_values, dtype=np.float32).reshape(-1, 1))
        Y_list.append(high_file['Occupancy'].to_numpy(dtype=np.float32).reshape(-1, 1))

    X_tensor = torch.tensor(np.stack(X_list), dtype=torch.float32)
    Y_tensor = torch.tensor(np.stack(Y_list), dtype=torch.float32)
    if X_tensor.shape != Y_tensor.shape:
        raise ValueError('Direct TimesNet 的输入和目标长度不一致。')

    # 复用V2_1训练MSE早停，使用末轮权重；没有验证集。
    model, test_data = TimesNet_for_DWT(
        X_tensor, Y_tensor, train_test_index, 'Direct', model_params, train_params,
        log_dir=log_dir, save_weights=save_weights)
    return model, test_data


def TimesNet_data_generate(model, data, test_index, log_dir=None):
    device = next(model.parameters()).device
    pred, truth, fig_list = timesnet_pred_generate(
        data, model, 'Direct', 'Blue', device, False, log_dir=log_dir)
    if len(pred) != len(test_index):
        raise ValueError('Direct TimesNet 预测长度与测试索引长度不一致。')
    result_df = pd.DataFrame({'Direct TimesNet': pred}, index=test_index)
    return result_df
