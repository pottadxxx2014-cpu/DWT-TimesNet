
import numpy as np
import pandas as pd
from pathlib import Path
from TimesNet_Process import TimesNet_model, TimesNet_data_generate
from runtime_analysis import (time_start, time_end, save_baseline_runtime, save_stage_runtime, isolated_rng)
from function_package import custom_round, DWT_process, DWT_two_frequency_method_process, backward_process
from scipy.interpolate import interp1d
from Markov_Process import build_matrix_for_days, markov_matrix_process
from CNN_Process import CNN_LSTM_model, CNN_data_generate
from DWT_CNN_TimesNet import CNN_based_transform, TimesNet_for_DWT, DWT_data_generating


def linear_interpolation_method(df_template):
    df = df_template.copy()
    df['baseline_forward'] = df['Truth'].where(((np.arange(len(df)) + 1) % 4 == 0), 0)

    # 线性插值
    linear_val = df['baseline_forward'].to_numpy()[3::4]
    N = len(linear_val)
    t = np.arange(N)
    t_aug = np.concatenate(([-1], t))
    y_aug = np.concatenate(([0], linear_val))
    t_new = np.linspace(-1, N - 1, 4 * N + 1)

    y_new_all_linear = np.interp(t_new, t_aug, y_aug)
    y_new_linear = y_new_all_linear[1:]

    df['Linear'] = y_new_linear
    df['Linear'] = df['Linear'].apply(custom_round)

    result_df = (df[['Linear']])

    return result_df


def spline_interpolation_method(df_template):
    df = df_template.copy()
    df['baseline_forward'] = df['Truth'].where(((np.arange(len(df)) + 1) % 4 == 0), 0)

    # 样条插值
    linear_val = df['baseline_forward'].to_numpy()[3::4]
    N = len(linear_val)
    t = np.arange(N)
    t_aug = np.concatenate(([-1], t))
    y_aug = np.concatenate(([0], linear_val))
    t_new = np.linspace(-1, N - 1, 4 * N + 1)

    spline_func = interp1d(t_aug, y_aug, kind='cubic')
    y_new_all_spline = spline_func(t_new)
    y_new_spline = y_new_all_spline[1:]

    df['Spline'] = y_new_spline
    df['Spline'] = df['Spline'].apply(custom_round)
    df.loc[df['Spline'] <= 0, 'Spline'] = 0

    result_df = (df[['Spline']])

    return result_df


def Markov_interpolation(high_occupancy_list, low_occupancy_list, train_test_index, test_index_name, max_set=25, log_dir=None):
    train_index, test_index = train_test_index

    # 获取转移矩阵
    train_start = time_start()
    matrix_list = build_matrix_for_days(high_occupancy_list, train_index, max_set)

    train_seconds = time_end(train_start)
    predict_start = time_start()

    # 利用转移矩阵生成
    result_df = markov_matrix_process(matrix_list, low_occupancy_list, test_index, test_index_name, steps=4, max_trials=1000)

    save_baseline_runtime(log_dir, 'Markov', train_seconds, time_end(predict_start), len(train_test_index[1]))
    return result_df


def CNN_interpolation(high_occupancy_list, low_occupancy_list, week_number, train_test_index, test_index_name, train_factor, log_dir=None, save_weights=False):
    # --------------------------------------------------------------
    # 设定模型参数值
    CNN_model_params = {
        "in_channels": 1,  # 输入数据通道数
        "conv1_out": 64,  # 第一层卷积输出通道数
        "conv2_out": 128,  # 第二层卷积输出通道数
        "conv3_out": 256,  # 第三层卷积输出通道数（映射特征数）
        "kernel_size": 3,  # 卷积核大小
        "padding": 1,  # 填充数，保证输出尺寸与输入相同
        "lstm_hidden": 512,  # LSTM 隐藏层单元数
        "num_lstm_layers": 1,  # LSTM 层数
        "width": 7,  # 一个batch包含的天数
        "dropout": 0.1
    }
    CNN_train_params = {
        "lr_rate": 0.001,  # Learning rate
        "num_epochs": int(500 * train_factor),  # 训练轮次
        "patience": 10  # 提前停止的容忍轮次
    }
    # --------------------------------------------------------------

    # 对低时频数据利用插值拓展，利用CNN映射到高时频数据（可以修改插值方法）
    model, test_data = CNN_LSTM_model(low_occupancy_list, high_occupancy_list, week_number, train_test_index, CNN_model_params, CNN_train_params, 'Linear', log_dir=log_dir, save_weights=save_weights)

    # 生成高时频数据
    result_df = CNN_data_generate(model, test_data, test_index_name, log_dir=log_dir)

    return result_df


def DWT_TimesNet_interpolation(high_occupancy_list, low_occupancy_list, week_number, train_test_index, test_index_name, train_factor, wavelet, level, picture, DWT_set=False, log_dir=None, save_weights=False):
    # --------------------------------------------------------------
    # 设定模型参数值
    model_params_CNN = {
        "in_channels": 1,  # 输入数据通道数
        "conv1_out": 128,  # 第一层卷积输出通道数
        "conv2_out": 256,  # 第二层卷积输出通道数
        "conv3_out": 512,  # 第三层卷积输出通道数（映射特征数）
        "kernel_size": 3,  # 卷积核大小
        "padding": 1,  # 填充数，保证输出尺寸与输入相同
        "lstm_hidden": 1024,  # LSTM 隐藏层单元数
        "num_lstm_layers": 1,  # LSTM 层数
        "width": 7,  # 一个batch包含的天数
        "dropout": 0.1
    }
    train_params_CNN = {
        "lr_rate": 0.001,  # Learning rate
        "num_epochs": int(500 * train_factor),  # 训练轮次
        "patience": 10  # 提前停止的容忍轮次
    }
    # --------------------------------------------------------------
    model_params_TimesNet_Max = {
        "top_k": 4,  # Top-k parameter
        "num_kernels": 3,  # Number of kernels
        "hidden_dim": 1,  # Hidden dimension size
        "task_name": 'imputation',
        "label_len": 1,
        "e_layers": 4,  # Number of TimesBlock layers
        "enc_in": 1,
        "d_model": 32,  # Embedding dimensions
        "d_ff": 64,  # Feed-forward layer size
        "embed": 'fixed',
        "freq": 't',
        "dropout": 0.1,
        "c_out": 1
    }
    train_params_TimesNet_Max = {
        "lr_rate": 0.001,
        "num_epochs": int(100 * train_factor),
        "patience": 10,
        "batch_size": 1
    }
    # --------------------------------------------------------------
    model_params_TimesNet_Middle = {
        "top_k": 2,  # Top-k parameter
        "num_kernels": 3,  # Number of kernels
        "hidden_dim": 1,  # Hidden dimension size
        "task_name": 'imputation',
        "label_len": 1,
        "e_layers": 3,  # Number of TimesBlock layers
        "enc_in": 1,
        "d_model": 32,  # Embedding dimensions
        "d_ff": 64,  # Feed-forward layer size
        "embed": 'fixed',
        "freq": 't',
        "dropout": 0.1,
        "c_out": 1
    }
    train_params_TimesNet_Middle = {
        "lr_rate": 0.001,
        "num_epochs": int(100 * train_factor),
        "patience": 10,
        "batch_size": 1
    }
    # --------------------------------------------------------------

    prepare_start = time_start()
    # 利用DWT方法获取不同频带下的人数波动（可以修改小波基函数）
    low_imfs_number_list, high_imfs_number_list, low_imfs_file_list, high_imfs_file_list = DWT_process(low_occupancy_list, high_occupancy_list, week_number, wavelet, level, picture)

    save_stage_runtime(log_dir, 'DWT_decomposition', time_end(prepare_start))

    # 对低时频数据利用插值拓展，并提取高时频数据的较低频分量，进行CNN映射（可以修改插值方法）
    CNN_model, test_data_for_CNN, fig_CNN_list = CNN_based_transform(low_imfs_file_list, high_imfs_file_list, week_number, train_test_index, model_params_CNN, train_params_CNN, 'Zero', picture, log_dir=log_dir, save_weights=save_weights)

    # 提取最高频分量和次高频分量
    X1, X2, Y1, Y2 = DWT_two_frequency_method_process(low_imfs_number_list, low_imfs_file_list, high_imfs_file_list, week_number, picture)

    # 利用TimesNet对两个分量进行拟合
    TimesNet_model_max, test_data_for_TimesNet_max = TimesNet_for_DWT(X1, Y1, train_test_index, 'Max', model_params_TimesNet_Max, train_params_TimesNet_Max, log_dir=log_dir, save_weights=save_weights)
    TimesNet_model_middle, test_data_for_TimesNet_middle = TimesNet_for_DWT(X2, Y2, train_test_index, 'Middle', model_params_TimesNet_Middle, train_params_TimesNet_Middle, log_dir=log_dir, save_weights=save_weights)

    # 生成高时频数据
    components = {}
    result_df = DWT_data_generating(CNN_model, test_data_for_CNN, TimesNet_model_max, test_data_for_TimesNet_max, TimesNet_model_middle, test_data_for_TimesNet_middle, test_index_name, wavelet, DWT_set, fig_CNN_list, picture, log_dir=log_dir, component_dict=components)

    # 先保存分支与真值；后续消融只读这个文件，不反推已取整的结果。
    if log_dir is not None:
        components['Truth'] = np.concatenate([high_occupancy_list[int(i)]['Occupancy'].to_numpy()
                                               for i in train_test_index[1]])
        columns = ['Truth', 'Low', 'D1', 'D2', 'Low_truth', 'D1_truth', 'D2_truth']
        pd.DataFrame(components, index=test_index_name)[columns].to_csv(
            Path(log_dir) / 'DWT_components.csv', encoding='utf-8-sig', float_format='%.17g')

    return result_df


def results_dataframe_merge(df_template, high_occupancy_list, low_occupancy_list, week_number, train_test_index, test_index_name, train_factor, wavelet_set, level_set, picture_set, max_set, log_dir=None, direct_set=True, save_weights=False):
    # 线性插值
    baseline_start = time_start()
    df_linear = linear_interpolation_method(df_template)
    save_baseline_runtime(log_dir, 'Linear', 0.0, time_end(baseline_start), len(train_test_index[1]))

    # 样条插值
    baseline_start = time_start()
    df_spline = spline_interpolation_method(df_template)
    save_baseline_runtime(log_dir, 'Spline', 0.0, time_end(baseline_start), len(train_test_index[1]))

    # 马尔可夫过程
    print('\nMarkov Interpolation Processing:')
    df_Markov = Markov_interpolation(high_occupancy_list, low_occupancy_list, train_test_index, test_index_name, max_set, log_dir=log_dir)

    # DWT-TimesNet方法
    print('\nDWT-TimesNet Interpolation Processing:')
    df_DWT_TimesNet = DWT_TimesNet_interpolation(high_occupancy_list, low_occupancy_list, week_number, train_test_index, test_index_name, train_factor, wavelet_set, level_set, picture_set, log_dir=log_dir, save_weights=save_weights)

    # CNN方法
    print('\nCNN Interpolation Processing:')
    df_CNN = CNN_interpolation(high_occupancy_list, low_occupancy_list, week_number, train_test_index, test_index_name, train_factor, log_dir=log_dir, save_weights=save_weights)

    # 拼接
    df_results_list = [df_template, df_linear, df_spline, df_Markov, df_CNN, df_DWT_TimesNet]
    df_results_all = pd.concat(df_results_list, axis=1)

    raw_results = df_results_all.copy()
    post_start = time_start()
    # 后处理
    back_number = 3  # 最后number列需要后处理
    df_last_n_columns = df_results_all.iloc[:, -back_number:]

    # 对每一列进行拆分，记作 'Generate'
    for col in df_last_n_columns.columns:
        df_new = pd.DataFrame({
            'Truth': df_results_all.iloc[:, 0],  # 提取第一列作为 Truth
            'Generate': df_last_n_columns[col]  # 拆分的列作为 Generate
        })

        df_results_all[col] = backward_process(df_new)

    save_stage_runtime(log_dir, 'Original_three_postprocess', time_end(post_start))

    # V2_2新增Direct TimesNet：放在原五种方法及其后处理之后，且隔离随机数。
    if direct_set:
        print('\nDirect TimesNet Interpolation Processing:')
        with isolated_rng():
            df_direct = TimesNet_interpolation(high_occupancy_list, low_occupancy_list, week_number,
                                               train_test_index, test_index_name, train_factor,
                                               log_dir=log_dir, save_weights=save_weights)
        post_start = time_start()
        pair = pd.DataFrame({'Truth': df_template['Truth'], 'Generate': df_direct.iloc[:, 0]}, index=test_index_name)
        direct_final = backward_process(pair)['pred_new'].to_numpy()
        location = df_results_all.columns.get_loc('DWT-TimesNet')
        df_results_all.insert(location, 'Direct TimesNet', direct_final)
        raw_results.insert(location, 'Direct TimesNet', df_direct.iloc[:, 0].to_numpy())
        save_stage_runtime(log_dir, 'Direct_TimesNet_postprocess', time_end(post_start))
    if log_dir is not None:
        raw_results.to_csv(Path(log_dir) / 'predictions_before_postprocess.csv', encoding='utf-8-sig', float_format='%.17g')
        df_results_all.to_csv(Path(log_dir) / 'predictions.csv', encoding='utf-8-sig')

    return df_results_all


def results_dataframe_merge_DWT(df_template, high_occupancy_list, low_occupancy_list, week_number, train_test_index, test_index_name, train_factor, level_set, picture_set, wavelet_list, log_dir=None, save_weights=False):

    df_DWT_TimesNet_list = []

    for wavelet_set in wavelet_list:
        # DWT-TimesNet方法
        print(f'\nDWT-TimesNet-{wavelet_set} Interpolation Processing:')
        wavelet_log = Path(log_dir) / ('wavelet_' + wavelet_set) if log_dir is not None else None
        if wavelet_log is not None: wavelet_log.mkdir(parents=True, exist_ok=True)
        df_DWT_TimesNet = DWT_TimesNet_interpolation(high_occupancy_list, low_occupancy_list, week_number, train_test_index, test_index_name, train_factor, wavelet_set, level_set, picture_set, DWT_set=True, log_dir=wavelet_log, save_weights=save_weights)
        df_DWT_TimesNet_list.append(df_DWT_TimesNet)

    # 拼接
    df_results_list = [df_template] + df_DWT_TimesNet_list
    df_results_all = pd.concat(df_results_list, axis=1)

    raw_results = df_results_all.copy()
    # 后处理
    back_number = len(df_results_list) - 1  # 最后number列需要后处理
    df_last_n_columns = df_results_all.iloc[:, -back_number:]


    # 对每一列进行拆分，记作 'Generate'
    for col in df_last_n_columns.columns:
        df_new = pd.DataFrame({
            'Truth': df_results_all.iloc[:, 0],  # 提取第一列作为 Truth
            'Generate': df_last_n_columns[col]  # 拆分的列作为 Generate
        })

        df_results_all[col] = backward_process(df_new)

    if log_dir is not None:
        raw_results.to_csv(Path(log_dir) / 'predictions_before_postprocess.csv', encoding='utf-8-sig', float_format='%.17g')
        df_results_all.to_csv(Path(log_dir) / 'predictions.csv', encoding='utf-8-sig')

    return df_results_all

def TimesNet_interpolation(high_occupancy_list, low_occupancy_list, week_number, train_test_index,
                          test_index_name, train_factor, log_dir=None, save_weights=False):
    # 新增的Direct基准：沿用最高频TimesNet容量，不改变原两个分支的参数。
    model_params = {
        "top_k": 4, "num_kernels": 3, "hidden_dim": 1,
        "task_name": 'imputation', "label_len": 1, "e_layers": 4,
        "enc_in": 1, "d_model": 32, "d_ff": 64,
        "embed": 'fixed', "freq": 't', "dropout": 0.1, "c_out": 1,
    }
    train_params = {
        "lr_rate": 0.001, "num_epochs": int(100 * train_factor),
        "patience": 10, "batch_size": 1,
    }
    model, test_data = TimesNet_model(low_occupancy_list, high_occupancy_list, week_number,
                                     train_test_index, model_params, train_params, 'Linear',
                                     log_dir=log_dir, save_weights=save_weights)
    return TimesNet_data_generate(model, test_data, test_index_name, log_dir=log_dir)
