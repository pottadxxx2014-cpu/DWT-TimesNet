
import torch.nn as nn
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm
from function_package import interpolate_occupancy_zero, interpolate_occupancy_linear, interpolate_occupancy_spline
import matplotlib.pyplot as plt
import time
from runtime_analysis import (time_start, time_end, start_training_record, finish_training_record,
                              save_training_record, save_rng_state, save_model_runtime)

# 定义集成 LSTM 的 CNN 模型
class LSTMCNN(nn.Module):
    def __init__(self, in_channels, conv1_out, conv2_out, conv3_out,
                 kernel_size, padding, lstm_hidden, num_lstm_layers, width, dropout):
        super(LSTMCNN, self).__init__()
        # 卷积层部分
        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=kernel_size, padding=padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=kernel_size, padding=padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(conv2_out, conv3_out, kernel_size=kernel_size, padding=padding)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)

        # LSTM 部分：
        # 卷积输出形状为 (batch, conv3_out, 96, 7)
        # 将 96 视为时序步长，每一步特征维度为 conv3_out * 7
        self.lstm_input_size = conv3_out * width
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=lstm_hidden,
                            num_layers=num_lstm_layers,
                            batch_first=True,
                            dropout=dropout if num_lstm_layers > 1 else 0.0,
                            bidirectional=True
                            )
        # 通过全连接层将 LSTM 输出映射回原始的 width（7）
        self.fc = nn.Linear(lstm_hidden * 2, width)

        # 保存模型配置信息
        self.config = {
            "in_channels": in_channels,
            "conv1_out": conv1_out,
            "conv2_out": conv2_out,
            "conv3_out": conv3_out,
            "kernel_size": kernel_size,
            "padding": padding,
            "lstm_hidden": lstm_hidden,
            "num_lstm_layers": num_lstm_layers,
            "width": width,
            "dropout": dropout
        }

    def forward(self, x):
        # x 的形状：(batch, 1, 96, 7)
        x = self.dropout1(self.relu1(self.conv1(x)))  # -> (batch, conv1_out, 96, 7)
        x = self.dropout2(self.relu2(self.conv2(x)))  # -> (batch, conv2_out, 96, 7)
        x = self.dropout3(self.relu3(self.conv3(x)))  # -> (batch, conv3_out, 96, 7)

        batch_size, channels, height, width = x.size()  # height 应为 96, width 为 7
        # 重排张量，将每一行视为一个时间步：转换为 (batch, 96, conv3_out * 7)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, height, -1)

        # LSTM 处理时序数据，输出形状为 (batch, 96, lstm_hidden)
        lstm_out, _ = self.lstm(x)
        # 对每个时序步通过全连接层映射为 7 维输出
        out = self.fc(lstm_out)  # (batch, 96, 7)
        # 增加 channel 维度，转换为 (batch, 1, 96, 7)
        out = out.unsqueeze(1)
        return out



def CNN_LSTM_model(low_file_list, high_file_list, week_number, train_test_index, model_params, train_params, method, log_dir=None, save_weights=False):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    indexs = []
    for day in days:
        for i in range(1, 97):  # 每天 96 个时段，从 1 到 96
            total_minutes = i * 15
            hours = total_minutes // 60
            minutes = total_minutes % 60
            time_label = f"{hours}:{minutes:02d}"
            indexs.append(f"{day} {time_label}")

    non_freq_dict = {}
    spline_dict = {}

    for week in range(week_number):
        low_file = low_file_list[week]
        high_file = high_file_list[week].copy()  # 不把Mon/Tue绘图标签写回原日期索引
        high_file.index = indexs

        non_freq_dict[f'Week_{week + 1}'] = high_file['Occupancy']

        # 对 low_file 进行零阶插值、线性插值、样条插值（可以对比）
        if method == 'Zero':
            spline_values = interpolate_occupancy_zero(low_file)
        elif method == 'Linear':
            spline_values = interpolate_occupancy_linear(low_file)
        else:
            spline_values = interpolate_occupancy_spline(low_file)
        spline_dict[f'Week_{week + 1}'] = pd.Series(spline_values, index=indexs)

    high_occ_file = pd.DataFrame(non_freq_dict, index=indexs)
    high_spline_file = pd.DataFrame(spline_dict, index=indexs)

    height = 96  # 行数（每天 96/7 = 96 ）
    width = 7  # 列数
    X_list = []  # 输入：high_spline_file（映射的源数据）
    Y_list = []  # 目标：high_non_frequency_file（映射目标）

    for i in range(1, week_number + 1):
        col_name = f'Week_{i}'
        x_series = high_spline_file[col_name].values  # shape (672,)
        y_series = high_occ_file[col_name].values  # shape (672,)
        x_img = x_series.reshape((width, height)).T  # V2_1：每列一天
        y_img = y_series.reshape((width, height)).T  # V2_1：目标采用相同排列

        x_img = np.expand_dims(x_img, axis=0)  # (1,96,7)
        y_img = np.expand_dims(y_img, axis=0)  # (1,96,7)
        X_list.append(x_img)
        Y_list.append(y_img)

    X_np = np.array(X_list)  # shape (week_number, 1, 96, 7)
    Y_np = np.array(Y_list)  # shape (week_number, 1, 96, 7)

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float32)

    train_index, test_index = train_test_index
    X_train = X_tensor[train_index]
    Y_train = Y_tensor[train_index]
    X_test = X_tensor[test_index]
    Y_test = Y_tensor[test_index]
    test_data = [X_test, Y_test]

    lr_rate = train_params['lr_rate']
    num_epochs = train_params['num_epochs']
    patience = train_params['patience']

    save_rng_state(log_dir, 'CNN')  # 只存随机状态，不重置/消耗随机数
    model = LSTMCNN(**model_params)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Using device for CNN:", device)

    # 暂停1s
    time.sleep(1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    best_loss = float('inf')
    patience_counter = 0

    yellow_desc = "\033[93mCNN-LSTM Processing \033[0m"
    bar_format = "\033[93m{l_bar}\033[93m{bar}\033[93m{r_bar}\033[0m"

    training_history = []
    start_training_record(model, device)
    for epoch in tqdm(range(num_epochs), desc=yellow_desc, bar_format=bar_format):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train.to(device))
        loss = criterion(outputs, Y_train.to(device))
        loss.backward()
        optimizer.step()

        training_history.append({'epoch': epoch + 1, 'train_mse': loss.detach().item()})

        # 如果当前 loss 更低，则更新 best_loss 并重置耐心计数器
        if loss.detach().item() < best_loss:
            best_loss = loss.detach().item()
            patience_counter = 0
        else:
            patience_counter += 1

        # 如果连续 early_stopping_patience 个 epoch loss 没有改善，则提前停止训练
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    finish_training_record(model, device)
    save_training_record(model, training_history, log_dir, 'CNN', model_params,
                         train_params, train_test_index, save_weights=save_weights)
    test_start = time_start(device)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test.to(device))
        val_loss = criterion(test_outputs, Y_test.to(device))
    model.test_evaluation_seconds = time_end(test_start, device)
    print("Test Loss for CNN Only:", val_loss.item())

    #
    # for i in range(3):
    #     pred_first = test_outputs[i, 0].cpu().numpy().T.reshape(-1)
    #     # 获取测试集对应原始标签
    #     orig_first = Y_test[i, 0].cpu().numpy().T.reshape(-1)
    #
    #     # 构造结果 DataFrame，将预测结果与原始数据放在一起
    #     result_df = pd.DataFrame({
    #         'Prediction': pred_first,
    #         'Original': orig_first
    #     }, index=indexs)
    #
    #     # 绘制对比图
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(result_df.index, result_df['Original'], label='Truth Data')
    #     plt.plot(result_df.index, result_df['Prediction'], label='Generated Lower-frequency Data', linestyle='--')
    #     plt.xlabel('Time')
    #     plt.ylabel('Occupancy')
    #     plt.xticks([])
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig(f"CNN Week {i + 1} Original vs Prediction.png")
    #     plt.close('all')

    return model, test_data


def CNN_data_generate(model, data ,test_index, log_dir=None):
    # 设备配置：有 cuda 就使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device for test data generating: {device}")
    criterion = nn.MSELoss()

    X_test, Y_test = data
    predict_start = time_start(device)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test.to(device))

    CNN_data_pred = test_outputs.cpu().numpy()[:, 0, :, :].transpose(0, 2, 1).reshape(-1)  # V2_1：先还原每天，再展开各周

    df = pd.DataFrame({
        "CNN": CNN_data_pred
    }, index=test_index)

    model.inference_seconds = time_end(predict_start, device)
    save_model_runtime(model, log_dir, 'CNN', 'CNN', len(test_index) // 672)
    return df