import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from TimesNet import Model
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from function_package import interpolate_occupancy_zero, interpolate_occupancy_linear, interpolate_occupancy_spline
from types import SimpleNamespace
import time
from runtime_analysis import (time_start, time_end, start_training_record, finish_training_record,
                              save_training_record, save_rng_state, save_model_runtime)
from matplotlib.gridspec import GridSpec

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



def CNN_based_transform(low_file_list, high_file_list, week_number, train_test_index, model_params, train_params, method, picture, log_dir=None, save_weights=False):
    # 创建一个index
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    indexs = []
    for day in days:
        for i in range(1, 97):  # 每天 96 个时段，从 1 到 96
            total_minutes = i * 15
            hours = total_minutes // 60
            minutes = total_minutes % 60
            # 格式化时间标签，例如 "0:15", "0:30"……"24:00"
            time_label = f"{hours}:{minutes:02d}"
            indexs.append(f"{day} {time_label}")

    non_freq_dict = {}
    spline_dict = {}

    for week in range(week_number):
        low_file = low_file_list[week]
        high_file = high_file_list[week].copy()  # 不把Mon/Tue绘图标签写回原日期索引
        high_file.index = indexs
        # 筛选 high_file 中以 "A" 或 "D" 开头的列
        imf_columns = [col for col in high_file.columns if col.startswith("A") or col.startswith("D")]
        # 去除最后两个分量（假设最后两个为高频部分，不需要）
        nonhigh_imf_cols = imf_columns[:-2]
        non_freq_dict[f'Week_frequency_{week + 1}'] = high_file[nonhigh_imf_cols].sum(axis=1)

        # 对 low_file 进行零阶插值、线性插值、样条插值（可以对比）
        if method == 'Zero':
            spline_values = interpolate_occupancy_zero(low_file)
        elif method == 'Linear':
            spline_values = interpolate_occupancy_linear(low_file)
        else:
            spline_values = interpolate_occupancy_spline(low_file)
        spline_dict[f'Week_frequency_{week + 1}'] = pd.Series(spline_values, index=indexs)

    high_non_frequency_file = pd.DataFrame(non_freq_dict, index=indexs)
    high_spline_file = pd.DataFrame(spline_dict, index=indexs)

    # 构建训练数据
    # 每一列（长度 672）重塑为 (96, 7)，再增加 channel 维度变为 (1,96,7)
    # -------------------------------
    height = 96  # 行数（每天 96/7 = 96 ）
    width = 7  # 列数
    X_list = []  # 输入：high_spline_file（映射的源数据）
    Y_list = []  # 目标：high_non_frequency_file（映射目标）

    for i in range(1, week_number + 1):
        col_name = f'Week_frequency_{i}'
        x_series = high_spline_file[col_name].values  # shape (672,)
        y_series = high_non_frequency_file[col_name].values  # shape (672,)
        x_img = x_series.reshape((width, height)).T  # V2_1：每列一天
        y_img = y_series.reshape((width, height)).T  # V2_1：目标采用相同排列
        # PyTorch 默认 channels-first：(C, H, W)
        x_img = np.expand_dims(x_img, axis=0)  # (1,96,7)
        y_img = np.expand_dims(y_img, axis=0)  # (1,96,7)
        X_list.append(x_img)
        Y_list.append(y_img)

    X_np = np.array(X_list)  # shape (week_number, 1, 96, 7)
    Y_np = np.array(Y_list)  # shape (week_number, 1, 96, 7)

    X_tensor = torch.tensor(X_np, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_np, dtype=torch.float32)

    # print("训练数据 X 的形状：", X_tensor.shape)  # 期望 (week_number,1,96,7)
    # print("标签数据 Y 的形状：", Y_tensor.shape)  # 期望 (week_number,1,96,7)

    # -------------------------------
    # 按 week_number 划分训练集和测试集（前80%作为训练集，后20%作为测试集）
    # -------------------------------
    train_index, test_index = train_test_index
    X_train = X_tensor[train_index]
    Y_train = Y_tensor[train_index]
    X_test = X_tensor[test_index]
    Y_test = Y_tensor[test_index]
    test_data = [X_test, Y_test]
    # print(f"训练集样本数：{X_train.shape[0]}, 测试集样本数：{X_test.shape[0]}")

    # -------------------------------

    lr_rate = train_params['lr_rate']
    num_epochs = train_params['num_epochs']
    patience = train_params['patience']

    # -------------------------------
    save_rng_state(log_dir, 'DWT_CNN')  # 只存随机状态，不重置/消耗随机数
    model = LSTMCNN(**model_params)

    # 设备自动选择（GPU or CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Using device for CNN:", device)

    # 暂停1s
    time.sleep(1)

    # -------------------------------
    # 训练模型
    # -------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)

    best_loss = float('inf')
    patience_counter = 0

    yellow_desc = "\033[33mCNN Processing Lower-Frequency Data\033[0m"
    bar_format = "\033[33m{l_bar}\033[33m{bar}\033[33m{r_bar}\033[0m"

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

    # torch.save(model.state_dict(), 'CNN_LSTM_model_state_dict.pth')

    # -------------------------------
    # 预测第一周数据并将预测结果展开为一列（672个数据）
    # -------------------------------
    finish_training_record(model, device)
    save_training_record(model, training_history, log_dir, 'DWT_CNN', model_params,
                         train_params, train_test_index, save_weights=save_weights)
    test_start = time_start(device)
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test.to(device))
        val_loss = criterion(test_outputs, Y_test.to(device))
    model.test_evaluation_seconds = time_end(test_start, device)
    print("Test Loss for Lower-Frequency:", val_loss.item())

    from matplotlib.ticker import MaxNLocator

    matplotlib.rcParams.update({
        "font.family": "Calibri",
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "legend.frameon": False,
        "legend.fontsize": 11,
        "figure.dpi": 300
    })

    fig_list = []
    for i in range(min(3, len(X_test))) if picture else []:
        pred_first = test_outputs[i, 0].cpu().numpy().T.reshape(-1)  # V2_1：每日布局还原
        orig_first = Y_test[i, 0].cpu().numpy().T.reshape(-1)

        result_df = pd.DataFrame({
            'Prediction': pred_first,
            'Original': orig_first
        }, index=indexs)

        fig, ax = plt.subplots(figsize=(9, 2.5))

        # Truth（灰色）
        ax.plot(
            result_df.index,
            result_df['Original'],
            color='#6E6E6E',
            linewidth=1.4,
            label='Truth'
        )

        # Prediction（红色虚线）
        ax.plot(
            result_df.index,
            result_df['Prediction'],
            color='#C0362C',
            linewidth=1.4,
            linestyle='--',
            label='Generated Lower-frequency Data'
        )

        ax.set_xlabel('Time step')
        ax.set_ylabel('Occupancy')

        # Occupancy一般是整数
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))

        step_per_day = 96
        num_days = 7
        ticks = np.arange(0, step_per_day * (num_days + 1), step_per_day)

        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)

        # 去掉上右边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['left'].set_linewidth(0.9)
        ax.spines['bottom'].set_linewidth(0.9)

        ax.grid(False)

        ax.legend(loc='upper right')

        plt.tight_layout()
        # plt.savefig(f'TimesNet CNN Week {i + 1}.png')
        # plt.close('all')

        fig_list.append(fig)

    return model, test_data, fig_list


def TimesNet_for_DWT(X, Y, train_test_index, name, model_params, train_params, log_dir=None, save_weights=False):
    # 训练配置
    batch_size = train_params['batch_size']
    learning_rate = train_params['lr_rate']
    num_epochs = train_params['num_epochs']
    patience_set = train_params['patience']
    min_delta = 1e-4

    # 设备配置：有 cuda 就使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == 'Max':
        print(f"Using device for TimesNet: {device}")

    config = SimpleNamespace(**model_params)

    save_rng_state(log_dir, 'TimesNet_' + name)
    model = Model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # 选择均方误差作为损失函数

    # 划分数据集（80%训练，20%测试）
    train_index, test_index = train_test_index
    train_dataset = TensorDataset(X[train_index], Y[train_index])
    test_dataset = TensorDataset(X[test_index], Y[test_index])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 训练模型
    if name == 'Max':
        purple_desc = "\033[35mTimesNet Processing Max-Frequency Data\033[0m"
        bar_format = "\033[35m{l_bar}\033[35m{bar}\033[35m{r_bar}\033[0m"
    elif name == 'Middle':
        purple_desc = "\033[36mTimesNet Processing Middle-Frequency Data\033[0m"
        bar_format = "\033[36m{l_bar}\033[36m{bar}\033[36m{r_bar}\033[0m"
    elif name == 'Direct':
        purple_desc = "\033[33mDirect TimesNet Processing\033[0m"
        bar_format = "\033[33m{l_bar}\033[33m{bar}\033[33m{r_bar}\033[0m"
    else:
        purple_desc = "\033[33mTimesNet Processing Higher-Frequency Data\033[0m"
        bar_format = "\033[33m{l_bar}\033[33m{bar}\033[33m{r_bar}\033[0m"

    best_loss = float('inf')
    no_improve_count = 0

    training_history = []
    start_training_record(model, device)
    for epoch in tqdm(range(num_epochs), desc=purple_desc, bar_format=bar_format):
        model.train()  # 设置模型为训练模式
        total_loss = 0.0

        # --------- 训练循环 ---------
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().item()

        # 计算一个 epoch 的平均训练损失
        avg_train_loss = total_loss / len(train_loader)
        training_history.append({'epoch': epoch + 1, 'train_mse': avg_train_loss})

        # --------- 早停逻辑 ---------
        # 如果 train_loss 较 best_loss 有明显改善，则更新
        if (best_loss - avg_train_loss) > min_delta:
            best_loss = avg_train_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 如果连续 no_improve_count >= patience，就停止
        if no_improve_count >= patience_set:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best Loss: {best_loss:.6f}")
            break

    # torch.save(model.state_dict(), f'TimesNet_MLP_model_state_dict_{name}.pth')

    finish_training_record(model, device)
    save_training_record(model, training_history, log_dir, 'TimesNet_' + name,
                         model_params, train_params, train_test_index, save_weights=save_weights)
    return model, test_loader


def timesnet_pred_generate(data_set, model_set, name_set, color_set, device, picture_set, log_dir=None):
    pred_list = []
    truth_list = []
    fig_list = []
    model_set.eval()
    prediction_seconds = 0.0

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_set):
            predict_start = time_start(device)
            # 数据迁移到 device
            x = x.to(device)
            y = y.to(device)
            outputs = model_set(x)
            # 收集整批；batch_size=1时与V2_1相同，较大batch不再丢样本。
            batch_pred = outputs.detach().cpu().numpy().reshape(len(outputs), -1)
            batch_truth = y.detach().cpu().numpy().reshape(len(y), -1)
            pred_list.extend(batch_pred)
            truth_list.extend(batch_truth)
            timesnet_pred = batch_pred[0]
            timesnet_truth = batch_truth[0]
            prediction_seconds += time_end(predict_start, device)

            # 如果是第一个批次，绘制对比曲线（后续批次只用于拼接结果，不再绘图）
            if picture_set:
                # 最多画前三个样本，不随batch大小改变合图数量；图内样式不变。
                remaining = max(0, 3 - (len(pred_list) - len(batch_pred)))
                for sample_index in range(min(len(batch_pred), remaining)):
                    timesnet_pred = batch_pred[sample_index]
                    timesnet_truth = batch_truth[sample_index]
                    from matplotlib.ticker import MaxNLocator

                    matplotlib.rcParams.update({
                        "font.family": "Calibri",
                        "font.size": 12,
                        "axes.labelsize": 13,
                        "axes.linewidth": 0.9,
                        "xtick.labelsize": 11,
                        "ytick.labelsize": 11,
                        "xtick.direction": "out",
                        "ytick.direction": "out",
                        "xtick.major.size": 3.5,
                        "ytick.major.size": 3.5,
                        "legend.frameon": False,
                        "legend.fontsize": 11,
                        "figure.dpi": 300
                    })

                    fig, ax = plt.subplots(figsize=(9, 2.5))

                    # Truth（灰色实线）
                    ax.plot(
                        timesnet_truth,
                        color='#6E6E6E',
                        linewidth=1.4,
                        label='Truth'
                    )

                    # Prediction（红色虚线）
                    ax.plot(
                        timesnet_pred,
                        color='#C0362C',
                        linewidth=1.4,
                        linestyle='--',
                        label=f'Generated {name_set}-frequency Data'
                    )

                    ax.set_xlabel('Time step')
                    ax.set_ylabel('Occupancy')

                    # y轴整数刻度，最多5个
                    ax.set_ylim(-4.5, 4.5)
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

                    # 固定 x 轴为 0, 96, 192 ...
                    n = len(timesnet_truth)
                    step_per_day = 96
                    ticks = np.arange(0, n + 1, step_per_day)
                    ax.set_xticks(ticks)

                    # 去掉上右边框
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    ax.spines['left'].set_linewidth(0.9)
                    ax.spines['bottom'].set_linewidth(0.9)

                    ax.grid(False)

                    ax.legend(loc='upper right')

                    plt.tight_layout()
                    # plt.savefig(f'TimesNet {name_set}{batch_idx}.png')
                    # plt.close('all')

                    fig_list.append(fig)

    # 拼接所有 batch 的数据
    TimesNet_pred = np.concatenate(pred_list)
    TimesNet_truth = np.concatenate(truth_list)

    model_set.inference_seconds = prediction_seconds
    group = 'Direct TimesNet' if name_set == 'Direct' else 'DWT-TimesNet'
    save_model_runtime(model_set, log_dir, 'TimesNet_' + name_set, group, len(TimesNet_pred) // 672)
    return TimesNet_pred, TimesNet_truth, fig_list


def merge_fig_lists(figMax_list, fig_middle_list, figCNN_list, save_prefix="Merged_Week"):
    assert len(figMax_list) == len(fig_middle_list) == len(figCNN_list), \
        "三个列表长度必须一致"

    merged_fig_list = []

    for i in range(len(figMax_list)):
        # 创建一个新的图，用来拼接三个图
        final_fig = plt.figure(figsize=(9, 7.5), dpi=300)

        # 创建 GridSpec 来安排子图的位置
        gs = GridSpec(3, 1, figure=final_fig)

        # 获取每个 fig 中的 axes（只需取第一个，因为每个 fig 只有一个 ax）
        ax_max = figMax_list[i].axes[0]
        ax_middle = fig_middle_list[i].axes[0]
        ax_cnn = figCNN_list[i].axes[0]

        # 将每个 ax 复制到最终图的大图中
        ax_final_1 = final_fig.add_subplot(gs[0, 0])  # 第一行
        ax_final_2 = final_fig.add_subplot(gs[1, 0])  # 第二行
        ax_final_3 = final_fig.add_subplot(gs[2, 0])  # 第三行

        # 复制 Max-frequency 图的内容
        for line in ax_max.get_lines():
            ax_final_1.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), label=line.get_label(), linestyle=line.get_linestyle())

        # 复制 Middle-frequency 图的内容
        for line in ax_middle.get_lines():
            ax_final_2.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), label=line.get_label(), linestyle=line.get_linestyle())

        # 复制 CNN 图的内容
        for line in ax_cnn.get_lines():
            ax_final_3.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), label=line.get_label(), linestyle=line.get_linestyle())

        # 设置每个子图的轴标签
        ax_final_1.set_ylabel('Occupancy', fontsize=14)
        ax_final_2.set_ylabel('Occupancy', fontsize=14)
        ax_final_3.set_ylabel('Occupancy', fontsize=14)
        ax_final_3.set_xlabel('Time Step', fontsize=13)

        # 保持原始的 ticks 格式
        ax_final_1.set_xticks(ax_max.get_xticks())
        ax_final_1.set_yticks(ax_max.get_yticks())
        ax_final_2.set_xticks(ax_middle.get_xticks())
        ax_final_2.set_yticks(ax_middle.get_yticks())
        ax_final_3.set_xticks(ax_cnn.get_xticks())
        ax_final_3.set_yticks(ax_cnn.get_yticks())

        # 设置 xticklabels 保持一致
        ax_final_1.set_xticklabels(ax_max.get_xticklabels())
        ax_final_2.set_xticklabels(ax_middle.get_xticklabels())
        ax_final_3.set_xticklabels(ax_cnn.get_xticklabels())

        # 保持 ylim 格式
        ax_final_1.set_ylim(ax_max.get_ylim())
        ax_final_2.set_ylim(ax_middle.get_ylim())
        ax_final_3.set_ylim(ax_cnn.get_ylim())

        # 设置 legend（保持图例）
        ax_final_1.legend(loc='upper right', fontsize=11)
        ax_final_2.legend(loc='upper right', fontsize=11)
        ax_final_3.legend(loc='upper right', fontsize=11)

        # 调整每个子图的格式（去掉轴线、网格等）
        for ax in [ax_final_1, ax_final_2, ax_final_3]:
            ax.tick_params(axis='both', labelsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)

        # 调整布局，避免裁切
        plt.tight_layout()

        # 保存拼接后的图像
        plt.savefig(f"{save_prefix}_{i + 1}.png", dpi=300)
        plt.close(final_fig)

        # 保存拼接后的图像对象
        merged_fig_list.append(final_fig)

    return merged_fig_list


def DWT_data_generating(model_CNN, data_CNN, model_TimesNet1, data_TimesNet1, model_TimesNet2, data_TimesNet2, test_index, wavelet, DWT_set, fig_CNN_list, picture, log_dir=None, component_dict=None):
    matplotlib.rcParams['font.family'] = 'Calibri'

    # 设备配置：有 cuda 就使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device for test data generating: {device}")
    criterion = nn.MSELoss()

    # 先获取前n-1本征量之和
    X_test_C, Y_test_C = data_CNN

    predict_start = time_start(device)
    model_CNN.eval()
    with torch.no_grad():
        test_outputs_C = model_CNN(X_test_C.to(device))
        val_loss = criterion(test_outputs_C, Y_test_C.to(device))

    Non_frequency_pred = test_outputs_C.cpu().numpy()[:, 0, :, :].transpose(0, 2, 1).reshape(-1)  # V2_1：按真实时间顺序还原
    # Non_frequency_truth = Y_test_C.cpu().numpy()[:, 0, :, :].transpose(0, 2, 1).reshape(-1)

    model_CNN.inference_seconds = time_end(predict_start, device)
    save_model_runtime(model_CNN, log_dir, 'DWT_CNN', 'DWT-TimesNet', len(test_index) // 672)

    TimesNet_pred_max, TimesNet_truth_max, fig_max_list = timesnet_pred_generate(data_TimesNet1, model_TimesNet1, 'Max', 'Blue', device, picture, log_dir=log_dir)
    TimesNet_pred_middle, TimesNet_truth_middle, fig_middle_list = timesnet_pred_generate(data_TimesNet2, model_TimesNet2, 'Middle', 'Green', device, picture, log_dir=log_dir)
    if picture:
        merge_fig_lists(fig_max_list, fig_middle_list, fig_CNN_list, save_prefix="TimesNet_Week")
        for fig in fig_max_list + fig_middle_list + fig_CNN_list:
            plt.close(fig)

    if component_dict is not None:
        component_dict.update({
            'Low': Non_frequency_pred, 'D1': TimesNet_pred_max, 'D2': TimesNet_pred_middle,
            'Low_truth': Y_test_C.cpu().numpy()[:, 0, :, :].transpose(0, 2, 1).reshape(-1),
            'D1_truth': TimesNet_truth_max, 'D2_truth': TimesNet_truth_middle,
        })

    if DWT_set:
        df = pd.DataFrame({
            f"DWT-TimesNet-{wavelet}": Non_frequency_pred + TimesNet_pred_max + TimesNet_pred_middle
        }, index=test_index)
    else:
        df = pd.DataFrame({
            f"DWT-TimesNet": Non_frequency_pred + TimesNet_pred_max + TimesNet_pred_middle
        }, index=test_index)

    return df
