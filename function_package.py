import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import torch
import pywt
import matplotlib
import time
from itertools import combinations


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 禁止非确定性算法
    torch.backends.cudnn.benchmark = False


def custom_round(x):
    # 得到下界
    lower = math.floor(x)
    # 得到小数部分
    frac = x - lower
    # 如果小数部分正好为0.5（考虑浮点误差）
    if abs(frac - 0.5) < 1e-9:
        # 随机选取向下或向上
        return lower if random.random() < 0.5 else lower + 1
    else:
        return round(x)


def generate_sheet_combinations(n_sheets=5, num=1, start_index=0):
    sheet_ids = list(range(start_index, start_index + n_sheets))
    comb_list = [list(c) for c in combinations(sheet_ids, num)]
    return comb_list


def create_train_test_split_indices(total_samples, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(total_samples)
    test_sample_size = int(total_samples * test_size)
    # 随机选取不重复的测试集索引
    test_indices = np.random.choice(indices, size=test_sample_size, replace=False)
    # 训练集索引为剩余部分
    train_indices = np.setdiff1d(indices, test_indices)

    return train_indices, test_indices


def create_train_test_split_indices_2(high_list1, total_list):
    m = len(high_list1)
    n = len(total_list)

    idxA = np.arange(m)  # array([0, 1, 2, ..., m-1])
    idxRest = np.arange(m, n)  # 剩下的：array([m, m+1, ..., n-1])

    return idxA, idxRest

def create_train_test_indices_by_sheet(train_sheet, week_number_list):
    train_sheet = set(train_sheet)

    train_idx = []
    test_idx = []

    start = 0
    for sheet_id, n_week in enumerate(week_number_list):
        sheet_indices = list(range(start, start + n_week))

        if sheet_id in train_sheet:
            train_idx.extend(sheet_indices)
        else:
            test_idx.extend(sheet_indices)

        start += n_week

    return np.array(train_idx), np.array(test_idx)


def split_weeks(df, factor):
    # 每周的行数
    week_length = 24 * factor * 7
    total_rows = len(df)
    # 计算完整的周数（如果整除不完，则舍弃剩余部分）
    num_weeks = total_rows // week_length
    week_dfs = [df.iloc[i*week_length:(i+1)*week_length].copy() for i in range(num_weeks)]
    return week_dfs


def read_data(flie_name, ignore_cols=0):
    raw_data = pd.read_excel(flie_name, index_col=[0])

    if ignore_cols > 0:
        raw_data = raw_data.iloc[:, ignore_cols:]

    columns = raw_data.columns.tolist()
    week_number = len(columns) // 7
    flatten_data = raw_data.to_numpy().ravel(order='F')

    new_index = []
    for col in columns:
        for time in raw_data.index:
            new_index.append(f"{col} {time}")

    new_df = pd.DataFrame(flatten_data, index=new_index, columns=['Occupancy'])

    new_df_subset = new_df.iloc[3::4]

    high_df_list = split_weeks(new_df, 4)
    low_df_list = split_weeks(new_df_subset, 1)

    return high_df_list, low_df_list, week_number


def read_data_multi_sheets(file_name, ignore_cols=0, sheet_names=None):
    xls = pd.ExcelFile(file_name)

    if sheet_names is None:
        sheet_names = xls.sheet_names  # 默认所有 sheet
    else:
        missing = [s for s in sheet_names if s not in xls.sheet_names]
        if missing:
            raise ValueError(f"这些 sheet 不存在: {missing}. 可用 sheet: {xls.sheet_names}")

    high_all = []
    low_all = []
    week_all = []

    for sname in sheet_names:
        raw_data = pd.read_excel(file_name, sheet_name=sname, index_col=[0])

        if ignore_cols > 0:
            raw_data = raw_data.iloc[:, ignore_cols:]

        columns = raw_data.columns.tolist()
        week_number = len(columns) // 7
        flatten_data = raw_data.to_numpy().ravel(order='F')

        new_index = []
        for col in columns:
            for time in raw_data.index:
                new_index.append(f"{col} {time}")

        new_df = pd.DataFrame(flatten_data, index=new_index, columns=['Occupancy'])
        new_df_subset = new_df.iloc[3::4]

        # 这里沿用你原来的拆周逻辑
        high_df_list = split_weeks(new_df, 4)
        low_df_list = split_weeks(new_df_subset, 1)

        # 关键：按 sheet 维度汇总
        high_all.extend(high_df_list)
        low_all.extend(low_df_list)
        week_all.append(week_number)

    return high_all, low_all, week_all


def index_generating(high_list, train_test_index):
    train_index, test_index = train_test_index
    test_index_name_list = []
    for test_number in test_index:
        test_index_name_list.extend(high_list[test_number].index.tolist())

    return test_index_name_list


def result_template_generate(high_list, train_test_index, test_index_name):
    train_index, test_index = train_test_index
    test_truth_list = []
    for test_number in test_index:
        test_truth_list.extend(high_list[test_number].squeeze())

    df = pd.DataFrame({
        'Truth': test_truth_list
    }, index=test_index_name)

    return df


def DWT_week(df_file, wavelet='haar', level=4, week=-1, name='Low', picture=True):
    """
    wavelet : str
        选择的小波基名称，如 'db4', 'haar', 'coif1', 'sym5' 等。
    """

    matplotlib.rcParams['font.family'] = 'Calibri'
    # 1) 读取原信号
    signal = df_file['Occupancy'].values

    # 2) 使用 pywt.wavedec 进行 DWT 分解
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)

    # 3) 对每一级系数做单独重构，得到对应尺度的子信号
    sub_signals = []
    for i in range(len(coeffs)):
        # 构造一个与 coeffs 同长度的列表，但除了第 i 个，其余都置为 zeros_like
        # 这样 inverse transform 时只保留第 i 个分量
        temp_coeffs = [np.zeros_like(c) for c in coeffs]
        temp_coeffs[i] = coeffs[i]  # 仅保留第 i 个分量
        # 还原得到时域子信号
        sub_signal = pywt.waverec(temp_coeffs, wavelet=wavelet)
        # 由于小波变换可能导致还原后的长度稍微 > 原始信号长度，需要切片对齐
        sub_signal = sub_signal[:len(signal)]
        sub_signals.append(sub_signal)

    num_levels = len(coeffs)  # 等于 level+1

    # 4) 把每一个子信号添加到 df_file 中，便于后续建模或可视化
    col_names = []
    col_names.append(f"A{level}")  # Approximation
    for d_lev in range(level, 0, -1):
        col_names.append(f"D{d_lev}")

    # 仅用于画图显示的标签
    display_names = []
    if name == 'High':
        display_names.append(r"$A_h$")
        for d_lev in range(level, 0, -1):
            display_names.append(rf"$D_{{h,{d_lev}}}$")
    else:
        display_names.append(r"$A_l$")
        for d_lev in range(level, 0, -1):
            display_names.append(rf"$D_{{l,{d_lev}}}$")

    df_temp = pd.DataFrame(
        data=np.column_stack(sub_signals),
        columns=col_names,
        index=df_file.index
    )

    # 合并回原 df
    df_file = pd.concat([df_file, df_temp], axis=1)

    # 5) 选择性绘图 —— 仅在 week <= 0 时绘制
    if picture:
        if week <= 0:

            matplotlib.rcParams.update({
                "font.family": "Calibri",
                "font.size": 13,
                "axes.labelsize": 14,
                "axes.linewidth": 0.9,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "xtick.direction": "out",
                "ytick.direction": "out",
                "xtick.major.size": 3.5,
                "ytick.major.size": 3.5,
                "figure.dpi": 300
            })

            fig, axes = plt.subplots(
                num_levels, 1,
                figsize=(9, 1.2 * num_levels),
                sharex=True
            )

            if num_levels == 1:
                axes = [axes]

            for i in range(num_levels):
                ax = axes[i]

                ax.plot(
                    sub_signals[i],
                    color="#C0362C",
                    linewidth=1.2
                )

                # 左侧标签（替代原 title）
                ax.set_ylabel(
                    display_names[i],
                    fontsize=14,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=2  # 更靠近y轴
                )

                # 保留y轴刻度
                ax.tick_params(axis='y', which='major', pad=6)

                from matplotlib.ticker import MaxNLocator
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

                # 去掉上右边框
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                ax.spines["left"].set_linewidth(0.9)
                ax.spines["bottom"].set_linewidth(0.9)

                ax.grid(False)

            # 只在最底部显示x轴标签
            n = len(sub_signals[0])
            if name == 'High':
                axes[-1].set_xlabel("Time step (15min)", fontsize=13)
                step_per_day = 96
            else:
                axes[-1].set_xlabel("Time step (1h)", fontsize=13)
                step_per_day = 24
            ticks = np.arange(0, n + 1, step_per_day)
            axes[-1].set_xticks(ticks)

            plt.tight_layout()
            plt.savefig(f'DWT_results_for_{name}_occupancy_data.png')
            plt.close('all')

    return num_levels, df_file


def DWT_process(low_occupancy_list, high_occupancy_list, week_number,
                wavelet='haar', level=2, picture=True):
    """
    类似 CEEMDAN_process，对多周的 Low/High occupancy 数据分别做小波分解。
    wavelet_list = [
    "haar",     # 等同于 db1，最短滤波器，对突变最敏感
    "db2",      # Daubechies 2，简易平滑，保留边缘信息
    "db4",      # Daubechies 4，常见通用小波，平滑度适中
    "db8",      # Daubechies 8，更长滤波器，更平滑
    "sym4",     # Symlet 4，近似对称，减少相移失真
    "coif1",    # Coiflet 入门，较高平滑度
    "coif5",    # Coiflet 最高阶之一，更佳平滑与正则
    "bior1.3",  # 双正交小波，分析/重构滤波器可不同
    "bior4.4",  # 常见双正交，用于图像/信号压缩与去噪
    "rbio3.7"   # 反向双正交，与 bior 互补，可对比相位特性
    ]
    """
    low_levels_list = []
    high_levels_list = []
    low_dwt_file_list = []
    high_dwt_file_list = []

    # 暂停1s
    time.sleep(1)

    blue_desc = "\033[34mDWT Processing Weeks\033[0m"
    bar_format = "\033[34m{l_bar}\033[34m{bar}\033[34m{r_bar}\033[0m"

    for week in tqdm(range(week_number), desc=blue_desc, bar_format=bar_format):
        # 对低时频 occupancy 做 DWT
        low_levels, low_file = DWT_week(
            df_file=low_occupancy_list[week],
            wavelet=wavelet,
            level=level,
            week=week,
            name='Low',
            picture=picture
        )
        # 对高时频 occupancy 做 DWT
        high_levels, high_file = DWT_week(
            df_file=high_occupancy_list[week],
            wavelet=wavelet,
            level=low_levels + 1,
            week=week,
            name='High',
            picture=picture
        )

        low_levels_list.append(low_levels)
        high_levels_list.append(high_levels)
        low_dwt_file_list.append(low_file)
        high_dwt_file_list.append(high_file)

    return low_levels_list, high_levels_list, low_dwt_file_list, high_dwt_file_list


def interpolate_occupancy_linear(low_file):
    # 取出原始的 Occupancy 数据和数据个数
    occ = low_file['Occupancy'].values
    N = len(occ)

    # 原始时间：0,1,...,N-1
    t = np.arange(N)
    # 在前面增加一个 t = -1 的点，其值与第一个点相同
    t_aug = np.concatenate(([-1], t))
    y_aug = np.concatenate(([occ[0]], occ))

    # 在 [-1, N-1] 区间上产生 4N+1 个等距点
    t_new = np.linspace(-1, N - 1, 4 * N + 1)

    # 用线性插值得到插值序列
    y_new_all = np.interp(t_new, t_aug, y_aug)

    # 跳过第一个点 (对应 t=-1 的插值结果), 保留后面 4N 个点
    y_new = y_new_all[1:]  # shape = (4N, )

    return y_new


def interpolate_occupancy_spline(low_file):
    # 取出原始的 Occupancy 数据和数据个数
    occ = low_file['Occupancy'].values
    N = len(occ)

    # 原始时间：0,1,...,N-1
    t = np.arange(N)
    # 在前面增加一个 t = -1 的点，其值与第一个点相同
    t_aug = np.concatenate(([-1], t))
    y_aug = np.concatenate(([occ[0]], occ))

    # 构造三次样条插值函数
    cs = CubicSpline(t_aug, y_aug)

    # 在区间 [0, N-1] 内均匀生成 4N 个点
    t_new = np.linspace(-1, N - 1, 4 * N + 1)
    y_new = cs(t_new)[1:]

    return y_new


def interpolate_occupancy_zero(low_file):
    # 取出原始的 Occupancy 数据和数据个数
    occ = low_file['Occupancy'].values
    N = len(occ)
    up_factor = 4
    new_length = up_factor * N
    y_new = np.empty(new_length)

    for i in range(N):
        y_new[up_factor * i: up_factor * (i + 1)] = occ[i]

    return y_new


def plot_fft_spectrum(x, week_index, name):
    # x.shape -> [1, T, 1]
    # rfft 之后 -> [1, T//2+1, 1]
    xf = torch.fft.rfft(x, dim=1)  # 沿时间维度 T 做 FFT

    # 取幅度，并去掉 batch 和通道维度 => [T//2+1]
    amplitude = xf.abs().squeeze().cpu().numpy()  # shape: (T//2+1,)

    # 构建一个频率轴，这里仅用下标 0~(T//2)
    T = x.shape[1]  # 时间序列长度
    freq_bins = range(T//2 + 1)

    # 绘图
    plt.figure()
    plt.plot(freq_bins, amplitude)
    plt.title(f"Week {week_index+1} - FFT Amplitude Spectrum in {name}")
    plt.xlabel("Frequency Bin")
    plt.ylabel("Amplitude")
    plt.savefig(f"Week {week_index+1} - FFT Amplitude Spectrum in {name}.png")
    plt.close('all')

    return


def linear_interpolate_DWT(low_file, low_number):
    occ = low_file['IMF{}'.format(low_number)].values
    N = len(occ)

    # 原序列时间索引: 0, 1, 2, ..., N-1
    t = np.arange(N)

    # 在开头加一个点: t=-1，其值与 occ[0] 相同
    t_aug = np.concatenate(([-1], t))  # [-1, 0, 1, 2, ..., N-1]
    y_aug = np.concatenate(([occ[0]], occ))  # [occ[0], occ[0], occ[1], ..., occ[N-1]]

    # 在 [-1, N-1] 区间上产生 4N+1 个等距点
    t_new = np.linspace(-1, N - 1, 4 * N + 1)

    # 用线性插值得到插值序列
    y_new_all = np.interp(t_new, t_aug, y_aug)

    # 跳过第一个点 (对应 t=-1 的插值结果), 保留后面 4N 个点
    y_new = y_new_all[1:]  # shape = (4N, )

    return y_new


def zero_interpolate_DWT(low_file, low_number):
    occ = low_file['D1'].values  # V2_1：明确读取小时D1，low_number仍是原分量总数
    N = len(occ)

    # 构造一个长度为 4N 的数组，默认全为 0
    y_new = np.zeros(4 * N, dtype=float)

    # 将每个原始数据点放到 y_new 对应位置 (步长 4)
    for i in range(N):
        y_new[4 * i] = occ[i]

    return y_new


def high_interpolate_DWT(low_file, low_number):
    occ = low_file['D1'].values  # V2_1：明确读取小时D1，low_number仍是原分量总数
    N = len(occ)

    # 构造一个长度为 4N 的数组，默认全为 0
    f1 = np.zeros(4 * N, dtype=float)

    # 填充 f1 数组，步长为 4
    for i in range(N):
        f1[4 * i] = -occ[i]
        f1[4 * i + 1] = occ[i]
        f1[4 * i + 2] = -occ[i]
        f1[4 * i + 3] = occ[i]

    return f1


def middle_interpolate_DWT(low_file, low_number):
    occ = low_file['D1'].values  # V2_1：明确读取小时D1，low_number仍是原分量总数
    N = len(occ)

    # 构造一个长度为 4N 的数组，默认全为 0
    f2 = np.zeros(4 * N, dtype=float)

    # 填充 f2 数组，步长为 4
    for i in range(N):
        f2[4 * i] = -occ[i]
        f2[4 * i + 1] = -occ[i]
        f2[4 * i + 2] = occ[i]
        f2[4 * i + 3] = occ[i]

    return f2

def DWT_two_frequency_method_process(low_number_list, low_file_list, high_file_list, week_number, picture):
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

    # 构建一个list储存数据
    dwt_tensor_max_list = []
    dwt_tensor_middle_list = []
    max_frequency_list = []
    middle_frequency_list = []

    for week in range(week_number):
        low_file = low_file_list[week]
        low_number = low_number_list[week]

        max_linear_frequency = zero_interpolate_DWT(low_file, low_number)
        max_order_frequency = high_interpolate_DWT(low_file, low_number)
        middle_order_frequency = middle_interpolate_DWT(low_file, low_number)

        high_file = high_file_list[week]
        high_file.index = indexs
        # 先提取high_file中最高本征频率的本征量
        imf_columns = [col for col in high_file.columns if col.startswith("D")]

        top_det_col = imf_columns[-1]
        middle_det_col = imf_columns[-2]
        # if week == 10:
        #     print(top_det_col)
        #     print(high_file[top_det_col].values)

        dwt_array_max = high_file[top_det_col].values
        dwt_array_middle = high_file[middle_det_col].values

        # 转成 PyTorch tensor，并调整形状为 [B, T, C] = [1, 672, 1]
        dwt_tensor_max = torch.tensor(dwt_array_max, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        dwt_tensor_middle = torch.tensor(dwt_array_middle, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        max_frequency_tensor = torch.tensor(max_order_frequency, dtype=torch.float32).view(1, -1, 1)
        middle_frequency_tensor = torch.tensor(middle_order_frequency, dtype=torch.float32).view(1, -1, 1)

        # 做 FFT 并画图
        if picture:
            if week <= 0:
                plot_fft_spectrum(dwt_tensor_max, week_index=week, name='High-frequency')
                plot_fft_spectrum(dwt_tensor_middle, week_index=week, name='Middle-frequency')
                plot_fft_spectrum(max_frequency_tensor, week_index=week, name='Zero-frequency-max')
                plot_fft_spectrum(middle_frequency_tensor, week_index=week, name='Zero-frequency-middle')

        # 收集
        dwt_tensor_max_list.append(dwt_tensor_max)
        dwt_tensor_middle_list.append(dwt_tensor_middle)
        max_frequency_list.append(max_frequency_tensor)
        middle_frequency_list.append(middle_frequency_tensor)

    dwt_tensor_max_all = torch.cat(dwt_tensor_max_list, dim=0)
    dwt_tensor_middle_all = torch.cat(dwt_tensor_middle_list, dim=0)
    max_frequency_all = torch.cat(max_frequency_list, dim=0)
    middle_frequency_all = torch.cat(middle_frequency_list, dim=0)

    return max_frequency_all, middle_frequency_all, dwt_tensor_max_all, dwt_tensor_middle_all


def zero_interpolate_IMF(low_file, low_number):
    # 取出指定 IMF 列的数据
    occ = low_file[f'IMF{low_number}'].values
    N = len(occ)

    # 构造一个长度为 4N 的数组，默认全为 0
    y_new = np.zeros(4 * N, dtype=float)

    # 将每个原始数据点放到 y_new 对应位置 (步长 4)
    for i in range(N):
        y_new[4 * i] = occ[i]

    return y_new


def FFT_method_process(low_number_list, low_file_list, high_file_list, week_number):
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

    # 构建一个list储存数据
    imf_tensor_list = []
    max_frequency_list = []

    for week in range(week_number):
        low_file = low_file_list[week]
        low_number = low_number_list[week]

        max_linear_frequency = zero_interpolate_IMF(low_file, low_number)

        high_file = high_file_list[week]
        high_file.index = indexs
        # 先提取high_file中最高本征频率的本征量
        imf_columns = [col for col in high_file.columns if col.startswith("IMF")]
        high_imf_col = imf_columns[-1]
        imf_array = high_file[high_imf_col].values

        # 转成 PyTorch tensor，并调整形状为 [B, T, C] = [1, 672, 1]
        imf_tensor = torch.tensor(imf_array, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        max_linear_frequency_tensor = torch.tensor(max_linear_frequency, dtype=torch.float32).view(1, -1, 1)

        # # 做 FFT 并画图
        # if week <= 0:
        #     plot_fft_spectrum(imf_tensor, week_index=week, name='High-frequency')
        #     plot_fft_spectrum(max_linear_frequency_tensor, week_index=week, name='Linear-frequency')

        # 收集
        imf_tensor_list.append(imf_tensor)
        max_frequency_list.append(max_linear_frequency_tensor)

    imf_tensor_all = torch.cat(imf_tensor_list, dim=0)
    max_frequency_all = torch.cat(max_frequency_list, dim=0)

    return max_frequency_all, imf_tensor_all


def backward_process(result_df):
    new_row = pd.DataFrame({col: [0] for col in result_df.columns})
    new_df = pd.concat([new_row, result_df], ignore_index=True)

    new_df['Diff'] = new_df['Generate'] - new_df['Truth']
    new_df['pred_new'] = new_df['Generate']
    new_df['pred_new'] = new_df['pred_new'].astype(float)

    total_length = len(new_df)
    for i in range(0, total_length, 4):
        diff_occ = new_df.loc[i, 'Diff']
        if 0 < i < total_length - 1:
            for j in range(-3, 4):
                new_df.loc[i + j, 'pred_new'] += - (1 - np.abs(j) / 4) * diff_occ
        if i == 0:
            for j in range(0, 4):
                new_df.loc[i + j, 'pred_new'] += - (1 - np.abs(j) / 4) * diff_occ
        if i == total_length - 1:
            for j in range(-3, 1):
                new_df.loc[i + j, 'pred_new'] += - (1 - np.abs(j) / 4) * diff_occ

    new_df = new_df.iloc[1:, :].copy()
    new_df.index = result_df.index
    new_df['pred_new'] = new_df['pred_new'].round(0)
    new_df.loc[new_df['pred_new'] <= 0, 'pred_new'] = 0
    new_df = new_df[['pred_new']]

    return new_df
