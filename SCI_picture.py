import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from function_package import read_data


def save_split_line_plots(y, split_list, out_dir="TimesNet_split_figures"):
    """
    把同一条序列按 split_list 中指定的份数分别切分并输出图片。
    命名规则：
        1-1.png  -> 分 1 份时的第 1 张
        7-2.png  -> 分 7 份时的第 2 张
        3-1.png  -> 分 3 份时的第 1 张
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_len = len(y)

    for n_parts in split_list:
        part_len = math.ceil(total_len / n_parts)

        for i in range(n_parts):
            start = i * part_len
            end = min((i + 1) * part_len, total_len)
            y_part = y[start:end]

            fig, ax = plt.subplots(figsize=(7 / n_parts, 1), dpi=300)
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")

            ax.plot(
                range(len(y_part)),
                y_part,
                color="#6A6A6A",
                linewidth=2.0,
                solid_capstyle="round",
                solid_joinstyle="round"
            )

            ax.axis("off")
            ax.margins(x=0, y=0.08)
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            save_path = out_dir / f"{n_parts}-{i+1}.png"
            plt.savefig(
                save_path,
                dpi=300,
                transparent=True,
                bbox_inches="tight",
                pad_inches=0
            )
            plt.close()


def save_black_week_and_fft(y, out_dir="TimesNet_split_figures"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 黑色一周原始数据图
    fig, ax = plt.subplots(figsize=(7, 1), dpi=300)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.plot(
        range(len(y)),
        y,
        color="black",
        linewidth=2.0,
        solid_capstyle="round",
        solid_joinstyle="round"
    )

    ax.axis("off")
    ax.margins(x=0, y=0.08)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(
        out_dir / "week_black.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()

    # 2) 更接近 TimesNet 的 FFT 图
    y_norm = y - np.mean(y)

    std = np.std(y_norm)
    if std > 1e-8:
        y_norm = y_norm / std

    fft_vals = np.fft.rfft(y_norm)
    fft_amp = np.abs(fft_vals)

    # 对应 TimesNet 里的 frequency_list[0] = 0
    fft_amp[0] = 0

    freq = np.arange(len(fft_amp))

    fig, ax = plt.subplots(figsize=(7, 1), dpi=300)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.plot(
        freq[1:],
        fft_amp[1:],
        color="black",
        linewidth=2.0,
        solid_capstyle="round",
        solid_joinstyle="round"
    )

    ax.axis("off")
    ax.margins(x=0, y=0.08)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(
        out_dir / "week_fft_black.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0
    )
    plt.close()


def TimesNet_SCI_picture():
    high_occupancy_list, low_occupancy_list, week_number = read_data(
        "occupancy_office_month.xlsx",
        ignore_cols=3
    )

    # 这里用 [1]
    df = high_occupancy_list[1].copy()

    if isinstance(df, pd.DataFrame):
        y = df.iloc[:, 0].to_numpy(dtype=float)
    else:
        y = pd.Series(df).to_numpy(dtype=float)

    out_dir = "TimesNet_split_figures"

    # 原来的灰色分割图
    save_split_line_plots(
        y=y,
        split_list=[1, 2, 3, 7],
        out_dir=out_dir
    )

    # 新增：黑色一周原始图 + 黑色 FFT 图
    save_black_week_and_fft(
        y=y,
        out_dir=out_dir
    )

    return


if __name__ == "__main__":
    TimesNet_SCI_picture()