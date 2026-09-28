import pandas as pd
from pathlib import Path


def WIFI_data_processing():
    INPUT_XLSX = r'./WIFI_origin_data.xlsx'
    DATETIME_COL = 0  # 时间戳列（若你的文件是列名，就改成字符串）
    LEVEL_COLS = ["Level2", "Level3", "Level4", "Level5", "Level6"]
    OUTPUT_XLSX = 'WIFI_raw_data.xlsx'
    OUTPUT_XLSX1 = 'WIFI_month_data.xlsx'

    df = pd.read_excel(INPUT_XLSX, parse_dates=[DATETIME_COL])
    dtcol = df.columns[DATETIME_COL] if isinstance(DATETIME_COL, int) else DATETIME_COL

    # 把时间戳设为索引（你要求“读取index”）
    df = df.set_index(dtcol)

    # ===== (2) 生成日期列表（去年份，按顺序），打印出来给你检查 =====
    # 先按出现顺序去重（不打乱顺序）
    unique_days = pd.DatetimeIndex(pd.unique(df.index.normalize()))
    # 依据数据行数，按 96 条/天 计算有效天数（如果最后不满 96 条，直接忽略）
    n_days = len(df) // 96
    unique_days = unique_days[:n_days]

    # 去掉年份 -> 'Jan8' 格式（英文月缩写 + 日）
    date_labels = [f"{d.strftime('%b')}{d.day}" for d in unique_days]

    # ===== 行索引：0:15, 0:30, ..., 23:45, 24:00 =====
    times = pd.date_range("00:15", "23:45", freq="15min").strftime("%H:%M").tolist()
    times = [t[1:] if t.startswith("0") else t for t in times] + ["24:00"]

    # 注：我们按你的要求，直接把每 96 条（00:00~23:45）对应到 0:15~24:00 的96个行标签

    # ===== (3) 逐列拆分 & 写入每个 Level 的 sheet =====
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        for col in LEVEL_COLS:
            if col not in df.columns:
                continue  # 某列不存在就跳过

            # 取该列所有值，并只保留完整天数部分（前 n_days*96 条）
            vals = df[col].to_numpy()
            total_needed = n_days * 96
            vals = vals[:total_needed]

            # 按天切块：每 96 条为一组 -> 形状 (n_days, 96)，再转置成 (96, n_days)
            # 这样就对应 行=96个时段、列=n_days 天
            mat = vals.reshape(n_days, 96).T

            col_labels = [f"{d}-L{col[-1]}" for d in date_labels]

            # 组装成 DataFrame
            wide = pd.DataFrame(mat, index=times, columns=col_labels)
            wide.index.name = "Time"

            # 写入一个 sheet（sheet 名就是 LevelX）
            wide.to_excel(writer, sheet_name=col)

    print("✅ Done:", OUTPUT_XLSX)


    with pd.ExcelWriter(OUTPUT_XLSX1, engine="openpyxl") as writer:
        for col in LEVEL_COLS:
            if col not in df.columns:
                continue  # 某列不存在就跳过

            # 取该列所有值，并只保留完整天数部分（前 n_days*96 条）
            vals = df[col].to_numpy()
            total_needed = n_days * 96
            vals = vals[:total_needed]

            # 按天切块：每 96 条为一组 -> 形状 (n_days, 96)，再转置成 (96, n_days)
            # 这样就对应 行=96个时段、列=n_days 天
            mat = vals.reshape(n_days, 96).T

            col_labels = [f"{d}-L{col[-1]}" for d in date_labels]

            # 组装成 DataFrame
            wide = pd.DataFrame(mat, index=times, columns=col_labels)
            wide.index.name = "Time"

            wide.iloc[:, :28].to_excel(writer, sheet_name=col)

    print("✅ Done:", OUTPUT_XLSX1)

    return


def beehub_data_processing():
    INPUT_CSV = './combined_Room4.csv'
    OUTPUT_XLSX = 'Beehub_raw_data.xlsx'

    # 读取
    df = pd.read_csv(INPUT_CSV, index_col=0)  # 保持你原来的读取方式（不改 index）

    # 只提取这一列
    col_name = 'occupant_count [number]'
    df_small = df[[col_name]].copy()

    # 每隔3行取一次（0,3,6, ...）
    df_sampled = df_small.iloc[::3].copy()

    # ===== 提取日期（日） =====
    # 把 index 转成 datetime（仅用于计算，不改变原 index）
    ts = pd.to_datetime(df_sampled.index, errors='coerce')  # 能解析含 +08:00 的时间
    unique_days = ts.normalize().unique()  # 变成当天00:00，去重并保持出现顺序

    # ===== 计算完整天数（每 96 条为一天） =====
    n_days = len(df_sampled) // 96
    unique_days = unique_days[:n_days]  # 只保留完整天

    # ===== 生成列标签，如 Sep7, Sep8, ... =====
    date_labels = [f"{d.strftime('%b')}{d.day}" for d in unique_days]

    # ===== 生成年内 96 个时间标签：0:15, 0:30, ..., 24:00 =====
    times = pd.date_range("00:15", "23:45", freq="15min").strftime("%H:%M").tolist()
    times = [t[1:] if t.startswith("0") else t for t in times] + ["24:00"]  # 只去掉一个前导0

    # ===== 拆分数据并组装矩阵 =====
    vals = df_sampled[col_name].to_numpy()[:n_days * 96]  # 用抽样后的数据！
    mat = vals.reshape(n_days, 96).T  # (96, n_days)

    col_labels = [f"{d}" for d in date_labels]  # 例如 Sep7-Occ
    wide = pd.DataFrame(mat, index=times, columns=col_labels)
    wide.index.name = "Time"

    # ===== 写入 Excel =====
    wide.iloc[:, 5:].to_excel(OUTPUT_XLSX, sheet_name="Occupancy")

    print("✅ Done:", OUTPUT_XLSX)

    return


if __name__ == "__main__":
    WIFI_data_processing()
    beehub_data_processing()


