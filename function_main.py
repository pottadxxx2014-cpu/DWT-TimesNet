from function_package import (set_seed, read_data, read_data_multi_sheets, index_generating, create_train_test_split_indices,
                              create_train_test_split_indices_2, result_template_generate, generate_sheet_combinations,
                              create_train_test_indices_by_sheet)
from model_package import results_dataframe_merge, results_dataframe_merge_DWT
from model_validation import result_analysis
from cross_analysis import cross_analysis_function_main
from DWT_analysis import DWT_analysis_main
from WIFI_analysis import WIFI_analysis_main
from pathlib import Path
from runtime_analysis import create_run, mark_completed, runtime_summary

CONFIG = {
    'one_simu':{
        'seed': 2026,  # 设置随机种子，default = 2025
        'train_factor': 1,  # 训练轮数衰减系数，利用小的factor来测试，default = 1
        'file_set': 'Raw',  # 有'Raw'、'Month' 两个数据集供使用，default = 'Raw'
        'test_size': 0.5,  # 测试集占比，default = 0.5
        'wavelet_set': 'bior1.5',  # 设置小波基的形式，default = 'haar'
        'level_set': 2,  # 设置低时频数据离散小波变换的级数，default = 2
        'outfile': './Results_V2_2_Raw',  # 设置结果输出文件夹
        'picture_set': True,  # 设置是否输出所有图片
        'direct_set': True,  # 增加Direct TimesNet，原五种方法不改
        'additional_set': True,  # 结果保存后补算附加指标
        'save_weights': False,  # 默认不保存大权重文件，分支预测和训练日志始终保存
    },
    'cross_simu':{
        'train_factor': 1,  # 训练轮数衰减系数，利用小的factor来测试，default = 1
        'file_set': 'Raw',  # 有'Raw'、'Month' 两个数据集供使用，default = 'Raw'
        'test_size': 0.5,  # 测试集占比，default = 0.5
        'wavelet_set': 'bior1.5',  # 设置小波基的形式，default = 'haar'
        'level_set': 2,  # 设置低时频数据离散小波变换的级数，default = 2
        'outfile': './Results_V2_2_Cross',  # 设置结果输出文件夹
        'picture_set': False,  # 设置是否输出所有图片
        'direct_set': True,  # 增加Direct TimesNet，原五种方法不改
        'additional_set': True,  # 结果保存后补算附加指标
        'save_weights': False,  # 默认不保存大权重文件，分支预测和训练日志始终保存
        'seed_start': 2025,  # 包含
        'seed_stop': 2225,  # 不包含，共200轮
        'skip_completed': True,  # 同配置/数据/源码下仅跳过完整且校验通过的轮次
    },
    'cross_analysis':{
        'folder_cross': './Results_V2_2_Cross',  # 设置交叉检验读取文件夹
        'outfile': 'Statistical_Results_V2_2.xlsx',  # 设置结果输出文件名称
    },
    'DWT_simu':{
        'train_factor': 1,  # 训练轮数衰减系数，利用小的factor来测试，default = 1
        'file_set': 'Raw',  # 有'Raw'、'Month' 两个数据集供使用，default = 'Raw'
        'test_size': 0.5,  # 测试集占比，default = 0.5
        'level_set': 2,  # 设置低时频数据离散小波变换的级数，default = 2
        'outfile': './Results_V2_2_DWT',  # 设置结果输出文件夹
        'picture_set': False,  # 设置是否输出所有图片
        'wavelet_list': ["haar", "db2", "db8", "sym2", "sym10", "coif1", "coif5", "bior1.1", "bior1.5", "bior4.4", "rbio1.1", "rbio4.4", "dmey"],  # 设置需要遍历的小波基函数
        'wavelet_list1': ["haar", "db2"],
                'additional_set': True,  # 结果保存后补算附加指标
        'save_weights': False,  # 默认不保存大权重文件，分支预测和训练日志始终保存
    },
    'DWT_analysis':{
        'folder_cross': './Results_V2_2_DWT',  # 设置小波基函数读取文件夹
    },
    'Beehub_simu':{
        'seed': 2026,  # 设置随机种子，default = 2025
        'train_factor': 1,  # 训练轮数衰减系数，利用小的factor来测试，default = 1
        'file_set': 'Raw',  # 有'Raw'、'Month' 两个数据集供使用，default = 'Raw'
        'wavelet_set': 'bior1.5',  # 设置小波基的形式，default = 'haar'
        'level_set': 2,  # 设置低时频数据离散小波变换的级数，default = 2
        'outfile': './Results_V2_2_Beehub',  # 设置结果输出文件夹
        'picture_set': False,  # 设置是否输出所有图片
        'direct_set': True,  # 增加Direct TimesNet，原五种方法不改
        'additional_set': True,  # 结果保存后补算附加指标
        'save_weights': False,  # 默认不保存大权重文件，分支预测和训练日志始终保存
    },
    'WIFI_simu': {
        'seed': 2026,  # 设置随机种子，default = 2025
        'train_factor': 1,  # 训练轮数衰减系数，利用小的factor来测试，default = 1
        'file_set': 'Raw',  # 有'Raw'、'Month' 两个数据集供使用，default = 'Raw'
        'wavelet_set': 'bior1.5',  # 设置小波基的形式，default = 'haar'
        'level_set': 2,  # 设置低时频数据离散小波变换的级数，default = 2
        'outfile': './Results_V2_2_WIFI-1',  # 设置结果输出文件夹
        'total_floor': 5,  # 总层数为5
        'train_floor_set': 1,  # 训练集选择的层数，默认为4
        'break_set': 100,  # 遍历几次跳出循环，默认设定为100不跳出
        'max_set': 225,  # 最大值
        'picture_set': False,  # 设置是否输出所有图片
        'direct_set': True,  # 增加Direct TimesNet，原五种方法不改
        'additional_set': True,  # 结果保存后补算附加指标
        'save_weights': False,  # 默认不保存大权重文件，分支预测和训练日志始终保存
    },
    'WIFI_analysis': {
        'folder_1': './Results_V2_2_WIFI-1',
        'folder_2': './Results_V2_2_WIFI-2',
        'folder_3': './Results_V2_2_WIFI-3',
        'folder_4': './Results_V2_2_WIFI-4',
    },
    'time_simu': {
        'seed': 2026, 'train_factor': 1, 'file_set': 'Raw', 'test_size': 0.5,
        'wavelet_set': 'bior1.5', 'level_set': 2,
        'outfile': './Results_V2_2_Time', 'picture_set': False,
        'direct_set': True, 'additional_set': True, 'save_weights': False,
    },
    'ablation_simu': {
        'source_folder': './Results_V2_2_Raw',
        'source_run': None,  # None：该目录中seed对应的最新V2_2分支记录；也可填完整案例路径
        'seed': 2026,
        'retrain': False,  # False复用独立分支；True逐变体只重训保留的分支（会明显更慢）
        'variant_list': ['Full', 'Low-only', 'Without-D1', 'Without-D2',
                         'Without-CNN-zero-low', 'Fixed-low-plus-high'],
    },
    'additional_analysis': {
        'source_folder': './Results_V2_2_Raw', 'source_run': None, 'seed': 2026,
        'bootstrap_set': False, 'bootstrap_number': 1000, 'block_weeks': 2,
    },
}


def main_one_simu():
    # 0.设置种子
    set_seed(CONFIG['one_simu']['seed'])
    train_factor = CONFIG['one_simu']['train_factor']

    # 1.读取高时频数据，生成对应的低时频数据，并按照周划分（忽略前三天，从周一开始）
    if CONFIG['one_simu']['file_set'] == 'Month':
        high_occupancy_list, low_occupancy_list, week_number = read_data('occupancy_office_month.xlsx', ignore_cols=3)
    else:
        high_occupancy_list, low_occupancy_list, week_number = read_data('occupancy_office_rawdata.xlsx', ignore_cols=3)

    # 2.随机划分训练集和测试集，生成结果模板
    train_test_index = create_train_test_split_indices(week_number, test_size=CONFIG['one_simu']['test_size'])
    test_index_name = index_generating(high_occupancy_list, train_test_index)
    template_df = result_template_generate(high_occupancy_list, train_test_index, test_index_name)
    log_dir = create_run(CONFIG['one_simu'], high_occupancy_list, train_test_index, CONFIG['one_simu']['seed'], tag='one')
    if log_dir is None: return


    # 3.根据不同的基准模型模拟生成高时频数据
    results_df_all = results_dataframe_merge(template_df, high_occupancy_list, low_occupancy_list, week_number, train_test_index, test_index_name, train_factor,
                                             CONFIG['one_simu']['wavelet_set'], CONFIG['one_simu']['level_set'], CONFIG['one_simu']['picture_set'], max_set=25, log_dir=log_dir, direct_set=CONFIG['one_simu']['direct_set'], save_weights=CONFIG['one_simu']['save_weights'])

    # 4.模型检验
    result_analysis(results_df_all, CONFIG['one_simu']['outfile'], CONFIG['one_simu']['picture_set'], CONFIG['one_simu']['seed'], cross_set=False, log_dir=log_dir, additional_set=CONFIG['one_simu']['additional_set'])
    mark_completed(log_dir)

    return


def main_cross_simu(seed_set):
    # 0.设置种子
    set_seed(seed_set)
    train_factor = CONFIG['cross_simu']['train_factor']

    # 1.读取高时频数据，生成对应的低时频数据，并按照周划分（忽略前三天，从周一开始）
    if CONFIG['cross_simu']['file_set'] == 'Month':
        high_occupancy_list, low_occupancy_list, week_number = read_data('occupancy_office_month.xlsx', ignore_cols=3)
    else:
        high_occupancy_list, low_occupancy_list, week_number = read_data('occupancy_office_rawdata.xlsx', ignore_cols=3)

    # 2.随机划分训练集和测试集，生成结果模板
    train_test_index = create_train_test_split_indices(week_number, test_size=CONFIG['cross_simu']['test_size'])
    test_index_name = index_generating(high_occupancy_list, train_test_index)
    template_df = result_template_generate(high_occupancy_list, train_test_index, test_index_name)
    log_dir = create_run(CONFIG['cross_simu'], high_occupancy_list, train_test_index, seed_set, tag='cross')
    if log_dir is None: return


    # 3.根据不同的基准模型模拟生成高时频数据
    results_df_all = results_dataframe_merge(template_df, high_occupancy_list, low_occupancy_list, week_number, train_test_index, test_index_name, train_factor,
                                             CONFIG['cross_simu']['wavelet_set'], CONFIG['cross_simu']['level_set'], CONFIG['cross_simu']['picture_set'], max_set=25, log_dir=log_dir, direct_set=CONFIG['cross_simu']['direct_set'], save_weights=CONFIG['cross_simu']['save_weights'])

    # 4.模型检验
    result_analysis(results_df_all, CONFIG['cross_simu']['outfile'], CONFIG['cross_simu']['picture_set'], seed_set, cross_set=True, log_dir=log_dir, additional_set=CONFIG['cross_simu']['additional_set'])
    mark_completed(log_dir)

    return


def main_cross_analysis():
    cross_analysis_function_main(CONFIG['cross_analysis']['folder_cross'], CONFIG['cross_analysis']['outfile'])
    return


def main_DWT_simu(seed_set):
    # 0.设置种子
    set_seed(seed_set)
    train_factor = CONFIG['DWT_simu']['train_factor']

    # 1.读取高时频数据，生成对应的低时频数据，并按照周划分（忽略前三天，从周一开始）
    if CONFIG['DWT_simu']['file_set'] == 'Month':
        high_occupancy_list, low_occupancy_list, week_number = read_data('occupancy_office_month.xlsx', ignore_cols=3)
    else:
        high_occupancy_list, low_occupancy_list, week_number = read_data('occupancy_office_rawdata.xlsx', ignore_cols=3)

    # 2.随机划分训练集和测试集，生成结果模板
    train_test_index = create_train_test_split_indices(week_number, test_size=CONFIG['DWT_simu']['test_size'])
    test_index_name = index_generating(high_occupancy_list, train_test_index)
    template_df = result_template_generate(high_occupancy_list, train_test_index, test_index_name)
    log_dir = create_run(CONFIG['DWT_simu'], high_occupancy_list, train_test_index, seed_set, tag='wavelet')
    if log_dir is None: return


    # 3.根据不同的基准模型模拟生成高时频数据
    results_df_all = results_dataframe_merge_DWT(template_df, high_occupancy_list, low_occupancy_list, week_number, train_test_index, test_index_name, train_factor,
                                                 CONFIG['DWT_simu']['level_set'], CONFIG['DWT_simu']['picture_set'], CONFIG['DWT_simu']['wavelet_list'], log_dir=log_dir, save_weights=CONFIG['DWT_simu']['save_weights'])

    # 4.模型检验
    result_analysis(results_df_all, CONFIG['DWT_simu']['outfile'], CONFIG['DWT_simu']['picture_set'], seed_set, cross_set=True, log_dir=log_dir, additional_set=CONFIG['DWT_simu']['additional_set'])
    for child in Path(log_dir).glob('wavelet_*'):
        if child.is_dir(): runtime_summary(child)
    print('小波结果仅作当前测试划分的探索性敏感性分析，不自动修改主案例小波。')

    return


def main_DWT_analysis():
    DWT_analysis_main(CONFIG['DWT_analysis']['folder_cross'])
    return

def main_generalization_test():
    # 0.设置种子
    set_seed(CONFIG['Beehub_simu']['seed'])
    train_factor = CONFIG['Beehub_simu']['train_factor']

    # 1.读取高时频数据，生成对应的低时频数据，并按照周划分（忽略前三天，从周一开始）
    if CONFIG['Beehub_simu']['file_set'] == 'Month':
        high_occupancy_list_A, low_occupancy_list_A, week_number_A = read_data('occupancy_office_month.xlsx', ignore_cols=3)
    else:
        high_occupancy_list_A, low_occupancy_list_A, week_number_A = read_data('occupancy_office_rawdata.xlsx', ignore_cols=3)

    # 1.1 读取Beehub的高时频数据，并按周划分
    high_occupancy_list_B, low_occupancy_list_B, week_number_B = read_data('Beehub_raw_data.xlsx')

    # 1.2 合并成数据集
    high_occupancy_list = high_occupancy_list_A + high_occupancy_list_B
    low_occupancy_list = low_occupancy_list_A + low_occupancy_list_B
    week_number = week_number_A + week_number_B

    # 2.随机划分训练集和测试集，生成结果模板
    train_test_index = create_train_test_split_indices_2(high_occupancy_list_A, high_occupancy_list)
    test_index_name = index_generating(high_occupancy_list, train_test_index)
    template_df = result_template_generate(high_occupancy_list, train_test_index, test_index_name)
    log_dir = create_run(CONFIG['Beehub_simu'], high_occupancy_list, train_test_index, CONFIG['Beehub_simu']['seed'], tag='external')
    if log_dir is None: return


    # 3.根据不同的基准模型模拟生成高时频数据
    results_df_all = results_dataframe_merge(template_df, high_occupancy_list, low_occupancy_list, week_number, train_test_index, test_index_name, train_factor,
                                             CONFIG['Beehub_simu']['wavelet_set'], CONFIG['Beehub_simu']['level_set'],  CONFIG['Beehub_simu']['picture_set'], max_set=25, log_dir=log_dir, direct_set=CONFIG['Beehub_simu']['direct_set'], save_weights=CONFIG['Beehub_simu']['save_weights'])

    # 4.模型检验
    result_analysis(results_df_all, CONFIG['Beehub_simu']['outfile'], CONFIG['Beehub_simu']['picture_set'], CONFIG['Beehub_simu']['seed'], cross_set=False, log_dir=log_dir, additional_set=CONFIG['Beehub_simu']['additional_set'])
    mark_completed(log_dir)

    return

def main_WIFI_test():
    # 0.设置种子
    set_seed(CONFIG['WIFI_simu']['seed'])
    train_factor = CONFIG['WIFI_simu']['train_factor']

    # 1.读取高时频数据，生成对应的低时频数据，并按照周划分
    if CONFIG['WIFI_simu']['file_set'] == 'Month':
        high_occupancy_list_all, low_occupancy_list_all, week_number_list = read_data_multi_sheets('WIFI_month_data.xlsx', ignore_cols=0)
    else:
        high_occupancy_list_all, low_occupancy_list_all, week_number_list = read_data_multi_sheets('WIFI_raw_data.xlsx', ignore_cols=0)

    # 2.随机划分训练集和测试集，生成结果模板
    train_sheet_list = generate_sheet_combinations(CONFIG['WIFI_simu']['total_floor'], CONFIG['WIFI_simu']['train_floor_set'], 0)
    max_break = CONFIG['WIFI_simu']['break_set']
    break_set = 0
    for train_sheet in train_sheet_list:
        train_test_index = create_train_test_indices_by_sheet(train_sheet, week_number_list)
        test_index_name = index_generating(high_occupancy_list_all, train_test_index)
        template_df = result_template_generate(high_occupancy_list_all, train_test_index, test_index_name)
        log_dir = create_run(CONFIG['WIFI_simu'], high_occupancy_list_all, train_test_index, CONFIG['WIFI_simu']['seed'], tag='wifi_floors' + '-'.join(map(str, train_sheet)))
        if log_dir is None: continue


        # 3.根据不同的基准模型模拟生成高时频数据
        results_df_all = results_dataframe_merge(template_df, high_occupancy_list_all, low_occupancy_list_all, sum(week_number_list), train_test_index, test_index_name, train_factor,
                                                 CONFIG['WIFI_simu']['wavelet_set'], CONFIG['WIFI_simu']['level_set'], CONFIG['WIFI_simu']['picture_set'], max_set=CONFIG['WIFI_simu']['max_set'], log_dir=log_dir, direct_set=CONFIG['WIFI_simu']['direct_set'], save_weights=CONFIG['WIFI_simu']['save_weights'])

        # 4.模型检验
        result_analysis(results_df_all, CONFIG['WIFI_simu']['outfile'], CONFIG['WIFI_simu']['picture_set'], CONFIG['WIFI_simu']['seed'], cross_set=False, log_dir=log_dir, additional_set=CONFIG['WIFI_simu']['additional_set'])
        mark_completed(log_dir)

        break_set += 1
        if max_break is not None and break_set >= max_break:
            break

    return


def main_WIFI_analysis():
    WIFI_analysis_main(CONFIG['WIFI_analysis']['folder_1'], CONFIG['WIFI_analysis']['folder_2'],
                       CONFIG['WIFI_analysis']['folder_3'], CONFIG['WIFI_analysis']['folder_4'])
    return


def main_time_simu():
    # 按周序号前半训练、后半测试；测试周的小时观测仍作为插值输入。
    import numpy as np
    set_seed(CONFIG['time_simu']['seed'])
    train_factor = CONFIG['time_simu']['train_factor']
    filename = 'occupancy_office_month.xlsx' if CONFIG['time_simu']['file_set'] == 'Month' else 'occupancy_office_rawdata.xlsx'
    high_occupancy_list, low_occupancy_list, week_number = read_data(filename, ignore_cols=3)
    n_test = int(week_number * CONFIG['time_simu']['test_size'])
    if n_test < 1 or n_test >= week_number: raise ValueError('时间划分需要非空训练与测试集。')
    split_at = week_number - n_test
    train_test_index = (np.arange(split_at), np.arange(split_at, week_number))
    test_index_name = index_generating(high_occupancy_list, train_test_index)
    template_df = result_template_generate(high_occupancy_list, train_test_index, test_index_name)
    log_dir = create_run(CONFIG['time_simu'], high_occupancy_list, train_test_index,
                         CONFIG['time_simu']['seed'], tag='time')
    if log_dir is None: return
    results_df_all = results_dataframe_merge(template_df, high_occupancy_list, low_occupancy_list, week_number,
        train_test_index, test_index_name, train_factor, CONFIG['time_simu']['wavelet_set'],
        CONFIG['time_simu']['level_set'], CONFIG['time_simu']['picture_set'], max_set=25,
        log_dir=log_dir, direct_set=CONFIG['time_simu']['direct_set'], save_weights=CONFIG['time_simu']['save_weights'])
    result_analysis(results_df_all, CONFIG['time_simu']['outfile'], CONFIG['time_simu']['picture_set'],
                    CONFIG['time_simu']['seed'], cross_set=False, log_dir=log_dir,
                    additional_set=CONFIG['time_simu']['additional_set'])
    mark_completed(log_dir)
    return


def main_ablation_simu():
    from ablation_analysis import ablation_main
    return ablation_main(CONFIG['ablation_simu'])


def main_additional_analysis():
    from ablation_analysis import resolve_source_run
    from additional_analysis import analyse_saved_run
    cfg = CONFIG['additional_analysis']
    folder = resolve_source_run(cfg, require_components=False)
    return analyse_saved_run(folder, bootstrap=cfg['bootstrap_set'],
                             n_boot=cfg['bootstrap_number'], block_weeks=cfg['block_weeks'])
