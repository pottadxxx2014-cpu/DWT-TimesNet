from function_main import *

def main(mode_set):
    mode = mode_set  # 保留数字菜单，函数调用不再依赖全局变量
    if mode == 1:
        main_one_simu()
    elif mode == 2:
        for seed_number in range(CONFIG['cross_simu']['seed_start'], CONFIG['cross_simu']['seed_stop']):
            main_cross_simu(seed_number)
    elif mode == 3:
        main_cross_analysis()
    elif mode == 4:
        seed_number = 2035
        main_DWT_simu(seed_number)
    elif mode == 5:
        main_DWT_analysis()
    elif mode == 6:
        # 各指标分别报告；p>0.05不能解释为分布等价
        main_generalization_test()
    elif mode == 7:
        # 4层训练，1层测试
        main_WIFI_test()
    elif mode == 8:
        main_WIFI_analysis()
    elif mode == 9:
        main_time_simu()
    elif mode == 10:
        main_ablation_simu()
    elif mode == 11:
        main_additional_analysis()
    return

if __name__ == "__main__":
    mode = int(input('Enter Mode:\n'
                     '(1) 模型单次训练\n'
                     '(2) 交叉检验\n'
                     '(3) 交叉检验结果分析\n'
                     '(4) 不同小波基函数\n'
                     '(5) 不同小波基函数结果分析\n'
                     '(6) 模型的外延性——不同建筑间推广，办公室1训练，办公室2测试\n'
                     '(7) 模型的外延性——对不同类型数据的通用性，利用WIFI数据\n'
                     '(8) WIFI数据分析\n'
                     '(9) 时间顺序划分，前半训练后半测试\n'
                     '(10) 独立消融实验（默认读取模式1已保存分支）\n'
                     '(11) 附加指标与可选分块区间（只读已有结果）\n'
                     '请输入需要的Mode:'))
    main(mode)