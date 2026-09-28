
import numpy as np
import pandas as pd
from tqdm import tqdm

def build_matrix_for_days(high_list, train_index, max_set=25):
    # 初始化转移矩阵
    transition_matrices = []

    train_high_list = [high_list[i] for i in train_index]

    week_df = pd.DataFrame(columns=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    for df in train_high_list:
        # Split each DataFrame into 7 days (96 rows for each day)
        days_data = {
            "Monday": df.iloc[0:96].values.flatten(),
            "Tuesday": df.iloc[96:192].values.flatten(),
            "Wednesday": df.iloc[192:288].values.flatten(),
            "Thursday": df.iloc[288:384].values.flatten(),
            "Friday": df.iloc[384:480].values.flatten(),
            "Saturday": df.iloc[480:576].values.flatten(),
            "Sunday": df.iloc[576:672].values.flatten(),
        }

        week_data = pd.DataFrame(days_data)

        week_df = pd.concat([week_df, week_data], ignore_index=True)

    for day in week_df.columns:
        occupancy_data = week_df[day].values  # Get the data for the specific day
        transition_matrix = compute_transition_matrix(occupancy_data, max_occupancy=max_set)
        transition_matrices.append(transition_matrix)

    return transition_matrices


def compute_transition_matrix(occupancy_data, max_occupancy):
    transition_matrix = np.zeros((max_occupancy + 1, max_occupancy + 1))

    for i in range(len(occupancy_data) - 1):
        current_state = occupancy_data[i]
        next_state = occupancy_data[i + 1]

        transition_matrix[current_state, next_state] += 1

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    for i in range(transition_matrix.shape[0]):
        if row_sums[i] == 0:
            transition_matrix[i, :] = 1 / (max_occupancy + 1)

    # Normalize the matrix again to ensure each row sums to 1
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    transition_matrix = transition_matrix / row_sums

    return transition_matrix


def simulate_next_state(current_state, markov_matrix, steps=4):
    simulated_sequence = [current_state]

    for _ in range(steps):
        # Use the Markov matrix to randomly select the next state
        next_state = np.random.choice(np.arange(len(markov_matrix)), p=markov_matrix[current_state])
        simulated_sequence.append(next_state)
        current_state = next_state

    return simulated_sequence


def markov_matrix_process(matrix_list, low_list, test_index, test_index_name, steps=4, max_trials=1000):
    test_low_list = [low_list[i] for i in test_index]

    test_data_all = []


    red_desc = "\033[32mMarkov Processing\033[0m"
    bar_format = "\033[32m{l_bar}\033[32m{bar}\033[32m{r_bar}\033[0m"

    for df in tqdm(test_low_list, desc=red_desc, bar_format=bar_format):
        for day in range(7):
            markov_matrix = matrix_list[day]
            days_data_list = df.iloc[24 * day: 24 + 24 * day].values.flatten().tolist()
            days_data_list = [0] + days_data_list  # 如果不想从 0 起步，可删除这一行

            for i in range(len(days_data_list) - 1):
                current_state = days_data_list[i]
                next_state = days_data_list[i + 1]

                # 记录“最佳候选”（与真实 next_state 的距离最近）
                best_seq = None
                best_dist = float("inf")
                chosen_seq = None

                for _ in range(max_trials):
                    simulated_sequence = simulate_next_state(current_state, markov_matrix, steps=steps)
                    last_state = simulated_sequence[-1]

                    # 命中：第 steps 步刚好等于真实 next_state，直接采用并跳出
                    if last_state == next_state:
                        chosen_seq = simulated_sequence
                        break

                    # 未命中：更新“最近”的候选
                    dist = abs(last_state - next_state)  # 如果状态是类别且有自定义距离，可替换这行
                    if dist < best_dist:
                        best_dist = dist
                        best_seq = simulated_sequence

                # 如果没命中，就用最近候选
                if chosen_seq is None:
                    chosen_seq = best_seq

                # 兜底：极端情况下 best_seq 可能仍是 None（比如 max_trials=0）
                if chosen_seq is None:
                    chosen_seq = simulate_next_state(current_state, markov_matrix, steps=steps)

                test_data_all.append(chosen_seq[1:])  # 不包含初始状态

    flattened_data = np.array(test_data_all).flatten()

    result_df = pd.DataFrame({
        "Markov": flattened_data
    }, index=test_index_name)

    return result_df