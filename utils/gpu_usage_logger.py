import subprocess
import time
import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ログファイルのパス
log_file = 'gpu_usage_log.csv'
graph_file = 'gpu_memory_usage_plot.png'

# ログファイルのヘッダーを書き込み
with open(log_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'GPU ID', 'Memory Used (MiB)'])

# GPU メモリ使用量を取得する関数
def get_gpu_memory_usage():
    try:
        utilization_result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, text=True, check=True
        )
        utilization_lines = utilization_result.stdout.strip().split('\n')
        gpu_data = []

        for line in utilization_lines:
            if not line.strip():
                continue
            data = line.split(',')
            if len(data) < 2:
                continue
            gpu_id = data[0].strip()
            memory_used = data[1].strip()
            gpu_data.append([gpu_id, memory_used])

        return gpu_data
    except Exception as e:
        print(f"Error retrieving GPU memory usage: {e}")
        return []

# メモリ使用量を時間ごとにログに記録し、プロット
def log_and_plot_gpu_memory_usage(interval=60, duration_hours=4, gpu_index_to_monitor=0):
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)

    timestamps = []
    memory_usage = []

    plt.figure(figsize=(10, 6))  # 初回に作成し、ループ内で更新する

    while datetime.now() < end_time:
        loop_start_time = time.time()

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        gpu_data = get_gpu_memory_usage()

        # GPU の数をチェックし、指定した GPU のデータを取得
        if len(gpu_data) == 0:
            print("Warning: No GPU data detected. Skipping logging.")
        else:
            gpu_index_to_monitor = min(gpu_index_to_monitor, len(gpu_data) - 1)  # 存在する GPU の範囲に制限
            gpu_id = gpu_data[gpu_index_to_monitor][0]

            try:
                memory_used = float(gpu_data[gpu_index_to_monitor][1])
            except ValueError:
                print("Error: Failed to convert memory usage to float.")
                continue

            # ログファイルに書き込み
            with open(log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, gpu_id, memory_used])

            # データをリストに保存
            timestamps.append(timestamp)
            memory_usage.append(memory_used)

            # プロットを更新
            plt.clf()
            plt.plot(timestamps, memory_usage, marker='o', linestyle='-')
            plt.title(f'GPU {gpu_id} Memory Usage Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Memory Used (MiB)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(graph_file)

        # 実際の処理時間を考慮してスリープ
        elapsed_time = time.time() - loop_start_time
        time.sleep(max(0, interval - elapsed_time))

if __name__ == '__main__':
    log_and_plot_gpu_memory_usage(interval=5, duration_hours=4, gpu_index_to_monitor=0)  # 5秒ごとにログを記録し、4時間後に終了
