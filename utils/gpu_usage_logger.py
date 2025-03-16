import subprocess
import time
import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# python3 ./tools/gpu_usage_logger.py で開始
# ログファイルのパスを指定
log_file = 'gpu_usage_log.csv'
graph_file = 'gpu_memory_usage_plot.png'

# ログファイルのヘッダーを書き込み
with open(log_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'GPU ID', 'Memory Used (MiB)'])

# nvidia-smi コマンドを実行して GPU のメモリ使用量を取得
def get_gpu_memory_usage():
    utilization_result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                                        stdout=subprocess.PIPE, text=True)
    utilization_lines = utilization_result.stdout.strip().split('\n')
    
    gpu_data = []
    
    for line in utilization_lines:
        data = line.split(',')
        gpu_id = data[0].strip()
        memory_used = data[1].strip()
        gpu_data.append([gpu_id, memory_used])
    
    return gpu_data

# メモリ使用量を時間ごとにログに記録し、プロット
def log_and_plot_gpu_memory_usage(interval=60, duration_hours=4):
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)

    timestamps = []
    memory_usage = []

    while datetime.now() < end_time:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        gpu_data = get_gpu_memory_usage()

        # 二番目のGPUのデータを取得
        if len(gpu_data) > 1:
            second_gpu_data = gpu_data[1]
            gpu_id = second_gpu_data[0]
            memory_used = float(second_gpu_data[1])

            # ログファイルに書き込み
            with open(log_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, gpu_id, memory_used])

            # データをリストに保存
            timestamps.append(timestamp)
            memory_usage.append(memory_used)

            # プロットを更新
            plt.figure(figsize=(10, 6))
            plt.plot(timestamps, memory_usage, marker='o')
            plt.title(f'GPU {gpu_id} Memory Usage Over Time')
            plt.xlabel('Timestamp')
            plt.ylabel('Memory Used (MiB)')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # グラフを保存
            plt.savefig(graph_file)

        # 指定された間隔で待機
        time.sleep(interval)

if __name__ == '__main__':
    log_and_plot_gpu_memory_usage(interval=5, duration_hours=4)  # 20秒ごとにログとプロットを更新し、4時間後に終了