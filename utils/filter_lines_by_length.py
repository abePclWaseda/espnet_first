import argparse

# コマンドライン引数のパーサーを作成
parser = argparse.ArgumentParser(description='各行の長さをフィルタリングして元のテキストをファイルに書き込むプログラム')
parser.add_argument('--max_length', type=int, default=4000, help='フィルタリングする最大行長（デフォルトは4000）')
args = parser.parse_args()

max_length = args.max_length

# ファイルパスの設定
input_file_path = 'egs2/librispeech_100/asr1/dump/raw/lm_train.txt'
output_file_path = f'egs2/librispeech_100/asr1/dump/raw/lm_train_filtered_{max_length}.txt'  # フィルタリングした後の元のテキストを記録する新しいファイル


# 各行の長さをフィルタリングして元のテキストをファイルに書き込むプログラム
try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # フィルタリングした行を出力ファイルに書き込む（長さがmax_length以下のもののみ）
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in lines:
            line_length = len(line.strip())
            if line_length <= max_length:  # 長さがmax_length以下のものをフィルタリング
                output_file.write(line)

    print(f"長さが{max_length}以下の行が'{output_file_path}'に書き込まれました。")

except FileNotFoundError:
    print(f"エラー: ファイル'{input_file_path}'が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {e}")