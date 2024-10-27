import matplotlib.pyplot as plt

# ファイルパスの設定
input_file_path = '/mnt/kiso-qnap/abe/b4/espnet/egs2/librispeech_100/asr1/dump/raw/test_clean/text'
line_count = 2700  # 読み込む最大行数
output_image_path = f'line_lengths_plot_first_{line_count}.png'  # 保存する画像ファイルのパス

# テキストファイルを読み込み、データをグラフ化するプログラム
lengths = []

try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        # 最初の{line_count}行だけを読み取る
        for i, line in enumerate(file):
            if i >= line_count:
                break
            # 各行のID部分(最初の20文字)を取り除いて、残りの文章部分の文字数を計算
            text_part = line[20:].strip()  # ID部分を無視して、残りのテキストのみを取得
            lengths.append(len(text_part))

    # x軸は行番号 (1から開始)
    lines = list(range(1, len(lengths) + 1))

    # グラフを作成
    plt.figure(figsize=(10, 6))
    plt.plot(lines, lengths, marker='o', linestyle='-', color='b')

    # グラフのタイトルとラベルを設定
    plt.title(f'Line Lengths First {line_count}')
    plt.xlabel('Line Number')
    plt.ylabel('Length (characters)')

    # グリッドを表示
    plt.grid(True)

    # x軸のラベルを適切な間隔で表示
    plt.xticks(range(0, len(lines), max(1, len(lines) // 10)))

    # グラフを保存
    plt.tight_layout()  # レイアウトを調整
    plt.savefig(output_image_path)

    print(f"The plot of the first {line_count} line lengths has been saved as '{output_image_path}'.")

except FileNotFoundError:
    print(f"Error: The file '{input_file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
