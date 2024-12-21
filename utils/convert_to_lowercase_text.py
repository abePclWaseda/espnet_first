input_file = '/mnt/kiso-qnap/abe/b4/espnet/egs2/librispeech_100/asr1/dump/raw/dev/text'
output_file = '/mnt/kiso-qnap/abe/b4/espnet/egs2/librispeech_100/asr1/dump/raw/dev/text_lower_text.txt'

with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        # 行を前後の空白を除去して取得
        line = line.strip()
        # 最初のスペースで分割
        parts = line.split(' ', 1)
        if len(parts) == 2:
            left_side = parts[0]
            right_side = parts[1].lower()
            # 左側と小文字化した右側を結合して出力
            f_out.write(f"{left_side} {right_side}\n")
        else:
            # スペースがない場合はそのまま出力
            f_out.write(line + '\n')