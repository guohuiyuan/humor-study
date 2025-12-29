import pandas as pd
import json
import random

# 配置
INPUT_CSV = "data/xhstext.csv"
OUTPUT_JSONL = "data/sharegpt_data_original.jsonl"

# 系统提示词
SYSTEM_PROMPT = "你是一个幽默风趣、熟悉中文互联网流行文化的“梗王”。请创作简短、犀利、带有自嘲或神回复性质的社交媒体文案。"
USER_INSTRUCTION = "生成一条“有梗”的文案。"

def main():
    # 读取数据
    df = pd.read_csv(INPUT_CSV)
    
    # 筛选标签为 '有梗' 的数据
    # 注意：根据你的CSV，标签列名是 'label'，文本是 'text'
    df_geng = df[df['label'] == '有梗'].copy()
    
    sharegpt_data = []
    
    for _, row in df_geng.iterrows():
        text = str(row['text']).strip()
        if not text:
            continue
            
        # 构建 ShareGPT 格式 (Single-turn 对话)
        conversation = {
            "conversations": [
                {
                    "from": "system",
                    "value": SYSTEM_PROMPT
                },
                {
                    "from": "user",
                    "value": USER_INSTRUCTION
                },
                {
                    "from": "assistant",
                    "value": text
                }
            ]
        }
        sharegpt_data.append(conversation)
    
    # 保存
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for entry in sharegpt_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"转换完成，共提取 {len(sharegpt_data)} 条数据至 {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()