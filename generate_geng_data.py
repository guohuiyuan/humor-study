import pandas as pd
import asyncio
import json
import random
import sys
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ================= 配置区域 =================

# vLLM 服务配置
VLLM_API_BASE = "http://10.249.42.129:8000/v1" 
VLLM_API_KEY = "apikey"
MODEL_NAME = 'qwen3-4b' 

# 生成配置
GENERATE_COUNT = 200      # 目标生成总数
BATCH_SIZE = 10            # 单次请求生成的条数
CONCURRENCY_LIMIT = 10     # 并发数 (建议设为 5-10，太高容易导致 vLLM 显存溢出或超时)
INPUT_CSV = "data/xhstext.csv"  
OUTPUT_JSONL = "data/sharegpt_data_synthetic.jsonl"
OUTPUT_CSV = "data/generated_geng_data.csv"
CHECKPOINT_FILE = "data/generation_checkpoint.json"

# 提示词模板
SYSTEM_PROMPT = """你是一个中文互联网“梗”生成器。请仔细观察给定的示例风格，生成类似的、简短的、幽默的、带有自嘲或网络流行语色彩的句子。
要求：
1. 输出必须是 JSON 列表格式：["句子1", "句子2", ...]
2. 风格要犀利、简短、有趣（类似小红书或微博的神回复）。
3. 不要解释，直接返回 JSON。"""

# ================= 核心逻辑 =================

def get_random_examples(df, n=5):
    """从现有数据中随机抽取n条作为参考"""
    if df.empty:
        return ["暂无参考数据"]
    # 防止抽取数量大于数据总量
    n = min(n, len(df))
    samples = df.sample(n)['text'].tolist()
    return samples

async def generate_batch(client, semaphore, df_source, pbar, jsonl_file, checkpoint_data):
    """
    单个并发任务：生成一批数据并立即写入文件
    返回：int (本次成功生成的数量)
    """
    async with semaphore:
        # 随机抽取 3-5 条现有数据作为 Few-shot (减少token消耗，3-5条足够)
        examples = get_random_examples(df_source, n=random.randint(3, 5))
        example_str = "\n".join([f"- {ex}" for ex in examples])
        
        # 即使我们要10条，让模型生成12条可以增加容错率
        request_count = BATCH_SIZE + 2
        
        user_content = f"""请模仿以下文案的风格和语气，创作 {BATCH_SIZE} 条全新的文案：

【参考样本】：
{example_str}

【任务】：
生成 {BATCH_SIZE} 条新的文案，保持上述风格。
请严格返回一个 JSON 字符串列表，例如：["文案1", "文案2"]。
"""

        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.95,    # 稍微调高温度
                top_p=0.95,
                max_tokens=2048,
                response_format={"type": "json_object"} 
            )

            content = response.choices[0].message.content.strip()
            
            # JSON 解析逻辑
            new_texts = []
            try:
                # 尝试清洗数据，防止 Markdown 代码块干扰
                if "```json" in content:
                    content = content.replace("```json", "").replace("```", "")
                
                start = content.find('[')
                end = content.rfind(']') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    parsed = json.loads(json_str)
                    if isinstance(parsed, list):
                        new_texts = parsed
            except json.JSONDecodeError:
                # 如果解析失败，这一批次就作废
                pass
            
            # 如果生成成功，写入文件
            if new_texts:
                # ShareGPT 固定的 Prompt
                ft_system = "你是一个幽默风趣、熟悉中文互联网流行文化的“梗王”。请创作简短、犀利、带有自嘲或神回复性质的社交媒体文案。"
                ft_user = "生成一条“有梗”的文案。"
                
                # 过滤掉过短的无效数据
                valid_texts = [t for t in new_texts if isinstance(t, str) and len(t) > 2]
                
                if not valid_texts:
                    return 0

                # 写入文件
                for text in valid_texts:
                    entry = {
                        "conversations": [
                            {"from": "system", "value": ft_system},
                            {"from": "user", "value": ft_user},
                            {"from": "assistant", "value": text}
                        ]
                    }
                    jsonl_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    jsonl_file.flush() # 确保立即写入磁盘
                
                count = len(valid_texts)
                
                # 更新全局状态
                checkpoint_data['generated_count'] += count
                checkpoint_data['batch_count'] += 1
                
                # 更新进度条
                pbar.update(count)
                
                # 保存检查点
                with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, ensure_ascii=False)
                
                return count
            
            return 0

        except Exception as e:
            print(f"\nRequest Error: {e}")
            return 0

async def main():
    # 1. 准备源数据
    if not os.path.exists(INPUT_CSV):
        print(f"Error: 找不到种子文件 {INPUT_CSV}")
        return
        
    df = pd.read_csv(INPUT_CSV)
    # 简单的容错读取
    if 'label' in df.columns:
        df_geng = df[df['label'] == '有梗'].dropna(subset=['text'])
    else:
        df_geng = df.dropna(subset=['text']) # 如果没有label列，就用全部
        
    print(f"加载种子数据：{len(df_geng)} 条")

    # 2. 检查点恢复逻辑
    checkpoint_data = {
        'generated_count': 0,
        'batch_count': 0,
        'total_target': GENERATE_COUNT
    }
    
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            print(f"-> 恢复进度: 已生成 {checkpoint_data['generated_count']} 条")
        except:
            print("-> 检查点文件损坏，重新开始")
    
    # 3. 初始化
    client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_API_BASE)
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    # 打开文件 (追加模式)
    jsonl_mode = 'a' if checkpoint_data['generated_count'] > 0 else 'w'
    jsonl_file = open(OUTPUT_JSONL, jsonl_mode, encoding='utf-8')
    
    # 进度条
    pbar = tqdm(total=GENERATE_COUNT, desc="生成进度", initial=checkpoint_data['generated_count'], unit="条")
    
    # ================= 动态循环核心修改 =================
    
    max_loop_safeguard = 0 
    MAX_LOOPS = (GENERATE_COUNT // BATCH_SIZE) * 5  # 防止无限死循环的安全阈值
    
    while checkpoint_data['generated_count'] < GENERATE_COUNT:
        # 1. 计算还缺多少
        remaining = GENERATE_COUNT - checkpoint_data['generated_count']
        
        # 2. 如果已经完成，退出循环
        if remaining <= 0:
            break
            
        # 3. 动态计算本轮需要发射多少任务
        # 逻辑：缺多少发多少，但单次不超过 CONCURRENCY_LIMIT * 2，避免积压太多任务
        batches_needed = (remaining + BATCH_SIZE - 1) // BATCH_SIZE
        tasks_to_launch = min(batches_needed, CONCURRENCY_LIMIT * 2) 
        
        # 4. 创建任务列表
        tasks = []
        for _ in range(tasks_to_launch):
            tasks.append(generate_batch(client, semaphore, df_geng, pbar, jsonl_file, checkpoint_data))
        
        # 5. 执行本轮任务
        await asyncio.gather(*tasks)
        
        # 安全检查：如果跑了太多轮还是没完成（可能是模型坏了），强制退出
        max_loop_safeguard += tasks_to_launch
        if max_loop_safeguard > MAX_LOOPS:
            print("\n警告：达到最大重试次数，可能 API 响应异常，提前结束。")
            break

    # ===================================================

    pbar.close()
    jsonl_file.close()

    # 6. 保存最终 CSV
    total_generated = checkpoint_data['generated_count']
    print(f"\n生成结束，当前总数: {total_generated} 条")

    if total_generated > 0:
        print("正在生成最终 CSV 文件...")
        all_texts = []
        with open(OUTPUT_JSONL, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    for conv in entry['conversations']:
                        if conv['from'] == 'assistant':
                            all_texts.append(conv['value'])
                            break
                except:
                    continue
        
        df_output = pd.DataFrame({
            'id': range(1, len(all_texts) + 1),
            'text': all_texts,
            'label': ['有梗'] * len(all_texts)
        })
        
        df_output.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        print(f"CSV 已保存: {OUTPUT_CSV}")
        
        # 只有真正达到目标才删除 checkpoint，方便下次补跑
        if total_generated >= GENERATE_COUNT:
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
    else:
        print("未生成有效数据。")

if __name__ == "__main__":
    # Windows 兼容性设置
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())