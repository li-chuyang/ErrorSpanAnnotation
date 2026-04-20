import torch
import pandas as pd
from comet import download_model, load_from_checkpoint
from ESA.annotation_loader import AnnotationLoader

# 1. 加载 ESA 数据集
print("正在加载数据...")
loader = AnnotationLoader(refresh_cache=False)
df = loader.get_view(["MQM-1", "ESA-1"], only_overlap=False)
error_df = df[df['ESA-1_error_spans'].notna()].copy()

# 2. 加载模型
model_name = "Unbabel/XCOMET-XL" 
print(f"正在准备模型: {model_name}...")
model_path = download_model(model_name) 
model = load_from_checkpoint(model_path)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")
tokenizer = model.encoder.tokenizer

# 3. 对齐函数
def get_token_labels(mt_text, error_spans_list, tokenizer_outputs):
    offsets = tokenizer_outputs['offset_mapping']
    token_labels = [0] * len(offsets)
    
    for span in error_spans_list:
        start = span.get('start_i')
        end = span.get('end_i')
        if start == 'missing' or end == 'missing' or start is None or end is None:
            continue
        
        start, end = int(start), int(end)
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == tok_end == 0: continue
            if max(start, tok_start) < min(end, tok_end):
                token_labels[i] = 1
    return token_labels

# 4. 运行处理
all_visual_data = []

print("开始处理样本...")
# 处理前 10 条有效数据
for idx, row in error_df.head(10).iterrows():
    source = row['source']
    mt = row['hypothesis']
    error_spans = row['ESA-1_error_spans'] 

    # 分词与标签生成
    encoding_mt = tokenizer(mt, return_offsets_mapping=True, add_special_tokens=True)
    labels = get_token_labels(mt, error_spans, encoding_mt)
    mt_tokens = tokenizer.convert_ids_to_tokens(encoding_mt["input_ids"])

    if sum(labels) == 0:
        print(f"跳过句子 {idx}: 仅包含漏译错误")
        continue

    # 准备模型输入
    input_data = [{"src": source, "mt": mt, "ref": "", "score": 0.0}]
    model_input = model.prepare_sample(input_data)

    # 剥离元组，获取输入字典
    while isinstance(model_input, (tuple, list)):
        model_input = model_input[0]

    # 移动到设备
    if torch.cuda.is_available():
        model_input = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in model_input.items()}

    # --- 核心改进：直接调用底层模型获取 hidden_states ---
    with torch.no_grad():
        # 这里直接访问 model.encoder 内部的底层的 transformer 模型
        # 通常是 XLM-RoBERTa
        outputs = model.encoder.model(
            input_ids=model_input["input_ids"],
            attention_mask=model_input["attention_mask"],
            output_hidden_states=True,
            return_dict=True
        )
    
    # 获取隐藏状态 (HF 模型返回的是字典或对象，一定包含 hidden_states)
    if "hidden_states" in outputs:
        hidden_states = outputs["hidden_states"]
    elif hasattr(outputs, "hidden_states"):
        hidden_states = outputs.hidden_states
    else:
        # 最后的保底方案
        print(f"无法获取句子 {idx} 的隐藏状态，可用的 Key: {outputs.keys()}")
        continue

    # 获取最后一层神经元 [Seq_Len, Hidden_Dim]
    last_layer = hidden_states[-1][0] 
    
    # 寻找 MT 在长序列中的起始位置 (切片逻辑)
    mt_input_ids = torch.tensor(encoding_mt["input_ids"]).to(last_layer.device)
    full_seq = model_input["input_ids"][0]
    mt_len = mt_input_ids.size(0)
    
    start_index = -1
    for i in range(len(full_seq) - mt_len + 1):
        if torch.equal(full_seq[i:i+mt_len], mt_input_ids):
            start_index = i
            break
    
    if start_index != -1:
        # 切片出 MT 对应的神经元部分
        mt_neurons = last_layer[start_index : start_index + mt_len]
        
        # 存入结果
        for i in range(len(mt_tokens)):
            all_visual_data.append({
                "Sentence_ID": idx,
                "Token": mt_tokens[i],
                "Is_Error": labels[i],
                # 如果你想看具体的神经元平均强度，可以保留下面这行
                "Neuron_Mean": mt_neurons[i].mean().item() 
            })
        print(f"成功处理句子 {idx}")

# 5. 保存结果
if all_visual_data:
    visual_df = pd.DataFrame(all_visual_data)
    visual_df.to_csv("alignment_check.csv", index=False, encoding='utf-8-sig')
    print(f"\n实验完成！结果已保存至 'alignment_check.csv'")
else:
    print("\n未处理任何有效数据。")