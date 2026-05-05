import torch
import pandas as pd
from tqdm import tqdm
from comet import download_model, load_from_checkpoint
from ESA.annotation_loader import AnnotationLoader
import os

# --- 1. 数据准备 ---
print("Loading ESA dataset...")
loader = AnnotationLoader(refresh_cache=False)
df = loader.get_view(["ESA-1"], only_overlap=False)
error_df = df[df['ESA-1_error_spans'].notna()].copy()

def has_valid_spans(spans):
    if not isinstance(spans, list): return False
    return any(str(s.get('start_i', 'missing')).isdigit() for s in spans)

valid_error_df = error_df[error_df['ESA-1_error_spans'].apply(has_valid_spans)]
print(f"Total valid samples: {len(valid_error_df)}")

# --- 2. 加载模型 ---
print("Loading xCOMET model (Unbabel/XCOMET-XL)...")
model_path = download_model("Unbabel/XCOMET-XL")
model = load_from_checkpoint(model_path)
model.eval()
if torch.cuda.is_available(): model.to("cuda")
tokenizer = model.encoder.tokenizer

# --- 3. 标签逻辑 ---
def get_token_labels(error_spans, offsets):
    token_labels = [0] * len(offsets)
    for span in error_spans:
        s, e = span.get('start_i'), span.get('end_i')
        if not str(s).isdigit() or not str(e).isdigit(): continue
        s, e = int(s), int(e)
        for i, (tok_s, tok_e) in enumerate(offsets):
            if tok_s == tok_e: continue
            if max(s, tok_s) < min(e, tok_e):
                token_labels[i] = 1
    return token_labels

all_vectors = []
all_labels = []
all_tokens =[]
num_layers = None

# ================= 终极雷达搜索函数 =================
def find_input_dict(obj):
    """
    不管 COMET 返回什么鬼结构，递归钻进去找带有 input_ids 的字典
    """
    # 如果本身就是我们要的字典
    if isinstance(obj, dict) and "input_ids" in obj:
        return obj
    # 如果是类似字典的 BatchEncoding 对象
    if hasattr(obj, "keys") and hasattr(obj, "items") and "input_ids" in obj.keys():
        return dict(obj.items())
    # 如果是元组或列表，挨个剥开往里找
    if isinstance(obj, (list, tuple)):
        for item in obj:
            res = find_input_dict(item)
            if res is not None:
                return res
    # 没找到返回 None
    return None
# ===================================================

# --- 4. 开始提取 ---
for idx, row in tqdm(valid_error_df.iterrows(), total=len(valid_error_df), desc="Extracting"):
    source, mt, spans = row['source'], row['hypothesis'], row['ESA-1_error_spans']
    
    enc = tokenizer(mt, return_offsets_mapping=True, add_special_tokens=True)
    labels = get_token_labels(spans, enc['offset_mapping'])
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'])
    
    input_data =[{"src": source, "mt": mt, "ref": "", "score": 0.0}]
    prepared = model.prepare_sample(input_data)
    
    # 调用雷达搜索函数
    input_dict = find_input_dict(prepared)
    
    if input_dict is None:
        raise ValueError(f"提取失败！完全找不到 input_ids。prepared 长这样: {prepared}")
    
    # 最终确保将 tensor 移到 GPU
    if torch.cuda.is_available():
        input_dict = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}

    with torch.no_grad():
        outputs = model.encoder.model(
            input_ids=input_dict["input_ids"], 
            attention_mask=input_dict["attention_mask"], 
            output_hidden_states=True
        )
        full_hidden_states = outputs.hidden_states 
        
        if num_layers is None:
            num_layers = len(full_hidden_states)
            print(f"\n[INFO] Detected Layers: {num_layers}")

    # 对齐逻辑 (使用最后一个匹配位置 occurrences[-1])
    core_mt_ids = enc['input_ids'][1:-1]
    full_ids = input_dict["input_ids"][0].cpu().tolist()
    
    occurrences =[]
    window_size = len(core_mt_ids)
    for j in range(len(full_ids) - window_size + 1):
        if full_ids[j : j + window_size] == core_mt_ids:
            occurrences.append(j)
            
    if occurrences:
        start_idx = occurrences[-1]
        
        # 提取 37 层数据
        # 1. 堆叠 [37, 1, Seq, Dim] ->[37, Seq, Dim]
        stacked = torch.stack(full_hidden_states).squeeze(1)
        # 2. 切片并转 CPU [37, Window, Dim]
        mt_all = stacked[:, start_idx : start_idx + window_size, :].cpu().float()
        # 3. 维度重排 -> [Window, 37, Dim]
        mt_all = mt_all.permute(1, 0, 2)
        
        for i in range(window_size):
            all_vectors.append(mt_all[i])
            all_labels.append(labels[1:-1][i])
            all_tokens.append(tokens[1:-1][i])
    else:
        print(f"Warning: Align failed for sample {idx}")

# --- 5. 保存 ---
output_dir = "extract_all_layers_output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "experiment_esa1_all_layers.pt")

print(f"\nStacking {len(all_vectors)} tokens...")
X_final = torch.stack(all_vectors)

print(f"Saving to {output_file}...")
torch.save({
    "X": X_final, 
    "y": torch.tensor(all_labels),
    "tokens": all_tokens
}, output_file)

print("\n" + "="*50)
print("SUCCESS!")
print(f"Shape: {X_final.shape}")
print("="*50)