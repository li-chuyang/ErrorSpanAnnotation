import pandas as pd
import numpy as np
from ESA.annotation_loader import AnnotationLoader
from ESA.utils import PROTOCOL_DEFINITIONS

# ================= 配置区 =================
# 你想检查的协议列表，可以放多个进行对比
# 建议检查: "ESA-1", "ESA-2", "MQM-1", "LLM"
CHECK_PROTOCOLS = ["ESA-1", "ESA-2", "MQM-1", "LLM"] 
SAMPLE_SIZE = 3  # 每个协议抽查多少个样本
# ==========================================

def has_valid_spans(spans):
    if not isinstance(spans, list): return False
    return any(str(s.get('start_i', 'missing')).isdigit() for s in spans)

def get_severity_counts(df, col_name):
    severities = []
    for spans in df[col_name].dropna():
        for s in spans:
            if str(s.get('start_i')).isdigit():
                severities.append(s.get('severity', 'unknown'))
    return pd.Series(severities).value_counts()

print("开始全能数据集扫描...")
loader = AnnotationLoader(refresh_cache=False)

# 1. 扫描所有定义的协议
print("\n[1] 协议库定义概览:")
available_protocols = list(PROTOCOL_DEFINITIONS.keys())
print(f"配置文件中共定义了 {len(available_protocols)} 个协议:")
print(f"列表: {available_protocols}")

# 2. 逐一深入分析
for proto in CHECK_PROTOCOLS:
    print(f"\n\n{'='*30} 正在深入分析协议: {proto} {'='*30}")
    
    try:
        # 获取该协议的完整视图
        df = loader.get_view([proto], only_overlap=False)
        error_col = f"{proto}_error_spans"
        
        if error_col not in df.columns:
            print(f"错误: 协议 {proto} 的数据中未发现列 {error_col}")
            continue

        # --- A. 基础统计 ---
        total_rows = len(df)
        with_errors = df[df[error_col].notna()]
        valid_coords = with_errors[with_errors[error_col].apply(has_valid_spans)]
        
        print(f"样本规模:")
        print(f"   - 总行数: {total_rows}")
        print(f"   - 包含错误标注的行数: {len(with_errors)}")
        print(f"   - 坐标明确(可用于实验)的行数: {len(valid_coords)}")
        print(f"   - 坐标缺失(missing)的行数: {len(with_errors) - len(valid_coords)}")

        # --- B. 领域与语言分布 ---
        print(f"\n领域分布 (Domains):")
        if 'domainID' in df.columns:
            print(df['domainID'].value_counts().to_string())
        
        print(f"\n翻译系统分布 (Top 5 Systems):")
        if 'systemID' in df.columns:
            print(df['systemID'].value_counts().head(5).to_string())

        # --- C. 错误严重程度分布 ---
        print(f"\n错误严重程度统计 (Span-level):")
        print(get_severity_counts(df, error_col).to_string())

        # --- D. 抽样内容还原 (对齐验证) ---
        print(f"\n🔍 抽样还原验证 (前 {SAMPLE_SIZE} 个可用样本):")
        samples = valid_coords.head(SAMPLE_SIZE)
        for idx, (_, row) in enumerate(samples.iterrows()):
            hyp = row['hypothesis']
            spans = row[error_col]
            print(f"\n   样本 {idx+1} [ID: {row.get('hypothesisID', 'N/A')}]")
            print(f"   HYP: {hyp[:100]}...")
            for s_idx, s in enumerate(spans):
                if str(s.get('start_i')).isdigit():
                    start, end = int(s['start_i']), int(s['end_i'])
                    txt = hyp[start:end]
                    print(f"      - 错误 {s_idx+1}: [{start}:{end}] -> \"{txt}\" ({s.get('severity')})")

    except Exception as e:
        print(f"分析协议 {proto} 时发生错误: {e}")

# 3. 跨协议重叠分析 (核心研究点)
print(f"\n\n{'='*30} 跨协议一致性分析 {'='*30}")
try:
    df_overlap = loader.get_view(CHECK_PROTOCOLS, only_overlap=True)
    print(f"在所选协议中，共有 {len(df_overlap)} 条句子是互相重叠的。")
    print("这说明这些句子同时被 ESA-1, ESA-2, MQM-1 标注过，非常适合做对比实验！")
except:
    print("无法进行重叠分析。")

print("\n扫描结束！")