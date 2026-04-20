from ESA.annotation_loader import AnnotationLoader
from ESA.utils import PROTOCOL_DEFINITIONS
import pandas as pd
import sys

# ================= Configuration Area =================
# 1. 在这里填入你想查看的协议名称 (可以填一个或多个)
# 可选值通常包括: "ESA-1", "ESA-2", "MQM-1", "ESAAI-1", "LLM", "MQM-IAA" 等
MY_SELECTION = ["ESA-1", "ESA-2", "MQM-1", "LLM"] 

# 2. 每个协议展示多少个样本进行切片验证
SAMPLE_COUNT = 3
# =====================================================

def has_valid_spans(spans):
    """Check if the annotation list contains valid numeric coordinates"""
    if not isinstance(spans, list): return False
    return any(str(s.get('start_i', 'missing')).isdigit() for s in spans)

print("🚀 Initializing AnnotationLoader...")
loader = AnnotationLoader(refresh_cache=False)

# 校验输入的协议名是否在定义中
valid_definitions = list(PROTOCOL_DEFINITIONS.keys())

for protocol in MY_SELECTION:
    if protocol not in valid_definitions:
        print(f"\n❌ Error: Protocol '{protocol}' is not defined in utils.py.")
        print(f"Available protocols are: {valid_definitions}")
        continue

    print(f"\n\n{'#'*60}")
    print(f"🔍 INSPECTING PROTOCOL: {protocol}")
    print(f"{'#'*60}")

    try:
        # Load view for this specific protocol
        df = loader.get_view([protocol], only_overlap=False)
        error_col = f"{protocol}_error_spans"

        if error_col not in df.columns:
            print(f"❌ Column '{error_col}' not found in the loaded data.")
            continue

        # 1. Basic Stats
        total_rows = len(df)
        valid_rows_df = df[df[error_col].notna()].copy()
        high_quality_df = valid_rows_df[valid_rows_df[error_col].apply(has_valid_spans)].copy()

        print(f"\n[Summary Statistics]")
        print(f" - Total segments: {total_rows}")
        print(f" - Segments with any error: {len(valid_rows_df)}")
        print(f" - High-quality (numeric coordinates): {len(high_quality_df)}")
        
        if high_quality_df.empty:
            print(f"💡 No high-quality samples found for {protocol}.")
            continue

        # 2. Domain & System Distribution
        print(f"\n[Distribution]")
        if 'domainID' in high_quality_df.columns:
            print("Domains:\n", high_quality_df['domainID'].value_counts().to_string())
        if 'systemID' in high_quality_df.columns:
            print("Top Systems:\n", high_quality_df['systemID'].value_counts().head(5).to_string())

        # 3. Content Inspection (Slicing Verification)
        print(f"\n[Sample Inspection - Top {SAMPLE_COUNT}]")
        samples = high_quality_df.head(SAMPLE_COUNT)
        
        for idx, (row_id, row) in enumerate(samples.iterrows()):
            hyp_text = row['hypothesis']
            spans = row[error_col]
            
            print(f"\n--- Sample {idx + 1} (Row Index: {row_id}) ---")
            print(f"MT: {hyp_text[:100]}..." if len(hyp_text) > 100 else f"MT: {hyp_text}")
            
            for s_idx, span in enumerate(spans):
                start = span.get('start_i')
                end = span.get('end_i')
                if str(start).isdigit() and str(end).isdigit():
                    s, e = int(start), int(end)
                    # Use Python string slicing to verify the character-level span
                    extracted = hyp_text[s:e]
                    severity = span.get('severity', 'N/A')
                    err_type = span.get('error_type', 'N/A')
                    print(f"   Error {s_idx+1}: [{s}:{e}] -> \"{extracted}\" | Severity: {severity} | Type: {err_type}")

    except Exception as e:
        print(f"⚠️ An error occurred while processing {protocol}: {e}")

print("\n" + "="*60)
print("Inspection Finished.")