from ESA.annotation_loader import AnnotationLoader
import pandas as pd

# 1. 加载数据 (这一步就是加载那个包)
loader = AnnotationLoader(refresh_cache=False)
# 这一行会把 ESA 和 MQM 的数据都加载进来，并合并成一个大的 Excel 表格
df = loader.get_view(["ESA-1", "MQM-1"], only_overlap=False)

# 2. 打印看看这表格有多少列，有多少行
print(f"数据总共有 {len(df)} 行，{len(df.columns)} 列。")

# 3. 看看前几列的名字，确认你拿到了哪些字段
print("\n--- 列名列表 ---")
print(df.columns.tolist())

# 4. 重点来了：查看前 5 行数据的“关键内容”
# 我们只挑几个最重要的列看：source (原文), hypothesis (译文), ESA-1_error_spans (错误标注)
print("\n--- 看看数据长什么样 ---")
pd.set_option('display.max_colwidth', 50) # 设置显示宽度，防止太长看不清
print(df[['source', 'hypothesis', 'ESA-1_error_spans']].head(5))

# 5. 深入看看一个有错误标注的样本
# 过滤出有 ESA 标注的行
error_df = df[df['ESA-1_error_spans'].notna()]
if not error_df.empty:
    print("\n--- 这是一个真实的错误标注样例 ---")
    sample = error_df.iloc[0]
    print(f"原文: {sample['source']}")
    print(f"译文: {sample['hypothesis']}")
    print(f"错误标注: {sample['ESA-1_error_spans']}")

# 筛选出有明确位置的错误
def has_valid_spans(spans):
    if not isinstance(spans, list): return False
    for s in spans:
        # 如果 start_i 和 end_i 都是数字，那这就是我们要的
        if str(s.get('start_i')).isdigit() and str(s.get('end_i')).isdigit():
            return True
    return False

# 过滤数据
valid_error_df = error_df[error_df['ESA-1_error_spans'].apply(has_valid_spans)]

print(f"总错误样本数: {len(error_df)}")
print(f"坐标明确的样本数: {len(valid_error_df)}")

if len(valid_error_df) > 0:
    print("\n看一个完美的样本:")
    print(valid_error_df['ESA-1_error_spans'].iloc[0])