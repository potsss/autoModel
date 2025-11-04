
"""
对抗性共演化系统 - 语义推断模块

这是系统的"感知"智能体，它将实现V1.0的真实推断逻辑。
它不再返回硬编码的字典，而是通过调用LLM占位符，基于真实的数据库Schema和数据样本
来动态地推断业务语义。

V1.1 架构约定：
- 接收真实的DataFrame作为输入，而不是DB连接。
- 从DataFrame中动态提取Schema和样本。
- 调用LLM(占位符)并返回其推断结果。
"""

from typing import List, Dict, Any
import pandas as pd

# V1.1: llm_infer_schema 现在由 llm_interface 模块管理
from llm_interface import llm_infer_schema


def _get_raw_schema_from_dataframes(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """
    [V1.1 实现] 从真实的DataFrame字典中提取原始的表和列结构。
    """
    print("[语义推断] 正在从DataFrame中提取原始Schema...")
    raw_schema = {name: list(df.columns) for name, df in dataframes.items()}
    print(f"  [语义推断] 发现 {len(raw_schema)} 个表。")
    return raw_schema

def _get_sample_data_from_dataframes(dataframes: Dict[str, pd.DataFrame], sample_size: int = 5) -> Dict[str, Dict[str, list]]:
    """
    [V1.1 实现] 从真实的DataFrame字典中提取少量样本数据。
    样本将以JSON兼容的格式返回，便于发送给LLM。
    """
    print("[语义推断] 正在从DataFrame中提取样本数据...")
    samples = {}
    for name, df in dataframes.items():
        # .to_dict('list') is a good way to get a JSON-like representation
        samples[name] = df.head(sample_size).to_dict('list')
    print(f"  [语义推断] 已为 {len(samples)} 个表提取样本。")
    return samples

def run_semantic_inference(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    [V1.1 实现] 运行完整的"语义推断"流程。
    它现在接收一个 dataframes 字典作为输入，而不是一个数据库连接。
    """
    print("\n--- \"语义推断\"模块启动 (V1.1 DataFrame版) ---")
    
    # 1. 从真实的DataFrame中提取原始的Schema
    raw_schema = _get_raw_schema_from_dataframes(dataframes)
    
    # 2. 从真实的DataFrame中提取样本数据
    sample_data = _get_sample_data_from_dataframes(dataframes)
    
    # 3. (关键) 调用 LLM (当前为占位符) 来进行推断
    #    注意：我们将原始schema和样本数据都传给它
    inferred_schema = llm_infer_schema(
        raw_schema_info=raw_schema, 
        sample_data=sample_data
    )
    
    print("--- \"语义推断\"模块完成 ---")
    return inferred_schema

if __name__ == '__main__':
    print("--- \"语义推断\"模块 (semantic_inference.py) V1.1 独立集成测试 ---")
    
    # 1. 准备模拟的输入 (一个DataFrame字典)
    mock_dataframes = {
        "bank_data": pd.DataFrame({
            "age": [30, 45, 22],
            "job": ["admin.", "services", "student"],
            "marital": ["married", "single", "single"],
            "y": ["no", "yes", "no"]
        }),
        "other_data": pd.DataFrame({
            "user_id": [1, 2, 3],
            "last_login": ["2024-01-01", "2024-02-15", "2024-03-10"]
        })
    }
    
    # 2. 运行推断流程
    schema = run_semantic_inference(mock_dataframes)
    
    # 3. 验证输出
    print("\n--- 推断结果验证 ---")
    print("生成的Schema (部分):")
    print(f"  UserProfile实体物理表: {schema.get('UserProfile', {}).get('physical_table')}")
    print(f"  UserProfile实体的字段数: {len(schema.get('UserProfile', {}).get('fields', []))}")
    
    assert "UserProfile" in schema, "Schema中应包含UserProfile实体"
    # 注意：由于LLM mock返回的是固定的tbl_user_01, 我们这里断言这个固定值
    assert schema['UserProfile']['physical_table'] == 'tbl_user_01', "物理表映射不正确"
    
    print("--- \"语义推断\"模块 V1.1 独立测试完毕 ---")