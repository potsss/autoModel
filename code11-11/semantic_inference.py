"""
对抗性共演化系统 - 语义推断模块

这是系统的"感知"智能体，它将实现V1.0的真实推断逻辑。
它不再返回硬编码的字典，而是通过调用LLM占位符，基于真实的数据库Schema和数据样本
来动态地推断业务语义。

V1.1 架构约定：
- 接收真实的DataFrame作为输入，而不是DB连接。
- 从DataFrame中动态提取Schema和样本。
- 调用LLM(占位符)并返回其推断结果。

V1.2 更新：
- 支持外部Schema配置，避免LLM幻觉问题
- 优先使用 schema_config.json，回退到LLM推断
"""

from typing import List, Dict, Any, Optional, TYPE_CHECKING
import pandas as pd
import json

# V1.1: llm_infer_schema 现在由 llm_interface 模块管理
try:
    from llm_interface import llm_infer_schema
except Exception as e:
    llm_infer_schema = None
    print(f"[语义推断][警告] 未找到 llm_interface.llm_infer_schema，将使用回退推断。原因: {e}")

# V1.2: 新增导入
try:
    from schema_config import SchemaConfig
except ImportError:
    SchemaConfig = None
    print("[语义推断][警告] 未找到 schema_config 模块，将只能使用LLM推断")

if TYPE_CHECKING:
    # 仅用于类型检查，避免Pylance黄线
    from schema_config import SchemaConfig as _SchemaConfig


def _get_raw_schema_from_dataframes(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    [V1.1 实现] 从真实的DataFrame字典中提取原始的表和列结构。
    """
    print("[语义推断] 正在从DataFrame中提取原始Schema...")
    raw_schema = {
        name: {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        for name, df in dataframes.items()
    }
    print(f"  [语义推断] 发现 {len(raw_schema)} 个表。")
    return raw_schema

def _get_sample_data_from_dataframes(dataframes: Dict[str, pd.DataFrame], sample_size: int = 5) -> Dict[str, Any]:
    """
    [V1.1 实现] 从真实的DataFrame字典中提取少量样本数据。
    样本将以JSON兼容的格式返回，便于发送给LLM。
    """
    print("[语义推断] 正在从DataFrame中提取样本数据...")
    samples = {}
    for name, df in dataframes.items():
        try:
            samples[name] = df.head(sample_size).to_dict(orient='records')
        except Exception as e:
            print(f"[语义推断][警告] 提取样本失败: 表={name}, 错误={e}")
            samples[name] = []
    print(f"  [语义推断] 已为 {len(samples)} 个表提取样本。")
    return samples


def run_semantic_inference(
    dataframes: Dict[str, pd.DataFrame],
    schema_config: Optional[Any] = None,  # SchemaConfig | None
    autofill_fields: bool = True
) -> Dict[str, Any]:
    """
    [V1.2 实现] 运行完整的"语义推断"流程。
    它现在接收一个 dataframes 字典作为输入，而不是一个数据库连接。
    """
    print("\n--- \"语义推断\"模块启动 (V1.2 DataFrame版) ---")

    # --- V1.2 新增: 优先使用外部Schema配置 ---
    if schema_config and getattr(schema_config, "schema", None):
        print("[语义推断] 检测到外部Schema配置,优先使用外部配置...")
        try:
            config_tables = set(schema_config.get_all_tables().keys())
        except Exception:
            config_tables = set()
        actual_tables = set(dataframes.keys())

        missing_in_data = config_tables - actual_tables
        missing_in_config = actual_tables - config_tables

        if missing_in_data:
            print(f"[语义推断][警告] 配置中定义但数据中缺失的表: {missing_in_data}")
        if missing_in_config:
            print(f"[语义推断][警告] 数据中存在但配置中未定义的表: {missing_in_config}")
            print("[语义推断] 这些表将使用LLM/回退自动推断...")

        # 使用外部配置
        try:
            schema_map = schema_config.to_semantic_schema()
        except Exception as e:
            print(f"[语义推断][警告] 外部配置转换失败，将回退到自动推断: {e}")
            schema_map = {}

        # 只保留实际存在的表
        schema_map = {k: v for k, v in schema_map.items() if k in actual_tables}
        print(f"[语义推断] 已从外部配置加载 {len(schema_map)} 个表的Schema")
        
        # [新增] 将schema_config的relationships也传递到schema_map中
        try:
            relationships = schema_config.get_relationships()
            if relationships:
                schema_map['relationships'] = relationships
                print(f"[语义推断] 已加载 {len(relationships)} 条关系映射到schema_map")
        except Exception as e:
            print(f"[语义推断][警告] 加载relationships失败: {e}")

        # --- 新增: 将实际DF中的缺失列自动补全到配置生成的schema_map（可开关） ---
        if autofill_fields:
            added_fields_total = 0
            for tbl, df in dataframes.items():
                if tbl in schema_map:
                    fields = schema_map[tbl].get("fields", {}) or {}
                    before = len(fields)
                    for col in df.columns:
                        if col not in fields:
                            fields[col] = {
                                "type": str(df[col].dtype),
                                "description": f"字段 {col} (auto-filled: 未在schema配置中定义)"
                            }
                    schema_map[tbl]["fields"] = fields
                    added = len(fields) - before
                    if added > 0:
                        print(f"[语义推断] 已为表 '{tbl}' 自动补全 {added} 个字段")
                        added_fields_total += added
            if added_fields_total:
                print(f"[语义推断] 字段自动补全完成，共新增 {added_fields_total} 个字段")
        # --- 新增结束 ---

        # 对于配置中缺失的表,使用LLM/回退推断
        if missing_in_config:
            print(f"[语义推断] 正在为 {len(missing_in_config)} 个未配置的表进行自动推断...")
            missing_dataframes = {k: v for k, v in dataframes.items() if k in missing_in_config}
            llm_schema = _run_llm_inference(missing_dataframes)
            schema_map.update(llm_schema)

        print("--- \"语义推断\"模块完成 (使用外部配置) ---\n")
        return schema_map

    # --- V1.1 原有逻辑: 使用LLM推断 ---
    print("[语义推断] 未提供外部Schema配置,使用LLM/回退自动推断...")
    schema_map = _run_llm_inference(dataframes)
    print("--- \"语义推断\"模块完成 (自动推断) ---\n")
    return schema_map


def _run_llm_inference(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    [V1.2 新增] 使用LLM进行Schema推断(从原有逻辑提取为独立函数)
    """
    # 1) 从真实的DataFrame中提取原始的Schema与样本
    raw_schema = _get_raw_schema_from_dataframes(dataframes)
    sample_data = _get_sample_data_from_dataframes(dataframes)

    # 2) 调用 LLM (若不可用则回退)
    inferred_schema: Dict[str, Any]
    if callable(llm_infer_schema):
        try:
            inferred_schema = llm_infer_schema(
                raw_schema_info=raw_schema,
                sample_data=sample_data
            )
        except Exception as e:
            print(f"[语义推断][警告] 调用LLM推断失败, 使用回退: {e}")
            inferred_schema = _create_fallback_schema(dataframes)
    else:
        inferred_schema = _create_fallback_schema(dataframes)

    # 3) 清理可能的幻觉表与关系
    actual_tables = set(dataframes.keys())
    if isinstance(inferred_schema, dict):
        hallucinated_tables = set(inferred_schema.keys()) - actual_tables
        if hallucinated_tables:
            print(f"[语义推断][警告] 推断结果包含不存在的表: {hallucinated_tables}，正在清理...")
            for t in hallucinated_tables:
                inferred_schema.pop(t, None)

        for table_name, table_info in inferred_schema.items():
            rels = table_info.get("relationships", []) if isinstance(table_info, dict) else []
            valid_rels = []
            for rel in rels:
                target_table = (rel or {}).get("to_entity")
                if target_table in actual_tables:
                    valid_rels.append(rel)
                else:
                    if target_table is not None:
                        print(f"[语义推断][清理] 移除 '{table_name}' 指向不存在表 '{target_table}' 的关系")
            if isinstance(table_info, dict):
                table_info["relationships"] = valid_rels

    print(f"[语义推断] 清理完成,最终保留 {len(inferred_schema)} 个表")
    return inferred_schema


def _create_fallback_schema(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    创建回退Schema (当LLM不可用或失败时使用)
    """
    print("[语义推断] 正在构建回退Schema...")
    schema_map: Dict[str, Any] = {}
    for table_name, df in dataframes.items():
        fields = {}
        for col in df.columns:
            fields[col] = {
                "type": str(df[col].dtype),
                "description": f"字段 {col}"
            }
        schema_map[table_name] = {
            "entity": table_name,
            "description": f"表 {table_name}",
            "primary_key": None,
            "fields": fields,
            "relationships": []
        }
    return schema_map


if __name__ == '__main__':
    # 轻量自测入口（可忽略）
    demo = {
        "application_train": pd.DataFrame({"SK_ID_CURR": [1, 2], "TARGET": [0, 1]}),
        "bureau": pd.DataFrame({"SK_ID_CURR": [1, 1], "AMT_CREDIT_SUM_DEBT": [1000.0, 500.0]})
    }
    result = run_semantic_inference(demo, schema_config=None)
    print(json.dumps({k: list(v.keys()) for k, v in result.items()}, ensure_ascii=False, indent=2))