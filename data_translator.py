"""
对抗性共演化系统 - 数据翻译官模块

这是 KnowledgeGraphInterface 接口的具体实现。
这个类是连接"标准业务世界"和"混乱物理数据"的核心桥梁。

V1.0 架构约定：
- 在内存中创建真实的Pandas数据库（模拟）
- 实现KnowledgeGraphInterface中的所有翻译功能
- 将物理表名和列名翻译为标准业务名称
- 提供统一的DataFrame接口用于特征工程
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# 导入"规范"和"DNA"
from knowledge_graph_interface import KnowledgeGraphInterface
from core_structures import FeatureGene


class KnowledgeGraphTranslator(KnowledgeGraphInterface):
    """
    V1.0 "翻译官"的具体实现。
    
    它在内部创建并持有一个"内存数据库"（Pandas DataFrames），
    并负责将标准业务请求"翻译"为对这些DataFrame的访问。

    外部数据注入示例：
    >>> ext_tables = {
    >>>     "my_bank_table": df  # 例如，一张包含原始交易或账户数据的表
    >>> }
    >>> translator = KnowledgeGraphTranslator(
    >>>     inferred_schema=schema_map,
    >>>     physical_target_table="my_bank_table",
    >>>     physical_target_column="label",
    >>>     dataframes=ext_tables
    >>> )
    >>> # 若推断模式中缺失该物理表/列，将自动构造 BankRecord 单实体标准模式继续工作。
    """
    
    def __init__(self, inferred_schema: Dict[str, Any], physical_target_table: str, physical_target_column: str, dataframes: Optional[Dict[str, pd.DataFrame]] = None):
        """
        构造函数。
        注意：它不需要 `raw_db_connection`，因为它将自己创建内存数据库。
        dataframes: 外部注入的物理表集合，键为物理表名，值为DataFrame。若提供，则跳过随机生成。
        """
        print("\n--- \"数据翻译官\"模块启动 ---")
        self.inferred_schema = inferred_schema
        # 当提供外部表时，跳过随机生成，直接使用
        if dataframes is not None:
            print("[翻译官] 检测到外部DataFrame注入，跳过内存数据库随机生成。")
            # 拷贝一份，避免外部引用被修改
            self.db_tables = {name: df.copy() for name, df in dataframes.items()}
            # 最小目标列处理：确保目标列存在并为数值型（int8）
            if physical_target_table in self.db_tables:
                df_tgt = self.db_tables[physical_target_table]
                if physical_target_column not in df_tgt.columns:
                    print(f"[翻译官] 目标列 '{physical_target_column}' 不存在，已在表 '{physical_target_table}' 中以0填充新增。")
                    df_tgt[physical_target_column] = 0
                # 尝试数值化并压缩
                try:
                    ser = pd.to_numeric(df_tgt[physical_target_column], errors='coerce').fillna(0).astype('int8')
                    df_tgt[physical_target_column] = ser
                except Exception:
                    print(f"[翻译官] 警告：目标列 '{physical_target_column}' 无法数值化，保持原始dtype。")
                self.db_tables[physical_target_table] = df_tgt
            else:
                print(f"[翻译官] 警告：未在外部数据中找到目标表 '{physical_target_table}'。")
        else:
            self.db_tables = self._create_in_memory_database(inferred_schema, physical_target_table, physical_target_column)

        # 如果推断模式中找不到目标物理表或目标列映射，则构造 BankRecord Fallback
        entity_name, entity_schema = self._find_entity_by_physical_name(self.inferred_schema, physical_target_table)
        missing_table = entity_schema is None
        missing_target_col = True
        if not missing_table:
            # 检查该实体是否包含目标列映射
            for f_std, f_map in entity_schema.get('fields', {}).items():
                if f_map.get('physical_column') == physical_target_column:
                    missing_target_col = False
                    break

        if missing_table or missing_target_col:
            fb = self._simple_schema_fallback(physical_target_table, physical_target_column)
            if fb:
                print(f"[翻译官] 应用简单模式Fallback：构造单实体 'BankRecord' 指向物理表 '{physical_target_table}'。")
                # 合并到推断模式（不覆盖原有实体）
                self.inferred_schema.update(fb)

        self.standard_schema_cache = self._build_standard_schema_cache()
        print("--- \"数据翻译官\"模块初始化完毕，内存数据库已创建 ---")

    def _find_entity_by_physical_name(self, schema: Dict[str, Any], physical_name: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """辅助函数：根据物理表名查找实体"""
        for entity_name, entity_data in schema.items():
            if entity_data.get('physical_table') == physical_name:
                return entity_name, entity_data
        return None, None

    def _find_physical_col_by_semantic_name(self, entity_schema: Dict[str, Any], semantic_name: str) -> Optional[str]:
        """辅助函数：根据标准字段名查找物理列名"""
        for field_data in entity_schema.get('fields', {}).values():
            if field_data.get('semantic_name') == semantic_name: # 假设 schema 中有 semantic_name
                return field_data.get('physical_column')
        # 回退：如果LLM没生成semantic_name，就按标准字段名本身找
        if semantic_name in entity_schema.get('fields', {}):
             return entity_schema['fields'][semantic_name]['physical_column']
        return None

    def _create_in_memory_database(self, schema: Dict[str, Any], physical_target_table: str, physical_target_column: str) -> Dict[str, pd.DataFrame]:
        """
        [V1.4 核心] 根据\"推断模式\"在内存中创建真实的模拟数据库 (完全动态版本)
        """
        print("[翻译官] 正在创建内存数据库 (V1.4 动态版)...")
        db_tables = {}
        
        num_users = 1000
        num_transactions = 5000

        try:
            # 1. 动态查找用户实体和日志实体
            user_entity_name, user_schema = self._find_entity_by_physical_name(schema, 'tbl_user_01')
            log_entity_name, log_schema = self._find_entity_by_physical_name(schema, 'tbl_log_05')

            if not user_schema or not log_schema:
                raise KeyError("无法在推断模式中找到 'tbl_user_01' 或 'tbl_log_05' 对应的实体。")

            # 2. 动态查找用户表中的列
            user_ids_col = self._find_physical_col_by_semantic_name(user_schema, 'UserID')
            income_col = self._find_physical_col_by_semantic_name(user_schema, 'AnnualIncome')
            gender_col = self._find_physical_col_by_semantic_name(user_schema, 'Gender')
            reg_dt_col = self._find_physical_col_by_semantic_name(user_schema, 'RegistrationDate')
            target_col = physical_target_column # 这个已经是物理名称，直接使用

            # 3. 创建用户表 DataFrame
            user_ids_data = [f"user_{i}" for i in range(num_users)]
            db_tables[user_schema['physical_table']] = pd.DataFrame({
                user_ids_col: user_ids_data,
                income_col: np.random.randint(20000, 250000, size=num_users),
                gender_col: np.random.choice(['M', 'F', 'O'], size=num_users, p=[0.45, 0.45, 0.1]),
                reg_dt_col: pd.to_datetime('2023-01-01') + pd.to_timedelta(np.random.randint(0, 730, size=num_users), 'd'),
                target_col: np.random.choice([0, 1], size=num_users, p=[0.9, 0.1])
            })

            # 4. 动态查找日志表中的列
            txn_id_col = self._find_physical_col_by_semantic_name(log_schema, 'TransactionID') or self._find_physical_col_by_semantic_name(log_schema, 'ActivityID')
            amount_col = self._find_physical_col_by_semantic_name(log_schema, 'TransactionAmount')
            time_col = self._find_physical_col_by_semantic_name(log_schema, 'Timestamp') or self._find_physical_col_by_semantic_name(log_schema, 'ActivityTimestamp')
            fk_col = self._find_physical_col_by_semantic_name(log_schema, 'UserID_FK') or self._find_physical_col_by_semantic_name(log_schema, 'UserIDReference')

            # 5. 创建日志表 DataFrame
            db_tables[log_schema['physical_table']] = pd.DataFrame({
                txn_id_col: [f"txn_{i}" for i in range(num_transactions)],
                amount_col: np.round(np.random.uniform(5.0, 1000.0, size=num_transactions), 2),
                time_col: pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 8760, size=num_transactions), 'h'),
                fk_col: np.random.choice(user_ids_data, size=num_transactions)
            })
            
            print(f"[翻译官] 内存表 '{user_schema['physical_table']}' 和 '{log_schema['physical_table']}' 已创建。")
            return db_tables

        except (KeyError, TypeError) as e:
            print(f"[翻译官] 创建内存数据库失败！处理推断模式时出错：{e}")
            return {}

    def _simple_schema_fallback(self, table_name: str, target_col: str) -> Dict[str, Any]:
        """
        当 inferred_schema 中无法找到对应物理表/列时，基于实际 DataFrame 自动构造一个
        单实体标准模式：实体名为 'BankRecord'，字段标准名=物理列名。
        """
        df = self.db_tables.get(table_name)
        if df is None or df.empty:
            print(f"[翻译官] Fallback 失败：表 '{table_name}' 不存在或为空。")
            return {}
        def infer_type(s: pd.Series) -> str:
            if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
                return 'integer'
            if pd.api.types.is_float_dtype(s):
                return 'number'
            if pd.api.types.is_datetime64_any_dtype(s):
                return 'datetime'
            return 'string'
        fields = {}
        for col in df.columns:
            fields[col] = {
                'physical_column': col,
                'type': infer_type(df[col])
            }
        if target_col in fields:
            fields[target_col]['role'] = 'target'
        return {
            'BankRecord': {
                'physical_table': table_name,
                'fields': fields
            }
        }

    def _build_standard_schema_cache(self) -> Dict[str, List[str]]:
        """辅助函数：从\"推断模式\"构建\"标准模式\"菜单"""
        standard_schema = {}
        for entity_name, entity_data in self.inferred_schema.items():
            standard_schema[entity_name] = list(entity_data['fields'].keys())
        return standard_schema

    # --- 接口实现 ---

    def get_standard_schema(self) -> Dict[str, List[str]]:
        """
        [V1.0 实现] 返回简化的"标准模式"，供"架构师"使用。
        """
        return self.standard_schema_cache

    def get_entity_dataframe(self, entity_name: str) -> pd.DataFrame:
        """
        [V1.0 实现] 获取一个标准业务实体对应的完整DataFrame。
        """
        print(f"  [翻译官] 收到请求: 获取实体 '{entity_name}'...")
        try:
            # 1. 找到物理表
            physical_table_name = self.inferred_schema[entity_name]['physical_table']
            df_raw = self.db_tables[physical_table_name].copy()
            
            # 2. (关键) 重命名列，从"物理名" -> "标准名"
            rename_map = {}
            for std_name, mapping in self.inferred_schema[entity_name]['fields'].items():
                rename_map[mapping['physical_column']] = std_name
            
            df_renamed = df_raw.rename(columns=rename_map)
            print(f"  [翻译官] 返回 '{entity_name}' (来自 {physical_table_name})，{len(df_renamed)} 行。")
            return df_renamed
            
        except KeyError as e:
            print(f"[翻译官-错误] 无法获取实体 '{entity_name}'。失败键：{e}")
            return pd.DataFrame()

    def get_relationship_keys(self) -> Dict[str, str]:
        """
        [V1.0 实现] 获取实体间的"关联键"（外键）。
        """
        # 单表场景：不返回任何关系映射
        if len(getattr(self, "db_tables", {}) or {}) <= 1:
            return {}
        # (这个实现是简化的，假设我们知道要查找的外键)
        relationships = {}
        try:
            # V1.3 修正: 动态查找包含外键的实体
            fk_entity_name = None
            fk_field_std = None
            for entity_name, entity_data in self.inferred_schema.items():
                for field_name, field_data in entity_data.get('fields', {}).items():
                    if field_data.get('type') == 'foreign_key':
                        fk_entity_name = entity_name
                        fk_field_std = field_name
                        break
                if fk_entity_name:
                    break
            
            if fk_entity_name and fk_field_std:
                fk_field_data = self.inferred_schema[fk_entity_name]['fields'][fk_field_std]
                # V1.3 修正: 使用动态实体名称
                relation_name = f"{fk_entity_name}_to_UserProfile"
                pk_entity_name = fk_field_data['references'].split('.')[0]
                pk_field_name = fk_field_data['references'].split('.')[1]
                relationships[relation_name] = {
                    'from_entity': fk_entity_name,
                    'from_key': fk_field_std,
                    'to_entity': pk_entity_name,
                    'to_key': pk_field_name
                }
        except KeyError:
            pass # 没有外键
        
        return relationships

    @staticmethod
    def get_standard_target_info(inferred_schema: Dict[str, Any], physical_table: str, physical_column: str) -> Dict[str, str]:
        """
        [V1.1 实现] 根据物理名称反向查找标准名称。
        """
        for entity_name, entity_data in inferred_schema.items():
            if entity_data.get('physical_table') == physical_table:
                for field_name, field_data in entity_data.get('fields', {}).items():
                    if field_data.get('physical_column') == physical_column:
                        print(f"  [翻译官] 成功将物理目标 '{physical_table}.{physical_column}' 映射到标准目标 '{entity_name}.{field_name}'")
                        return {'entity': entity_name, 'field': field_name}
        # fallback
        print(f"  [翻译官] 未找到精确映射，使用 Fallback: 'BankRecord.{physical_column}' 对应 '{physical_table}.{physical_column}'")
        return {'entity': 'BankRecord', 'field': physical_column}



if __name__ == "__main__":
    # 我们需要先导入"感知"模块，来为"翻译官"提供"推断模式"
    import semantic_inference
    
    print("--- \"数据翻译官\"模块独立测试 ---")

    # 1. 模拟一个数据库连接对象 (它在 V1.0 中没被使用，但 API 需要它)
    class MockDBConnection:
        pass
    
    # 2. (步骤1) 运行"感知"，获取"推断模式"
    schema_map = semantic_inference.run_semantic_inference(MockDBConnection())
    
    # 3. (步骤2) 实例化"翻译官"，传入"推断模式"
    #    (在 __init__ 内部，它会自动创建内存数据库)
    translator = KnowledgeGraphTranslator(inferred_schema=schema_map)
    
    # 4. (步骤3) 测试"翻译官"的API
    print("\n--- 测试 API: get_standard_schema() ---")
    std_schema = translator.get_standard_schema()
    import json
    print(json.dumps(std_schema, indent=2))
    assert 'UserProfile' in std_schema
    
    print("\n--- 测试 API: get_entity_dataframe('UserProfile') ---")
    user_df = translator.get_entity_dataframe('UserProfile')
    print(f"成功获取 'UserProfile' DataFrame，形状: {user_df.shape}")
    print("列名 (应为标准业务名称):")
    print(list(user_df.columns))
    assert 'AnnualIncome' in user_df.columns
    assert 'col_xyz_01' not in user_df.columns # 验证已重命名
    
    print("\n--- 测试 API: get_relationship_keys() ---")
    keys = translator.get_relationship_keys()
    print("获取的关系:")
    print(json.dumps(keys, indent=2))
    assert 'UserTransaction_to_UserProfile' in keys

    print("\n--- \"数据翻译官\"模块测试完毕 ---")
