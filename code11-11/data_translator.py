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
    
    def __init__(self, inferred_schema: Dict[str, Any], physical_target_table: str, physical_target_column: str, dataframes: Optional[Dict[str, pd.DataFrame]] = None, disable_entity_fallback: bool = False):
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

        # 如果允许，才执行实体/目标列的兜底构造
        if not disable_entity_fallback:
            # 如果推断模式中找不到目标物理表或目标列映射，则构造简单Fallback
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
                    # 取回刚生成的实体名（即物理表名）
                    fb_entity_name = list(fb.keys())[0]
                    print(f"[翻译官][提示] 应用简单模式Fallback：构造单实体 '{fb_entity_name}' 指向物理表 '{physical_target_table}'。")
                    # 合并到推断模式（不覆盖原有实体）
                    self.inferred_schema.update(fb)

        self.entity_map = {}
        for entity_data in self.inferred_schema.get('entities', []):
            entity_name = entity_data.get('name')
            if entity_name:
                self.entity_map[entity_name] = {
                    "physical_table": entity_data.get('physical_table', list(self.db_tables.keys())[0] if self.db_tables else 'bank'),
                    "fields": {
                        col: {"physical_column": col} for col in entity_data.get('columns', [])
                    }
                }
        
        # [修复] 构建标准Schema缓存，供LLM基因生成使用
        self.standard_schema_cache = self._build_standard_schema_cache()
        
        print("--- \"数据翻译官\"模块初始化完毕，内存数据库已创建 ---")

    def _find_entity_by_physical_name(self, schema, physical_name):
        """根据物理表名查找标准实体"""
        for entity_data in schema.get('entities', []):
            # This is a simplification, assuming physical_table is part of the entity name or a property.
            if physical_name in entity_data.get('name', '').lower() or physical_name == entity_data.get('physical_table'):
                return entity_data.get('name'), entity_data
        # Fallback for V1.1
        print(f"[翻译官] 找不到与物理表 '{physical_name}' 完全匹配的实体，将使用启发式规则。")
        # 返回物理表名作实体名，占位的空schema（会触发后续fallback补全）
        return physical_name, {"fields": {}}

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
            target_col = self._find_physical_col_by_semantic_name(user_schema, 'IsDefault') or physical_target_column

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
        # 使用物理表名作为标准实体名，避免历史上的 'BankRecord' 命名残留
        return {
            table_name: {
                'physical_table': table_name,
                'fields': fields
            }
        }

    def _build_standard_schema_cache(self):
        """
        [V1.1] 创建一个简化的标准 Schema 缓存，用于基因生成。
        格式: { "EntityName": ["field1", "field2"], ... }
        """
        standard_schema = {}
        
        # 方式1: 从 inferred_schema 的 entities 列表读取
        for entity_data in self.inferred_schema.get('entities', []):
            entity_name = entity_data.get('name')
            if entity_name and 'columns' in entity_data:
                standard_schema[entity_name] = entity_data['columns']
        
        # 方式2: 从 entity_map 读取(兼容Fallback模式)
        if not standard_schema and hasattr(self, 'entity_map'):
            for entity_name, entity_info in self.entity_map.items():
                fields = list(entity_info.get('fields', {}).keys())
                if fields:
                    standard_schema[entity_name] = fields
        
        # 方式3: 直接从物理表读取(最后兜底)
        if not standard_schema and hasattr(self, 'db_tables'):
            for table_name, df in self.db_tables.items():
                # 使用表名作为实体名
                standard_schema[table_name] = df.columns.tolist()
        
        return standard_schema

    # --- 接口实现 ---

    def get_standard_schema(self) -> Dict[str, List[str]]:
        """
        [V1.0 实现] 返回简化的"标准模式"，供"架构师"使用。
        """
        return self.standard_schema_cache

    def get_entity_dataframe(self, entity_name: str) -> pd.DataFrame:
        """
        [V1.5 实现] 获取一个标准业务实体对应的完整DataFrame。
        支持通过实体名或物理表名查询。
        """
        print(f"  [翻译官] 收到请求: 获取实体 '{entity_name}'...")
        try:
            # 1. 尝试直接通过实体名查找
            if entity_name in self.inferred_schema:
                physical_table_name = self.inferred_schema[entity_name].get('physical_table')
                if physical_table_name and physical_table_name in self.db_tables:
                    df_raw = self.db_tables[physical_table_name].copy()
                    
                    # 2. (关键) 重命名列，从"物理名" -> "标准名"
                    rename_map = {}
                    for std_name, mapping in self.inferred_schema[entity_name]['fields'].items():
                        phys_col = mapping.get('physical_column', std_name)
                        if phys_col in df_raw.columns:
                            rename_map[phys_col] = std_name
                    
                    df_renamed = df_raw.rename(columns=rename_map)
                    print(f"  [翻译官] 返回 '{entity_name}' (来自 {physical_table_name})，{len(df_renamed)} 行。")
                    return df_renamed
            
            # 2. 尝试通过物理表名反查实体名
            found_entity = self._find_entity_name_by_physical_table(entity_name)
            if found_entity and found_entity in self.inferred_schema:
                physical_table_name = self.inferred_schema[found_entity].get('physical_table', entity_name)
                if physical_table_name in self.db_tables:
                    df_raw = self.db_tables[physical_table_name].copy()
                    
                    rename_map = {}
                    for std_name, mapping in self.inferred_schema[found_entity]['fields'].items():
                        phys_col = mapping.get('physical_column', std_name)
                        if phys_col in df_raw.columns:
                            rename_map[phys_col] = std_name
                    
                    df_renamed = df_raw.rename(columns=rename_map)
                    print(f"  [翻译官] 返回 '{found_entity}' (通过物理表名 {entity_name} 反查，实际表: {physical_table_name})，{len(df_renamed)} 行。")
                    return df_renamed
            
            # 3. 兜底：直接把entity_name当作物理表名
            if entity_name in self.db_tables:
                df_raw = self.db_tables[entity_name].copy()
                print(f"  [翻译官-兜底] 直接返回物理表 '{entity_name}'，{len(df_raw)} 行（未重命名列）。")
                return df_raw
            
            print(f"[翻译官-错误] 无法获取实体 '{entity_name}'，未找到对应的物理表。")
            return pd.DataFrame()
            
        except Exception as e:
            print(f"[翻译官-错误] 获取实体 '{entity_name}' 时发生异常: {e}")
            return pd.DataFrame()

    def get_relationship_keys(self) -> Dict[str, str]:
        """
        [V1.5 实现] 获取实体间的"关联键"（外键）。
        优先从外部schema_config读取relationships配置，实现完整的跨表映射。
        """
        relationships = {}
        
        # [新增] 优先尝试从外部schema_config读取relationships
        if hasattr(self, 'inferred_schema') and isinstance(self.inferred_schema, dict):
            # 检查是否有schema_config提供的relationships配置
            schema_relationships = self.inferred_schema.get('relationships', [])
            if schema_relationships:
                print(f"[翻译官] 从schema_config读取到 {len(schema_relationships)} 条关系映射")
                for rel in schema_relationships:
                    try:
                        # 解析关系配置
                        from_table = rel.get('from_table')
                        from_field = rel.get('from_field')
                        to_table = rel.get('to_table')
                        to_field = rel.get('to_field')
                        rel_type = rel.get('type', 'many_to_one')
                        
                        if not all([from_table, from_field, to_table, to_field]):
                            continue
                        
                        # 查找对应的实体名称（物理表名 -> 标准实体名）
                        from_entity = self._find_entity_name_by_physical_table(from_table)
                        to_entity = self._find_entity_name_by_physical_table(to_table)
                        
                        if not from_entity or not to_entity:
                            print(f"[翻译官-警告] 无法映射关系中的实体: {from_table} -> {to_table}")
                            continue
                        
                        # 生成关系键名
                        relation_name = f"{from_entity}_to_{to_entity}"
                        relationships[relation_name] = {
                            'from_entity': from_entity,
                            'from_key': from_field,
                            'to_entity': to_entity,
                            'to_key': to_field,
                            'type': rel_type,
                            'description': rel.get('description', ''),
                            'physical_from_table': from_table,
                            'physical_to_table': to_table
                        }
                        print(f"[翻译官] 已加载关系: {relation_name} ({from_table}.{from_field} -> {to_table}.{to_field})")
                    except Exception as e:
                        print(f"[翻译官-警告] 解析关系配置失败: {e}")
                        continue
                
                if relationships:
                    return relationships
        
        # [兜底] 单表场景：不返回任何关系映射
        if len(getattr(self, "db_tables", {}) or {}) <= 1:
            print("[翻译官] 单表场景，无跨表关系")
            return {}
        
        # [兜底] 尝试从inferred_schema的fields中查找外键
        try:
            for entity_name, entity_data in self.inferred_schema.items():
                if entity_name in ['relationships', 'entities', 'tables']:
                    continue
                for field_name, field_data in entity_data.get('fields', {}).items():
                    if field_data.get('type') == 'foreign_key' or field_data.get('is_foreign_key'):
                        fk_ref = field_data.get('references', {})
                        if isinstance(fk_ref, dict):
                            pk_table = fk_ref.get('table')
                            pk_field = fk_ref.get('field')
                        elif isinstance(fk_ref, str) and '.' in fk_ref:
                            pk_table, pk_field = fk_ref.split('.', 1)
                        else:
                            continue
                        
                        pk_entity = self._find_entity_name_by_physical_table(pk_table)
                        if not pk_entity:
                            pk_entity = pk_table
                        
                        relation_name = f"{entity_name}_to_{pk_entity}"
                        relationships[relation_name] = {
                            'from_entity': entity_name,
                            'from_key': field_name,
                            'to_entity': pk_entity,
                            'to_key': pk_field
                        }
                        print(f"[翻译官-兜底] 从字段推断关系: {relation_name}")
        except Exception as e:
            print(f"[翻译官-警告] 兜底关系推断失败: {e}")
        
        return relationships
    
    def _find_entity_name_by_physical_table(self, physical_table: str) -> Optional[str]:
        """
        根据物理表名查找对应的标准实体名称
        """
        # 方式1: 从entities列表中查找
        for entity_data in self.inferred_schema.get('entities', []):
            if entity_data.get('physical_table') == physical_table:
                return entity_data.get('name')
        
        # 方式2: 从tables配置中查找
        tables_config = self.inferred_schema.get('tables', {})
        if physical_table in tables_config:
            return tables_config[physical_table].get('entity_name', physical_table)
        
        # 方式3: 直接从顶层schema查找
        for entity_name, entity_data in self.inferred_schema.items():
            if isinstance(entity_data, dict) and entity_data.get('physical_table') == physical_table:
                return entity_name
        
        # 兜底: 使用物理表名本身
        return physical_table

    @staticmethod
    def get_standard_target_info(inferred_schema: Dict[str, Any], physical_table: str, physical_column: str) -> Dict[str, str]:
        """
        [V1.1 实现] 根据物理名称反向查找标准名称。
        """
        # 方式1：从 entities 列表查找
        for entity_data in inferred_schema.get('entities', []):
            entity_name = entity_data.get('name')
            # 简化匹配：实体名包含物理表名，或实体记录有 physical_table 并等于之
            if physical_table in entity_data.get('name', '').lower() or physical_table == entity_data.get('physical_table'):
                for field_name in entity_data.get('columns', []):
                    if field_name.lower() == physical_column.lower():
                        print(f"  [翻译官] 成功将物理目标 '{physical_table}.{physical_column}' 映射到标准目标 '{entity_name}.{field_name}'")
                        return {'entity': entity_name, 'field': field_name}

        # 方式2：未找到时，直接采用物理表名作为标准实体名（与Fallback策略一致）
        print(f"  [翻译官][提示] 未找到精确映射，已采用标准名 '{physical_table}.{physical_column}' 对应物理 '{physical_table}.{physical_column}'")
        return {'entity': physical_table, 'field': physical_column}



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
