"""
Schema配置管理模块

负责从外部JSON文件加载表结构和关系定义,
为语义推断模块提供准确的Schema信息,避免LLM的"幻觉"问题。

功能:
- 加载和验证schema_config.json
- 提供表信息、关系、业务规则的查询接口
- 转换为semantic_inference兼容的格式
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

class SchemaConfig:
    """Schema配置管理器"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        初始化Schema配置管理器
        
        Args:
            config_path: schema_config.json文件路径,如果为None则不加载
        """
        self.config_path = config_path
        self.schema = None
        
        if config_path and config_path.exists():
            self.load_from_file(config_path)
    
    def load_from_file(self, path: Path) -> Dict[str, Any]:
        """
        从JSON文件加载Schema配置
        
        Args:
            path: JSON配置文件路径
            
        Returns:
            加载的完整schema字典
        """
        print(f"[Schema配置] 正在从 '{path.name}' 加载配置...")
        with open(path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        
        # 验证必要字段
        self._validate_schema()
        
        print(f"[Schema配置] 配置加载成功:")
        print(f"  - 数据集: {self.schema.get('dataset_info', {}).get('name', 'Unknown')}")
        print(f"  - 表数量: {len(self.schema.get('tables', {}))}")
        print(f"  - 关系数量: {len(self.schema.get('relationships', []))}")
        print(f"  - 业务规则数量: {len(self.schema.get('business_rules', []))}")
        
        return self.schema
    
    def _validate_schema(self):
        """验证Schema配置的完整性"""
        if not self.schema:
            raise ValueError("Schema配置为空")
        
        required_keys = ['tables', 'relationships']
        for key in required_keys:
            if key not in self.schema:
                raise ValueError(f"Schema配置缺少必需字段: {key}")
        
        # 检查关系中引用的表是否都存在
        tables = set(self.schema.get('tables', {}).keys())
        for rel in self.schema.get('relationships', []):
            if rel['from_table'] not in tables:
                raise ValueError(f"关系中引用的表 '{rel['from_table']}' 不存在")
            if rel['to_table'] not in tables:
                raise ValueError(f"关系中引用的表 '{rel['to_table']}' 不存在")
        
        print("[Schema配置] 配置验证通过 ✓")
    
    def get_table_info(self, table_name: str) -> Optional[Dict]:
        """
        获取指定表的完整信息
        
        Args:
            table_name: 表名
            
        Returns:
            表的配置字典,如果表不存在则返回None
        """
        if not self.schema:
            return None
        return self.schema.get('tables', {}).get(table_name)
    
    def get_all_tables(self) -> Dict[str, Dict]:
        """获取所有表的配置"""
        if not self.schema:
            return {}
        return self.schema.get('tables', {})
    
    def get_relationships(self, table_name: Optional[str] = None) -> List[Dict]:
        """
        获取关系列表,可按表名筛选
        
        Args:
            table_name: 如果指定,则只返回与该表相关的关系
            
        Returns:
            关系列表
        """
        if not self.schema:
            return []
        
        rels = self.schema.get('relationships', [])
        
        if table_name:
            return [r for r in rels 
                   if r['from_table'] == table_name or r['to_table'] == table_name]
        
        return rels
    
    def get_foreign_keys(self, table_name: str) -> Dict[str, Dict]:
        """
        获取表的所有外键字段
        
        Args:
            table_name: 表名
            
        Returns:
            {字段名: {references: {table: ..., field: ...}}}
        """
        table_info = self.get_table_info(table_name)
        if not table_info:
            return {}
        
        fks = {}
        for field_name, field_info in table_info.get('fields', {}).items():
            if field_info.get('is_foreign_key'):
                fks[field_name] = field_info.get('references', {})
        
        return fks
    
    def to_semantic_schema(self) -> Dict[str, Any]:
        """
        转换为semantic_inference兼容的schema格式
        
        这个方法将schema_config.json的格式转换为
        semantic_inference模块期望的schema_map格式
        
        Returns:
            符合semantic_inference格式的schema字典
        """
        if not self.schema:
            return {}
        
        schema_map = {}
        tables = self.schema.get('tables', {})
        
        for table_name, table_info in tables.items():
            # 构建字段映射
            fields = {}
            for field_name, field_info in table_info.get('fields', {}).items():
                fields[field_name] = {
                    "type": field_info.get('type', 'unknown'),
                    "description": field_info.get('description', ''),
                    "business_meaning": field_info.get('business_meaning', '')
                }
            
            # 构建关系列表
            relationships = []
            for rel in self.get_relationships(table_name):
                if rel['from_table'] == table_name:
                    relationships.append({
                        "to_entity": rel['to_table'],
                        "via_field": rel['from_field'],
                        "type": rel['type'],
                        "description": rel.get('description', '')
                    })
            
            # 组装表信息
            schema_map[table_name] = {
                "entity": table_info.get('entity_name', table_name),
                "description": table_info.get('description', ''),
                "primary_key": table_info.get('primary_key'),
                "fields": fields,
                "relationships": relationships
            }
        
        return schema_map
    
    def get_business_rules(self) -> List[Dict]:
        """获取所有业务规则"""
        if not self.schema:
            return []
        return self.schema.get('business_rules', [])
    
    def get_feature_hints(self) -> List[Dict]:
        """获取特征工程提示"""
        if not self.schema:
            return []
        return self.schema.get('feature_engineering_hints', [])
    
    def get_target_info(self) -> Dict[str, str]:
        """获取目标变量信息"""
        if not self.schema:
            return {}
        return self.schema.get('target', {})