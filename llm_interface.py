"""
对抗性共演化系统 - 真实 LLM 接口模块

本模块负责与真实的大语言模型 API 进行交互。
它替换了原有的 llm_placeholders.py，提供了生产级的实现。

V1.1 架构约定：
- 使用 OpenAI SDK 格式与一个兼容的 LLM 端点进行通信。
- 集中管理 API Client 的初始化。
- 为系统中的所有 LLM 调用提供健壮的实现，包含 prompt 工程、API 调用和错误处理。
"""

import os
import json
from openai import OpenAI
from typing import List, Dict, Any, Optional
import pandas as pd


# --- 客户端初始化 ---
# 使用您提供的真实 LLM 接口信息
# 注意: 为简化操作，此处直接使用您提供的信息。在生产环境中，强烈建议使用环境变量来管理密钥。
client = OpenAI(
    base_url="http://112.51.6.147:8003/v1",
    api_key="zmccdictbigdata"
)
MODEL_NAME = "qwen3-32b-awq-local"

def _call_llm(prompt: str, is_json: bool = True) -> Optional[str]:
    """通用 LLM 调用函数"""
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response_format = {"type": "json_object"} if is_json else {"type": "text"}

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            # response_format=response_format # 如果您的模型不支持此参数，请注释掉此行
        )
        return response.choices[0].message.content
    except ConnectionError as e:
        print(f"[LLM Real - 错误] API 连接失败: {e}")
        return None
    except TypeError as e:
        print(f"[LLM Real - 错误] API 调用时发生类型错误: {e}")
        return None
    except Exception as e:
        print(f"[LLM Real - 错误] API 调用失败: {e}")
        return None

def llm_infer_schema(raw_schema_info: Dict[str, Any], sample_data: Dict[str, Any], num_tables: int) -> Dict[str, Any]:
    """
    [真实实现] 使用 LLM 进行"语义推断"。
    V1.2: 根据表的数量动态生成Prompt。
    """
    print("\n[LLM Real] 正在调用真实 LLM API 进行\"语义推断\"...")
    
    # 根据 num_tables 动态构建 prompt
    if num_tables == 1:
        table_name = list(raw_schema_info.keys())[0]
        prompt = f"""
        作为一名数据架构师，请根据以下单张表的结构和数据样本，推断出一个业务层面的逻辑“实体”。

        **重要指令**:
        1.  因为只有一张表，你必须只创建一个逻辑实体。
        2.  将这张表的所有列都归入这一个实体中。
        3.  实体名称应能概括表的业务含义 (例如 "User" 或 "Transaction")。
        4.  `physical_table` 必须是 '{table_name}'。
        5.  不要创建任何 `relationships`，返回一个空列表 `[]`。

        这是我原始的数据库表和列:
        {json.dumps(raw_schema_info, indent=2)}

        这是表的一些数据样本:
        {json.dumps(sample_data, indent=2)}

        请严格按照以下JSON格式输出，不要包含任何额外的解释或Markdown标记：
        {{
          "entities": [
            {{
              "name": "EntityName",
              "primary_key": "some_id_column",
              "physical_table": "{table_name}",
              "columns": ["column1", "column2", "..."]
            }}
          ],
          "relationships": []
        }}
        """
    else:
        # 原有的多表 prompt
        prompt = f"""
        作为一名数据架构师，请根据原始的数据库表结构和数据样本，推断出业务层面的逻辑“实体”和它们之间的“关系”。

        这是我原始的数据库表和列:
        {json.dumps(raw_schema_info, indent=2)}

        这是每个表的一些数据样本:
        {json.dumps(sample_data, indent=2)}

        请为我生成一个描述业务逻辑的JSON。这个JSON应该包含：
        1.  `entities`: 一个实体列表。每个实体应有 `name` (例如 "UserProfile")，`primary_key`，以及 `physical_table` (对应原始表名)。
        2.  `relationships`: 一个关系列表。每个关系应描述两个实体如何通过外键关联，包括 `from_entity`, `to_entity`, `from_column`, `to_column`。

        请严格按照以下JSON格式输出，不要包含任何额外的解释或Markdown标记：
        {{
          "entities": [
            {{
              "name": "EntityName",
              "primary_key": "id",
              "physical_table": "original_table_name",
              "columns": ["id", "field1", "field2"]
            }}
          ],
          "relationships": [
            {{
              "from_entity": "Entity1",
              "to_entity": "Entity2",
              "from_column": "entity2_id",
              "to_column": "id"
            }}
          ]
        }}
        """
        
    response_str = _call_llm(prompt)
    if response_str:
        try:
            clean_response_str = response_str.strip().replace('```json', '').replace('```', '')
            schema_dict = json.loads(clean_response_str)
            print(f"  (真实 LLM 已成功返回推断的模式)")
            return schema_dict
        except json.JSONDecodeError:
            print(f"  (真实 LLM 返回了无效的JSON, 内容: {response_str})")
            # Fallback to a simple schema if LLM fails
            return { "entities": [], "relationships": [] }
    return { "entities": [], "relationships": [] }

def llm_generate_genes(
    standard_schema: Dict[str, List[str]], 
    target_variable: str
) -> List[Dict[str, Any]]:
    """
    [真实实现] 使用 LLM 进行"基因创生"（头脑风暴）。
    """
    print("\n[LLM Real] 正在调用真实 LLM API 进行\"基因创生\"...")
    
    # 根据 schema 中的实体数量动态构建 prompt
    num_entities = len(standard_schema)
    
    if num_entities == 1:
        # 单表场景的 prompt
        entity_name = list(standard_schema.keys())[0]
        prompt = f"""
        作为一名顶尖的数据科学家，请为我进行特征工程的头脑风暴。

        我的目标是预测目标变量: `{target_variable}`

        我只有一个数据实体（表）可用，它的名字是 `{entity_name}`，数据模式如下:
        {json.dumps(standard_schema, indent=2)}

        **重要提示：你必须严格从上述提供的唯一实体中选择字段。所有特征的 `op` 类型必须是 'LATEST'。**

        请为我生成一个包含至少10个特征工程创意的JSON列表。每个创意是一个“基因”，代表一种直接从表中取值的特征。

        请严格按照以下JSON格式输出一个列表，不要包含任何额外的解释或Markdown标记：
        [
          {{
            "op": "LATEST",
            "path": "{entity_name}.FieldName"
          }}
        ]
        """
    else:
        # 多表场景的 prompt (原始 prompt)
        prompt = f"""
        作为一名顶尖的数据科学家，请为我进行特征工程的头脑风暴。

        我的目标是预测目标变量: `{target_variable}`

        这是我可用的、已经标准化的数据模式:
        {json.dumps(standard_schema, indent=2)}

        **重要提示：你必须严格遵守提供的Schema。只使用Schema中存在的实体和字段。不要虚构任何新的实体或关系。**

        请为我生成一个包含至少10个特征工程创意的JSON列表。每个创意是一个“基因”，代表一种特征提取方法。
        基因的类型可以是：
        1.  直接特征 (op: 'LATEST'): 直接从主实体获取。
        2.  聚合特征 (op: 'AVG', 'COUNT', 'SUM', 'MAX', 'MIN'): 对关联表进行跨表聚合，可以指定时间窗口 `window` (单位：天)。

        请严格按照以下JSON格式输出一个列表，不要包含任何额外的解释或Markdown标记：
        [
          {{
            "op": "OPERATION_NAME",
            "path": "EntityName.FieldName",
            "window": 30
          }},
          {{
            "op": "LATEST",
            "path": "UserProfile.AnnualIncome"
          }}
        ]
        """
        
    response_str = _call_llm(prompt)
    if response_str:
        try:
            clean_response_str = response_str.strip().replace('```json', '').replace('```', '')
            gene_list = json.loads(clean_response_str)
            print(f"  (真实 LLM 已成功返回 {len(gene_list)} 个基因创意)")
            return gene_list
        except json.JSONDecodeError:
            print(f"  (真实 LLM 返回了无效的JSON, 内容: {response_str})")
            return []
    return []


def llm_critique_causality(
    feature_list: List[str], 
    target_variable: str
) -> Dict[str, Any]:
    """
    [真实实现] 使用 LLM 进行"因果批判"。
    """
    print("\n[LLM Real] 正在调用真实 LLM API 进行\"因果批判\"...")
    
    features_str = ", ".join(feature_list)
    prompt = f"""
    作为一名资深的、负责任的数据科学家，请评估以下特征与目标变量之间可能存在的因果风险。

    目标变量: {target_variable}
    特征列表: {features_str}

    请特别关注以下几点：
    1.  是否存在伪关联或数据泄露的风险？
    2.  是否存在引入社会偏见（如性别、地域歧视）的风险？
    3.  这些特征是否是结果而非原因？

    请严格以JSON格式返回你的分析，只包含两个字段，不要有任何额外解释或Markdown标记：
    -   "risk_score": 一个0.0到1.0的浮点数，0.0代表无风险，1.0代表极高风险。
    -   "justification": 一段简短的文字，解释你给出该分数的核心理由。
    """
    
    response_str = _call_llm(prompt)
    if response_str:
        try:
            clean_response_str = response_str.strip().replace('```json', '').replace('```', '')
            critique_dict = json.loads(clean_response_str)
            print(f"  (真实 LLM 已返回风险评分: {critique_dict.get('risk_score')})")
            return critique_dict
        except json.JSONDecodeError:
            print(f"  (真实 LLM 返回了无效的JSON, 内容: {response_str})")
            return {'risk_score': 0.1, 'justification': 'LLM response was not valid JSON.'}
    return {'risk_score': 0.1, 'justification': 'LLM API call failed.'}

def llm_generate_cross_table_genes(
    secondary_schema: Dict[str, List[str]], 
    primary_entity_name: str,
    primary_key_name: str,
    target_variable: str
) -> List[Dict[str, Any]]:
    """
    [V1.2 实现] 使用 LLM 专注于创造跨表聚合特征。
    """
    print("\n[LLM Real] 正在调用真实 LLM API 进行\"跨表基因创生\"...")
    
    prompt = f"""
    作为一名顶尖的数据科学家，请为我进行跨表特征工程的头脑风暴。

    我的目标是预测主实体 `{primary_entity_name}` 上的目标变量 `{target_variable}`。

    主实体的关联键是 `{primary_key_name}`。

    现在，请专注于以下**副表**，为我创造一些有价值的聚合特征。你需要通过一个外键（名称可能与主键相似）将这些副表关联回主实体。

    这是我可用的副表数据模式:
    {json.dumps(secondary_schema, indent=2)}

    请为我生成一个包含至少10个特征工程创意的JSON列表。所有创意都必须是**聚合特征**。
    基因的类型必须是: 'AVG', 'COUNT', 'SUM', 'MAX', 'MIN'。
    你可以对时间戳字段指定 `window` (单位：天)来进行时间窗口内的聚合。

    请严格按照以下JSON格式输出一个列表，不要包含任何额外的解释或Markdown标记：
    [
      {{
        "op": "COUNT",
        "path": "SecondaryEntityName.PrimaryKey",
        "window": 30
      }},
      {{
        "op": "AVG",
        "path": "SecondaryEntityName.SomeNumericField"
      }}
    ]
    """
    response_str = _call_llm(prompt)
    if response_str:
        try:
            clean_response_str = response_str.strip().replace('```json', '').replace('```', '')
            gene_list = json.loads(clean_response_str)
            print(f"  (真实 LLM 已成功返回 {len(gene_list)} 个跨表基因创意)")
            return gene_list
        except json.JSONDecodeError:
            print(f"  (真实 LLM 返回了无效的JSON, 内容: {response_str})")
            return []
    return []
