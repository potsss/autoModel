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

def llm_infer_schema(raw_schema_info: Dict[str, Any], sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    [真实实现] 使用 LLM 进行"语义推断"。
    """
    print("\n[LLM Real] 正在调用真实 LLM API 进行\"语义推断\"...")
    
    prompt = f"""
    作为一名数据架构师，请根据原始的数据库表结构和数据样本，推断出业务层面的逻辑“实体”和它们之间的“关系”。

    这是我原始的数据库表和列:
    {json.dumps(raw_schema_info, indent=2)}

    这是每个表的一些数据样本:
    {json.dumps(sample_data, indent=2)}

    请为我生成一个描述业务逻辑的JSON。这个JSON应该包含：
    1.  `entities`: 一个实体列表。每个实体应有 `name` (例如 "UserProfile") 和 `primary_key`。
    2.  `relationships`: 一个关系列表。每个关系应描述两个实体如何通过外键关联，包括 `from_entity`, `to_entity`, `from_column`, `to_column`。

    请严格按照以下JSON格式输出，不要包含任何额外的解释或Markdown标记：
    {{
      "entities": [
        {{
          "name": "EntityName",
          "primary_key": "id",
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
    
    prompt = f"""
    作为一名顶尖的数据科学家，请为我进行特征工程的头脑风暴。

    我的目标是预测目标变量: `{target_variable}`

    这是我可用的、已经标准化的数据模式:
    {json.dumps(standard_schema, indent=2, ensure_ascii=False)}

    **重要约束**: 你生成的所有基因的 path 必须使用上述数据模式中实际存在的表名(EntityName)和字段名(FieldName)。
    不要创造新的表名或字段名。

    请为我生成一个包含至少10个特征工程创意的JSON列表。每个创意是一个"基因"，代表一种特征提取方法。
    基因的类型可以是：
    1.  直接特征 (op: 'LATEST'): 直接从主实体获取某个字段的最新值。
    2.  聚合特征 (op: 'AVG', 'COUNT', 'SUM', 'MAX', 'MIN'): 对关联表进行跨表聚合，可以指定时间窗口 `window` (单位：天)。

    示例(仅供参考,请使用实际存在的表名和字段):
    [
      {{
        "op": "LATEST",
        "path": "application_train.AMT_CREDIT"
      }},
      {{
        "op": "AVG",
        "path": "bureau.AMT_CREDIT_SUM",
        "window": 365
      }}
    ]

    请严格按照以下JSON格式输出一个列表，不要包含任何额外的解释或Markdown标记。
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
    [V1.2] 使用 LLM 专注于创造“跨表聚合”特征，并通过更通用的Prompt引导其生成高质量特征。
    参数：
    - secondary_schema: {副表实体名: [字段...]}（不包含主实体）
    - primary_entity_name: 主实体名（用于描述业务上下文）
    - primary_key_name: 主实体主键（用于提示关联方式）
    - target_variable: 目标变量（例如 "UserProfile.IsDefault"）
    返回：形如 [{"op": "COUNT", "path": "Entity.Field", "window": 30}, ...]
    """
    print("\n[LLM Real] 正在调用真实 LLM API 进行\"跨表基因创生\"...")
    try:
        prompt = f"""
作为一名顶尖的数据科学家，请为我进行跨表特征工程的头脑风暴。

我的主实体是 `{primary_entity_name}`，目标变量是 `{target_variable}`，主实体的关联键是 `{primary_key_name}`。

现在，请专注于以下“副表”来创造跨表聚合特征：
{json.dumps(secondary_schema, indent=2, ensure_ascii=False)}

---
**高级特征工程技巧 (请在构思时参考):**

1.  **差值特征 (Difference Features):** 寻找代表“计划”与“实际”的成对字段 (例如 `PLANNED_DATE` vs `ACTUAL_DATE`, `PLANNED_AMOUNT` vs `ACTUAL_AMOUNT`)。基于这些字段的**差值**进行聚合，通常能创造出极具预测能力的特征。例如，`AVG(ACTUAL_DATE - PLANNED_DATE)` 可以衡量平均延迟情况。

2.  **状态条件聚合 (Conditional Aggregations):** 如果一个表包含“状态”或“类型”字段 (例如，值为 'Active'/'Closed', 'Approved'/'Refused' 的列)，请考虑根据不同状态进行聚合。例如，`SUM(Amount WHERE Status='Active')` 或 `COUNT(ID WHERE Type='Refused')`。

3.  **时间窗口特征 (Time-Window Features):** 最近的行为往往更重要。请善用 `window` 参数来创造描述近期行为的特征。例如，`COUNT(ID, window=30)` 代表近30天的事件次数，`SUM(Amount, window=90)` 代表近90天的总金额。

---

要求：
1) 仅输出聚合类基因，op 必须为: 'AVG', 'COUNT', 'SUM', 'MAX', 'MIN'。
2) path 必须引用上面提供的副表实体及其字段，禁止引用主实体或不存在的字段。
3) 请综合运用上述高级技巧，创造出有业务洞察力的特征。

请严格输出一个 JSON 列表，不要包含任何解释或 Markdown：
[
  {{"op": "COUNT", "path": "SecondaryEntity.SomeID", "window": 30}},
  {{"op": "AVG",   "path": "SecondaryEntity.SomeAmount"}}
]
"""
        response_str = _call_llm(prompt)
        if response_str:
            clean = response_str.strip().replace('```json', '').replace('```', '')
            genes = json.loads(clean)
            if isinstance(genes, list):
                return genes
    except Exception as e:
        print(f"[LLM Real] 跨表基因创生失败: {e}")
    return []
