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
        
        # 对于兼容OpenAI的接口，response_format参数可能不是所有模型都支持
        # 如果遇到问题，可以移除这个参数，并在prompt中更强地约束模型输出JSON
        response_format = {"type": "json_object"} if is_json else {"type": "text"}

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            # response_format=response_format # 如果您的模型不支持此参数，请注释掉此行
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[LLM Real - 错误] API 调用失败: {e}")
        return None

def llm_infer_schema(
    db_conn: Any, 
    raw_schema_info: Dict[str, List[str]], 
    sample_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    [真实实现] 使用 LLM 进行"语义推断"。
    """
    print("\n[LLM Real] 正在调用真实 LLM API 进行\"语义推断\"...")
    
    prompt = f"""
    你是一个数据库专家，你的任务是分析一个原始的、命名混乱的数据库模式，并推断出其背后干净、标准的业务语义。

    这是原始的数据库信息：
    1. 表结构: {json.dumps(raw_schema_info, indent=2)}
    2. 表内数据抽样: { {k: v.to_dict('list') for k, v in sample_data.items()} }

    请根据以上信息，为我生成一个JSON格式的“推断模式”，需要包含以下内容：
    1.  将物理表名（如 'tbl_user_01'）映射到有意义的业务实体名（如 'UserProfile'）。
    2.  对于每个实体，将其混乱的物理列名（如 'col_xyz_01'）映射到标准的业务字段名（如 'AnnualIncome'）。
    3.  为每个字段提供类型（id, numeric, categorical, datetime, foreign_key, target）和简短的描述。
    4.  如果发现外键关系，请在 'references' 字段中明确指出其关联的标准实体和字段（如 'UserProfile.UserID'）。

    请严格按照以下JSON结构输出，不要包含任何额外的解释或Markdown标记：
    {{
      "BusinessEntityName": {{
        "physical_table": "raw_table_name",
        "fields": {{
          "StandardFieldName": {{
            "physical_column": "raw_column_name",
            "type": "field_type",
            "description": "description of the field"
          }}
        }}
      }}
    }}
    """
    
    response_str = _call_llm(prompt)
    if response_str:
        try:
            # LLM的返回可能包含Markdown代码块，需要先清理
            clean_response_str = response_str.strip().replace('```json', '').replace('```', '')
            inferred_schema = json.loads(clean_response_str)
            print("  (真实 LLM 已成功返回\"推断模式\")")
            return inferred_schema
        except json.JSONDecodeError:
            print(f"  (真实 LLM 返回了无效的JSON, 内容: {response_str})")
            return {{}}
    return {{}}


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
    {json.dumps(standard_schema, indent=2)}

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
            return {{'risk_score': 0.1, 'justification': 'LLM response was not valid JSON.'}}
    return {{'risk_score': 0.1, 'justification': 'LLM API call failed.'}}
