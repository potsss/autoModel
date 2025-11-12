"""
对抗性共演化系统 - 架构师模块

这是本项目的核心模块，包含四个核心类：
- GeneGenerator: 基因生成器
- FeatureEngine: V3真实特征引擎  
- FitnessEvaluator: 真实评估器
- EvolutionaryEngine: 演化引擎

V1.0 架构约定：
- 实现完整的机器学习流水线
- 支持跨表聚合特征工程
- 真实的模型训练和评估
- 遗传算法优化
"""

import time
import random
import json
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import shap
SHAP_AVAILABLE = True
# 统一禁止进入建模特征的字段（防泄漏）
# 增加常见主键/ID 列，避免误入建模
EXCLUDED_COLUMNS = set({
    "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV"
})  # 可按需扩展，比如 {"duration", "y"}
# 新增：兜底所需的 sklearn 组件
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, early_stopping
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 导入我们项目中的其他模块
from core_structures import ModelingGene, FeatureGene, TransformGene, ModelGene, FilterGene, ModelingChromosome
from knowledge_graph_interface import KnowledgeGraphInterface
from llm_interface import llm_generate_cross_table_genes


class GeneGenerator:
    """
    负责调用 LLM (占位符) 来动态生成初始基因池。
    """
    def __init__(self, translator: KnowledgeGraphInterface, target_variable: str, feature_config: Optional[Dict[str, Any]] = None):
        self.translator = translator
        self.target_variable = target_variable
        self.feature_config = feature_config or {}

    def _machine_screen_features(self, entity_name: str, top_k: Optional[int] = None) -> List[FeatureGene]:
        """
        使用一个轻量的 LGBM 模型对主表做预筛选，返回 Top-K 的 LATEST 特征。
        - 若训练失败或数据不满足条件，则降级为“从该表采样少量列”的兜底策略。
        - 只生成 LATEST 特征（不做任何跨表聚合）。
        """
        try:
            df = self.translator.get_entity_dataframe(entity_name)
        except Exception as e:
            print(f"[架构师-基因][机器筛选-错误] 读取实体 {entity_name} 失败: {e}")
            return []

        if df is None or getattr(df, "empty", True):
            print(f"[架构师-基因][机器筛选-兜底] 实体 {entity_name} 数据为空")
            return []

        # 解析目标字段
        tgt_field = None
        if isinstance(self.target_variable, str) and '.' in self.target_variable:
            _, tgt_field = self.target_variable.split('.', 1)
        # 兜底：若目标列不在表中，则不做机器筛选
        if not tgt_field or tgt_field not in df.columns:
            print(f"[架构师-基因][机器筛选-提示] 目标列 {tgt_field} 不在实体 {entity_name} 中，跳过机器筛选")
            return []

        y = df[tgt_field]
        X = df.drop(columns=[tgt_field])

        # 额外移除 ID/主键型列，避免被误选为特征
        try:
            id_like_cols = [c for c in X.columns if 'id' in c.lower() or 'sk_id' in c.lower()]
            if id_like_cols:
                X = X.drop(columns=id_like_cols, errors='ignore')
        except Exception:
            pass

        # 基本健壮性检查
        if len(X) < 50:
            print(f"[架构师-基因][机器筛选-兜底] 样本过少({len(X)}行)，跳过机器筛选")
            return []
        if X.shape[1] == 0:
            print("[架构师-基因][机器筛选-兜底] 无可用特征列")
            return []
        if getattr(y, 'nunique', lambda: 0)() < 2:
            print("[架构师-基因][机器筛选-兜底] 目标变量仅一个类别，跳过机器筛选")
            return []

        # 处理类别型特征
        try:
            for col in X.select_dtypes(include=['object', 'category']).columns:
                X[col] = X[col].astype('category')
        except Exception as e:
            print(f"[架构师-基因][机器筛选-提示] 类别型转换失败: {e}")

        # 动态 top_k：可由配置覆盖；否则根据比例计算
        try:
            std_schema = self.translator.get_standard_schema() or {}
            num_tables = len(std_schema)
        except Exception:
            num_tables = 1
        
        if top_k is None:
            cfg_main = (self.feature_config or {}).get('main_table', {})
            total_features = X.shape[1]
            
            if num_tables <= 1:
                # 单表场景，使用 ratio 计算
                ratio = float(cfg_main.get('lgbm_top_k_single_ratio', 0.25))
                top_k = int(total_features * ratio)
            else:
                # 多表场景，使用 ratio 计算
                ratio = float(cfg_main.get('lgbm_top_k_multi_ratio', 0.15))
                top_k = int(total_features * ratio)
        
        # 确保 top_k 在合理范围内 [5, total_features]
        top_k = max(5, min(top_k, X.shape[1]))

        # 训练 LGBM 获取重要性
        try:
            lgbm = LGBMClassifier(random_state=42, verbose=-1, n_estimators=60, max_depth=6)
            lgbm.fit(X, y)
            importances = pd.Series(getattr(lgbm, 'feature_importances_', np.zeros(X.shape[1])), index=X.columns)
            if float(importances.sum()) == 0.0:
                print("[架构师-基因][机器筛选-兜底] 特征重要性全为0，跳过机器筛选")
                return []
            selected = importances.nlargest(top_k).index.tolist()
            print(f"[架构师-基因][机器筛选] 主表Top-{len(selected)}: {selected[:10]}...")
            return [FeatureGene(op='LATEST', path=f"{entity_name}.{c}") for c in selected if c not in EXCLUDED_COLUMNS and c != tgt_field]
        except Exception as e:
            print(f"[架构师-基因][机器筛选-兜底] LGBM训练失败: {e}")
            return []

    def generate_initial_pool(self) -> List[ModelingGene]:
        """
        [V1.6 - 混合策略] 生成初始基因池：
        1) 直接加载主表所有可用特征作为 LATEST 基因。
        2) 追加 LLM 生成的跨表聚合特征。
        3) 始终追加若干 ModelGene，保证演化可运行。
        """
        print("[架构师-基因] 启动基因生成：[混合策略] 主表全特征 + LLM跨表特征")

        gene_pool: List[ModelingGene] = []

        # 1) 主表名称/目标字段解析
        primary_entity = None
        target_field = None
        if isinstance(self.target_variable, str) and '.' in self.target_variable:
            primary_entity, target_field = self.target_variable.split('.', 1)
        else:
            primary_entity, target_field = 'application_train', 'TARGET'
        
        # 2) 直接加载主表所有特征
        try:
            main_df = self.translator.get_entity_dataframe(primary_entity)
            if main_df is not None and not main_df.empty:
                cols_to_exclude = {target_field} | EXCLUDED_COLUMNS
                available_cols = [col for col in main_df.columns if col not in cols_to_exclude]
                for col in available_cols:
                    gene_pool.append(FeatureGene(op="LATEST", path=f"{primary_entity}.{col}"))
                print(f"[架构师-基因] ✓ 已从主表 '{primary_entity}' 加载 {len(available_cols)} 个特征基因。")
            else:
                raise ValueError(f"无法获取主表 '{primary_entity}' 的数据。")
        except Exception as e:
            print(f"[架构师-基因][错误] 加载主表特征失败: {e}")

        # 2.5) [新增] 加载副表的预聚合特征
        try:
            standard_schema = self.translator.get_standard_schema() or {}
            if standard_schema and isinstance(standard_schema, dict):
                secondary_tables = [k for k in standard_schema.keys() if k != primary_entity]
                total_secondary_features = 0
                
                for table_name in secondary_tables:
                    try:
                        table_df = self.translator.get_entity_dataframe(table_name)
                        if table_df is not None and not table_df.empty:
                            # 获取副表的所有列（排除主键/外键）
                            cols_to_exclude = EXCLUDED_COLUMNS | {'SK_ID_CURR', 'SK_ID_PREV', 'SK_ID_BUREAU'}
                            available_cols = [col for col in table_df.columns if col not in cols_to_exclude]
                            
                            # 为副表特征创建聚合基因 (使用 AVG 作为默认聚合)
                            for col in available_cols:
                                # 根据字段类型选择聚合方式
                                if col.startswith(('BB_', 'POS_', 'CC_', 'STATUS_')):  # 预聚合特征
                                    gene_pool.append(FeatureGene(op="AVG", path=f"{table_name}.{col}"))
                                else:  # 原始字段
                                    gene_pool.append(FeatureGene(op="AVG", path=f"{table_name}.{col}"))
                            
                            total_secondary_features += len(available_cols)
                            print(f"[架构师-基因] ✓ 已从副表 '{table_name}' 加载 {len(available_cols)} 个特征基因。")
                    except Exception as e:
                        print(f"[架构师-基因][警告] 加载副表 '{table_name}' 失败: {e}")
                
                if total_secondary_features > 0:
                    print(f"[架构师-基因] ✓ 副表特征总计: {total_secondary_features} 个")
        except Exception as e:
            print(f"[架构师-基因][警告] 加载副表特征失败: {e}")

        # 3) 追加 LLM 生成的跨表特征
        # [V1.7 临时禁用] 副表特征已通过预聚合包含，手动加载到基因池
        print("[架构师-基因] 跳过LLM跨表特征生成（副表特征已手动加载）")
        try:
            standard_schema = self.translator.get_standard_schema() or {}
            relationships = self.translator.get_relationship_keys() or {}
            
            if False:  # 临时禁用LLM调用
                secondary_schema = {k: v for k, v in standard_schema.items() if k != primary_entity}
                primary_key = 'SK_ID_CURR' # [V1.6 修正] 直接使用已知的主键，因为标准schema是简化的字符串列表
                
                print(f"[架构师-基因] (LLM跨表) 主实体={primary_entity}, 主键={primary_key}, 副表数={len(secondary_schema)}")
                
                genes_json = llm_generate_cross_table_genes(
                    secondary_schema=secondary_schema,
                    primary_entity_name=primary_entity,
                    primary_key_name=primary_key,
                    target_variable=self.target_variable
                )
                
                llm_gene_count = 0
                if isinstance(genes_json, list):
                    for g in genes_json:
                        try:
                            op = g.get('op')
                            path = g.get('path')
                            if not op or not path or '.' not in path: continue
                            entity, field = path.split('.', 1)
                            if entity == primary_entity or entity not in standard_schema or field in EXCLUDED_COLUMNS: continue
                            
                            # 简单校验关系是否存在
                            has_relation = any(
                                rel.get('from_entity') == entity and rel.get('to_entity') == primary_entity
                                for _, rel in relationships.items()
                            )
                            if not has_relation: continue

                            gene_pool.append(FeatureGene(op=op, path=path, window=g.get('window')))
                            llm_gene_count += 1
                        except Exception:
                            continue
                print(f"[架构师-基因] ✓ 已从LLM加载 {llm_gene_count} 个跨表特征基因。")
            else:
                print("[架构师-基因] 单表场景：不调用LLM。")
        except Exception as e:
            print(f"[架构师-基因][警告] LLM 跨表基因生成异常: {e}")

        # 4) 无论如何，确保有多个模型基因
        model_configs = [
            {"alg": "LogisticRegression", "params": {"solver": "liblinear", "class_weight": "balanced", "random_state": 42}},
            {"alg": "RandomForestClassifier", "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}},
            {"alg": "LGBMClassifier", "params": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "random_state": 42, "verbose": -1, "objective": "binary"}}
        ]
        for config in model_configs:
            gene_pool.append(ModelGene(alg=config["alg"], params=config["params"]))

        # [V1.7 调试] 将生成的特征基因保存到文件
        try:
            feature_genes_to_log = [g for g in gene_pool if isinstance(g, FeatureGene)]
            log_content = "--- Generated Feature Genes ---\n"
            log_content += f"Total Feature Genes: {len(feature_genes_to_log)}\n\n"
            
            main_table_genes = [f"  - {g.op}: {g.path}" for g in feature_genes_to_log if g.path.startswith(primary_entity)]
            cross_table_genes = [f"  - {g.op}: {g.path} (window: {g.window})" for g in feature_genes_to_log if not g.path.startswith(primary_entity)]

            log_content += f"Main Table Genes ({len(main_table_genes)}):\n"
            log_content += "\n".join(main_table_genes)
            log_content += f"\n\nCross-Table Genes ({len(cross_table_genes)}):\n"
            log_content += "\n".join(cross_table_genes)

            with open("generated_genes_log.txt", "w", encoding="utf-8") as f:
                f.write(log_content)
            print("[架构师-基因] ✓ 已将生成的特征基因记录到 generated_genes_log.txt")
        except Exception as log_e:
            print(f"[架构师-基因][警告] 记录生成的基因失败: {log_e}")

        # 5) 去重
        seen = set()
        deduped: List[ModelingGene] = []
        for g in gene_pool:
            if isinstance(g, FeatureGene):
                key = ("F", g.op, g.path, g.window)
            elif isinstance(g, ModelGene):
                key = ("M", g.alg, tuple(sorted(g.params.items())) if isinstance(g.params, dict) else None)
            else:
                key = ("O", repr(g))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(g)

        print(f"[架构师-基因] 成功生成 {len([x for x in deduped if isinstance(x, FeatureGene)])} 个特征基因 + {len([x for x in deduped if isinstance(x, ModelGene)])} 个模型基因。")
        return deduped


class FeatureEngine:
    """
    V3 真实特征工程引擎。
    负责将一个"染色体"翻译成一个可训练的 (X, y) 矩阵。
    """
    def __init__(self, translator: KnowledgeGraphInterface, standard_target_variable: str):
        self.translator = translator
        self.relationships = translator.get_relationship_keys()
        # (V1.1 新增) 保存标准目标变量信息
        self.target_entity, self.target_field = standard_target_variable.split('.')

    def build_features(self, chromosome: ModelingChromosome) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
        """
        V3 核心逻辑：
        1. (暂不实现) 处理 FilterGene
        2. 获取基础实体 (UserProfile)
        3. 处理 FeatureGene (LATEST 和 跨表聚合)
        4. (暂不实现) 处理 TransformGene
        5. 返回 X, y, 和特征类型列表
        """
        
        # 1) 取目标实体
        base_df = self.translator.get_entity_dataframe(self.target_entity)
        if base_df is None or base_df.empty:
            # 直接兜底到“单表全列基线”
            return self._build_baseline_X(pd.DataFrame(), self.target_field)

        # y (V1.3 修正: 使用与 baseline 一致的鲁棒查找逻辑)
        y_col_name = None
        normalized_target = self.target_field.lower().replace('_', '')
        if self.target_field in base_df.columns:
            y_col_name = self.target_field
        else:
            for c in base_df.columns:
                if c.lower().replace('_', '') == normalized_target:
                    y_col_name = c
                    break
        
        if y_col_name:
            y = base_df[y_col_name]
        else:
            y = pd.Series(0, index=base_df.index, dtype=int)


        # 2) X 初始化 - 使用列表收集所有特征，最后一次性concat避免碎片化
        feature_dfs = []
        base_index_df = pd.DataFrame(index=base_df.index)

        # 3) 解析 FeatureGene（允许 LATEST 直接从当前目标实体取值）
        feature_genes = [g for g in chromosome.genes if isinstance(g, FeatureGene)]
        if not feature_genes:
            # 没有任何特征基因：自动采样少量LATEST特征，避免全列兜底
            print(f"[特征引擎-提示] 染色体无特征基因，自动采样主表LATEST特征...")
            safe_cols = [
                c for c in base_df.columns
                if c not in EXCLUDED_COLUMNS and c != self.target_field
            ]
            sampled = safe_cols[: min(16, len(safe_cols))]
            X_sampled = base_df[sampled].copy()
            X_sampled.columns = [f"LATEST_{self.target_entity}_{c}" for c in sampled]
            print(f"[特征引擎-提示] 已自动采样 {X_sampled.shape[1]} 个LATEST特征")
            if X_sampled.shape[1] == 0:
                # 实在拿不到，再回退到"单表全列"
                return self._build_baseline_X(base_df, self.target_field)
            else:
                # 直接返回采样特征
                numeric_features = X_sampled.select_dtypes(include=[np.number]).columns.tolist()
                categorical_features = [c for c in X_sampled.columns if c not in numeric_features]
                return X_sampled, y, numeric_features, categorical_features

        for gene in feature_genes:
            try:
                entity_name, field_name = gene.path.split('.', 1)
            except Exception:
                continue

            feature_name = f"{gene.op}_{entity_name}_{field_name}"
            if gene.window:
                feature_name += f"_{gene.window}d"
            
            # 检查是否已经在收集列表中
            already_exists = any(feature_name in df.columns for df in feature_dfs if not df.empty)
            if already_exists:
                continue

            if gene.op == 'LATEST':
                # [修复] 允许从当前目标实体直接取值，即使实体名不完全匹配
                # 优先检查实体名匹配，否则尝试直接查找字段
                found = False
                if entity_name == self.target_entity and field_name in base_df.columns:
                    found = True
                elif field_name in base_df.columns:
                    # 兜底：即使实体名不匹配，如果字段存在也使用
                    found = True
                    print(f"[特征引擎-兜底] 实体名不匹配({entity_name}!={self.target_entity})，但字段 {field_name} 存在，使用该字段")
                
                if found:
                    if field_name in EXCLUDED_COLUMNS:
                        print(f"[特征引擎] 跳过被禁用字段(LATEST): {field_name}")
                        continue
                    # 收集到列表而不是直接添加
                    feat_df = pd.DataFrame({feature_name: base_df[field_name]}, index=base_df.index)
                    feature_dfs.append(feat_df)
                continue
            elif gene.op in ['AVG', 'COUNT', 'SUM', 'MAX', 'MIN']:
                # [V1.6 提升] 跨表聚合特征：优先按 from/to 实体匹配关系，其次按命名约定匹配
                relation = None
                matched_key = None

                # 1) 直接按 from/to 实体匹配（最稳妥，适配 schema_config）
                try:
                    for rel_key, rel_info in (self.relationships or {}).items():
                        if rel_info.get('from_entity') == entity_name and rel_info.get('to_entity') == self.target_entity:
                            relation = rel_info
                            matched_key = rel_key
                            break
                except Exception:
                    pass

                # 2) 命名约定作为退路（仅通用模式，移除硬编码实体名）
                if relation is None:
                    relation_key = f"{entity_name}_to_{self.target_entity}"
                    if relation_key in (self.relationships or {}):
                        relation = self.relationships[relation_key]
                        matched_key = relation_key

                if not relation:
                    print(f"[特征引擎-警告] 找不到 {entity_name} -> {self.target_entity} 的关系映射，跳过 {gene.path}")
                    continue

                print(f"[特征引擎] 使用关系: {matched_key} 构建跨表特征 {feature_name}")
                
                # 获取关系键
                fk_from = relation['from_key']
                pk_to = relation['to_key']
                from_entity = relation.get('from_entity', entity_name)
                
                # 获取事实表数据
                fact_df = self.translator.get_entity_dataframe(from_entity)
                if fact_df is None or fact_df.empty:
                    print(f"[特征引擎-警告] 无法获取实体 {from_entity} 的数据")
                    continue
                
                # 检查字段是否存在
                if field_name not in fact_df.columns:
                    print(f"[特征引擎-警告] 字段 {field_name} 不在 {from_entity} 中")
                    continue
                
                # 检查外键是否存在
                if fk_from not in fact_df.columns:
                    print(f"[特征引擎-警告] 外键 {fk_from} 不在 {from_entity} 中")
                    continue
                
                # 检查主键是否存在于主表
                if pk_to not in base_df.columns:
                    print(f"[特征引擎-警告] 主键 {pk_to} 不在目标表 {self.target_entity} 中")
                    continue

                # 执行聚合
                agg_op = gene.op.lower()
                if agg_op == 'avg': agg_op = 'mean'
                if agg_op == 'count':
                    agg_op = 'count'

                try:
                    agg_series = fact_df.groupby(fk_from)[field_name].agg(agg_op)
                    agg_series.name = feature_name
                    agg_df = agg_series.reset_index()
                    agg_df.columns = [pk_to, feature_name]
                    merged = base_df[[pk_to]].merge(agg_df, on=pk_to, how='left')
                    # 收集到列表而不是直接添加到X
                    feat_df = pd.DataFrame({feature_name: merged[feature_name].values}, index=base_df.index)
                    feature_dfs.append(feat_df)
                    print(f"[特征引擎] ✓ 成功构建跨表特征 {feature_name} ({len(agg_df)} 聚合值)")
                except Exception as merge_error:
                    print(f"[特征引擎-警告] 聚合或合并失败 {feature_name}: {merge_error}")
                    continue

        # 4) 批量合并所有特征，避免DataFrame碎片化
        if feature_dfs:
            X = pd.concat([base_index_df] + feature_dfs, axis=1, copy=False)
        else:
            X = base_index_df
        
        # 5) 清理 & 判空
        X = X.dropna(axis=1, how='all')
        if X is None or X.shape[1] == 0:
            # 还是没产出特征 → 优先自动采样主表少量 LATEST 特征，避免"一步到位全列兜底"
            print(f"[特征引擎-提示] 基因未产出特征，自动采样主表少量LATEST特征...")
            safe_cols = [
                c for c in base_df.columns
                if c not in EXCLUDED_COLUMNS and c != self.target_field
            ]
            sampled = safe_cols[: min(16, len(safe_cols))]
            X_sampled = base_df[sampled].copy()
            X_sampled.columns = [f"LATEST_{self.target_entity}_{c}" for c in sampled]
            X = X_sampled
            print(f"[特征引擎-提示] 已自动采样 {X.shape[1]} 个LATEST特征")
            if X.shape[1] == 0:
                # 实在拿不到，再回退到"单表全列"
                print(f"[特征引擎-警告] 无法采样任何特征，回退到全列基线")
                return self._build_baseline_X(base_df, self.target_field)

        # 6) 列类型拆分
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        return X, y, numeric_features, categorical_features

    def _build_baseline_X(self, df: pd.DataFrame, target_col: str):
        """
        单表全列基线特征（除目标列外全部进 X）。
        注意：这里返回的是 DataFrame/Series；具体缺失值补齐/标准化/OneHot 交给 Evaluator 的 Pipeline 做。
        """
        if df is None or df.empty:
            # 返回空壳，Evaluator 会判空
            return pd.DataFrame(), pd.Series(dtype=int), [], []

        df = df.copy()
        
        # y 强制为 0/1（保留原有映射）
        # V1.2 修正：更鲁棒的大小写不敏感查找（处理下划线）
        y_col_name = None
        normalized_target = target_col.lower().replace('_', '')
        if target_col in df.columns:
            y_col_name = target_col
        else:
            for c in df.columns:
                if c.lower().replace('_', '') == normalized_target:
                    y_col_name = c
                    break
        
        if y_col_name:
            y = df[y_col_name]
        else:
            # 如果真的找不到，则创建全为0的列作为兜底
            y = pd.Series(0, index=df.index)

        # y 强制为 0/1（保留原有映射）
        if y.dtype == object:
            y = y.map({'yes': 1, 'no': 0, 'y': 1, 'n': 0}).fillna(y).astype(str)
            y = y.astype('category').cat.codes
        y = y.astype(int)

        # 一次性剔除目标列 + 禁用列
        to_drop = {y_col_name, target_col} | (EXCLUDED_COLUMNS & set(df.columns))
        X = df.drop(columns=[c for c in to_drop if c in df.columns], errors='ignore')

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = [c for c in X.columns if c not in numeric_features]

        print(f"[特征引擎-兜底] 使用全列表基线(已排除 {sorted([c for c in to_drop if c is not None])}): "
              f"num={len(numeric_features)}, cat={len(categorical_features)}, X形状={X.shape}")
        return X, y, numeric_features, categorical_features


class FitnessEvaluator:
    """
    V1.0 真实适应度评估器。
    调用特征引擎，并使用 Sklearn Pipeline 真实地训练和评估模型。
    """
    def __init__(self, feature_engine: FeatureEngine):
        self.feature_engine = feature_engine

    def evaluate(self, chromosome: ModelingChromosome, calculate_shap: bool = False) -> Dict[str, Any]:
        """
        [V1.1] 执行完整的"评估"流程 (使用3-折交叉验证)，并返回一个包含"分数"和"成本"的字典。
        [V1.4] 新增：可选择性地计算并返回 SHAP 值。
        [V1.5] 修正：对齐基线脚本的预处理和评估流程。
        """
        start_time = time.time()
        
        try:
            # 1) 特征构造
            X, y, num_features_orig, cat_features_orig = self.feature_engine.build_features(chromosome)

            # 检查 y 是否包含多于一个类别
            if y.nunique() < 2:
                print(f"[评估错误] 目标变量 y 只包含一个类别，无法进行评估。")
                return {'auc': 0.0, 'evaluation_time_ms': (time.time() - start_time) * 1000, 'error': 'Target variable has less than 2 classes.'}

            # 双保险：如果上游误入任何被禁用列，这里再一次剔除
            for bad_col in list(EXCLUDED_COLUMNS):
                if hasattr(X, "columns") and bad_col in X.columns:
                    print(f"[评估器] 发现被禁用列残留，已移除: {bad_col}")
                    X = X.drop(columns=[bad_col])
                    if bad_col in num_features_orig:
                        num_features_orig.remove(bad_col)
                    if bad_col in cat_features_orig:
                        cat_features_orig.remove(bad_col)

            # --- DEBUG: feature matrix & target overview ---
            try:
                print(f"[调试] X type={type(X)}, shape={getattr(X, 'shape', None)}")
                y_series = y if isinstance(y, pd.Series) else pd.Series(y)
                print(f"[调试] y分布: {y_series.value_counts(dropna=False).to_dict()}")
                print(f"[调试] num_features_orig={len(num_features_orig)}, cat_features_orig={len(cat_features_orig)}")
                if hasattr(X, "head"):
                    print(f"[调试] X前5列: {list(X.columns)[:5]}")
            except Exception as _dbg_e:
                print(f"[调试] 打印特征信息失败: {str(_dbg_e)}")

            # 运行时断言，确保 duration 已剔除
            print(f"[调试] X列样本(前5): {list(X.columns)[:5]}")
        
            # [V1.5 修正] 对齐基线脚本的预处理：LabelEncoder for categoricals
            X_processed = X.copy()
            categorical_features_in_X = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 确保所有类别特征都被 LabelEncoder 处理
            for col in categorical_features_in_X:
                le = LabelEncoder()
                # 填充 NaN 以便 LabelEncoder 处理
                placeholder = '---missing---'
                X_processed[col] = X_processed[col].astype(str).fillna(placeholder)
                X_processed[col] = le.fit_transform(X_processed[col])
            
            # [V1.5 修正] 对数值特征进行缺失值填充 (SimpleImputer)
            numerical_features_in_X = X_processed.select_dtypes(include=np.number).columns.tolist()
            if numerical_features_in_X:
                imputer = SimpleImputer(strategy='median')
                X_processed[numerical_features_in_X] = imputer.fit_transform(X_processed[numerical_features_in_X])

            # 更新特征列表以反映 LabelEncoder 后的状态
            # LightGBM 可以直接处理 LabelEncoder 后的整数类别特征
            num_features_final = X_processed.select_dtypes(include=np.number).columns.tolist()
            cat_features_final = [col for col in X_processed.columns if col not in num_features_final] # 此时cat_features_final中的列也是数值型了

            # 3. 模型
            model_gene = next((g for g in chromosome.genes if isinstance(g, ModelGene)), None)
            if model_gene:
                if model_gene.alg == 'LogisticRegression':
                    model_template = LogisticRegression(**model_gene.params)
                elif model_gene.alg == 'RandomForestClassifier':
                    model_template = RandomForestClassifier(**model_gene.params)
                elif model_gene.alg == 'LGBMClassifier':
                    model_template = LGBMClassifier(**model_gene.params)
                else: # Fallback
                    print(f"[警告] 未知的模型算法: {model_gene.alg}，使用默认的 LogisticRegression。")
                    model_template = LogisticRegression(solver='liblinear', random_state=42)
            else: # Fallback if no model gene
                print(f"[警告] 染色体中没有模型基因，使用默认的 LogisticRegression。")
                model_template = LogisticRegression(solver='liblinear', random_state=42)

            # 5. [V1.1] K-折交叉验证 (手动循环以支持 early_stopping 和 categorical_feature)
            print("[调试] 开始3-折交叉验证...")
            kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            aucs = []
            
            # [V1.5 修正] 手动循环以支持 LGBM 的 early_stopping 和 categorical_feature
            for fold, (train_index, val_index) in enumerate(kfold.split(X_processed, y)):
                X_train, X_val = X_processed.iloc[train_index], X_processed.iloc[val_index]
                y_train, y_val = y.iloc[train_index], y.iloc[val_index]

                # 重新实例化模型，确保每个折叠都是独立的
                fold_model = model_template.__class__(**model_template.get_params())

                # LGBM 特殊处理：early_stopping 和 categorical_feature
                if isinstance(fold_model, LGBMClassifier):
                    fold_model.fit(X_train, y_train,
                                   eval_set=[(X_val, y_val)],
                                   eval_metric='auc',
                                   callbacks=[lgb.early_stopping(10, verbose=False)],
                                   categorical_feature=[X_processed.columns.get_loc(c) for c in cat_features_final])
                else:
                    fold_model.fit(X_train, y_train)

                preds = fold_model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, preds)
                aucs.append(auc)
                print(f"  Fold {fold+1} AUC: {auc:.4f}")

            auc_mean = np.mean(aucs)
            auc_std = np.std(aucs)
            print(f"[调试] 交叉验证完成，AUCs={np.round(aucs, 4)}, Mean AUC={auc_mean:.4f}, Std={auc_std:.4f}")

            result = {
                'auc': auc_mean,
                'auc_mean': auc_mean,
                'auc_std': auc_std,
                'feature_count': X_processed.shape[1] # 使用处理后的特征数量
            }

            # 8. [V1.4] 可选的 SHAP 分析
            if calculate_shap:
                print("[SHAP] 启动 SHAP 分析...")
                try:
                    # 为 SHAP 分析训练一个最终模型 (使用全部数据)
                    final_shap_model = model_template.__class__(**model_template.get_params())

                    # LGBM 特殊处理：categorical_feature
                    if isinstance(final_shap_model, LGBMClassifier):
                        final_shap_model.fit(X_processed, y,
                                             categorical_feature=[X_processed.columns.get_loc(c) for c in cat_features_final])
                    else:
                        final_shap_model.fit(X_processed, y)

                    # 仅为 Tree 模型优化
                    if SHAP_AVAILABLE and isinstance(final_shap_model, (LGBMClassifier, RandomForestClassifier)):
                        explainer = shap.TreeExplainer(final_shap_model)
                        # SHAP 值直接在 X_processed 上计算
                        shap_values = explainer.shap_values(X_processed)
                        
                        # 处理 SHAP 值（对于二分类，通常用类别1的值）
                        shap_values_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values
                        
                        # 计算每个特征的平均绝对 SHAP 值
                        mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
                        
                        # 将 SHAP 值映射回特征名
                        shap_dict = dict(zip(X_processed.columns, mean_abs_shap))
                        result['shap_values'] = shap_dict
                        print(f"[SHAP] ✓ 分析完成，获得 {len(shap_dict)} 个特征的 SHAP 值。")
                    else:
                        print(f"[SHAP] 警告: 模型 {type(final_shap_model).__name__} 不是受支持的 Tree 模型，跳过 SHAP 分析。")

                except Exception as shap_e:
                    print(f"[SHAP] 错误: SHAP 分析失败: {shap_e}")
                    result['shap_error'] = str(shap_e)

            result['evaluation_time_ms'] = (time.time() - start_time) * 1000
            return result

        except Exception as e:
            print(f"[评估错误] {str(e)}")
            evaluation_time_ms = (time.time() - start_time) * 1000
            return {'auc': 0.0, 'evaluation_time_ms': evaluation_time_ms, 'error': str(e)}


class EvolutionaryEngine:
    """
    实现遗传算法（选择、交叉、变异）的纯逻辑。
    """
    def __init__(self, gene_pool: List[ModelingGene]):
        self.gene_pool = gene_pool
        self.feature_genes = [g for g in gene_pool if isinstance(g, FeatureGene)]
        self.model_genes = [g for g in gene_pool if isinstance(g, ModelGene)]
        # (未来可添加 Transform/Filter 基因)

    def _get_gene_key(self, gene: ModelingGene) -> Any:
        """为 ModelingGene 生成一个可哈希的键"""
        if isinstance(gene, FeatureGene):
            return ("F", gene.op, gene.path, gene.window)
        elif isinstance(gene, ModelGene):
            # 将字典转换为可哈希的元组，确保可哈希性
            return ("M", gene.alg, tuple(sorted(gene.params.items())) if isinstance(gene.params, dict) else None)
        elif isinstance(gene, TransformGene): # 假设 TransformGene 也有可能出现，虽然目前没有用到
            return ("T", gene.op, tuple(sorted(gene.inputs)) if isinstance(gene.inputs, list) else gene.inputs)
        elif isinstance(gene, FilterGene): # 假设 FilterGene 也有可能出现，虽然目前没有用到
            return ("L", gene.condition)
        else:
            return ("O", repr(gene)) # 其他未知基因类型，用repr兜底

    def initialize_population(self, size: int, config: Optional[Dict[str, Any]] = None) -> List[ModelingChromosome]:
        """V1.3 智能初始化 (动态特征数量，可配置)：确保染色体可被评估
        config: {
          'min_features_ratio': float,  # 默认0.15
          'max_features_ratio': float,  # 默认0.40
          'max_features_floor': int     # 默认8
        }
        """
        population = []
        if not self.feature_genes or not self.model_genes:
            raise ValueError("基因池中缺少必要的 FeatureGene 或 ModelGene！")

        total_features = len(self.feature_genes)
        # 根据配置/默认参数，动态决定初始特征数量范围
        cfg = config or {}
        min_ratio = float(cfg.get('min_features_ratio', 0.15))
        max_ratio = float(cfg.get('max_features_ratio', 0.40))
        max_floor = int(cfg.get('max_features_floor', 8))

        min_count = max(1, int(total_features * min_ratio))
        percentage_based_max = min(total_features, max(min_count, int(total_features * max_ratio)))
        max_count = min(total_features, max(percentage_based_max, max_floor))
        if min_count > max_count:
            min_count = max_count
        print(f"[演化引擎] 动态初始化特征数范围: [{min_count}, {max_count}] (总特征池: {total_features})")

        for _ in range(size):
            genes = []
            # 从动态范围中随机选择特征数量
            if min_count >= max_count:
                num_features = min_count
            else:
                num_features = random.randint(min_count, max_count)
            if num_features > 0:
                genes.extend(random.sample(self.feature_genes, num_features))
            genes.append(random.choice(self.model_genes))
            population.append(ModelingChromosome(genes=genes))
        return population

    def select(self, population: List[ModelingChromosome], scores: List[float]) -> List[ModelingChromosome]:
        """V1.0 锦标赛选择"""
        selected = []
        pop_size = len(population)
        for _ in range(pop_size):
            i, j = random.sample(range(pop_size), 2)
            winner_idx = i if scores[i] > scores[j] else j
            selected.append(population[winner_idx])
        return selected

    def crossover(self, parent1: ModelingChromosome, parent2: ModelingChromosome) -> ModelingChromosome:
        """V1.0 单点交叉（简化版），V1.3 增加去重"""
        # (简化：只交换 FeatureGene，保持 ModelGene 不变)
        p1_features = [g for g in parent1.genes if isinstance(g, FeatureGene)]
        p2_features = [g for g in parent2.genes if isinstance(g, FeatureGene)]
        # 继承父1的模型，若不存在则随机兜底
        mg = [g for g in parent1.genes if isinstance(g, ModelGene)]
        model_gene = mg[0] if mg else (random.choice(self.model_genes) if self.model_genes else None)

        # 确保有足够的特征进行交叉
        if len(p1_features) <= 1 or len(p2_features) <= 1:
            combined_features = p1_features + p2_features
        else:
            cut = random.randint(1, min(len(p1_features), len(p2_features)) - 1)
            combined_features = p1_features[:cut] + p2_features[cut:]

        # 去重保持顺序（按语义键 op+path+window 去重）
        seen = set()
        child_features = []
        for g in combined_features:
            key = self._get_gene_key(g) # 修正：使用 _get_gene_key
            if key in seen:
                continue
            seen.add(key)
            child_features.append(g)
        genes = child_features + ([model_gene] if model_gene is not None else [])
        return ModelingChromosome(genes=genes)

    def mutate(self, chromosome: ModelingChromosome) -> ModelingChromosome:
        """V1.1 变异 (增加/删除/替换)"""
        if random.random() < 0.3:
            genes = chromosome.genes[:]  # 操作副本
            mutation_roll = random.random()

            # 20% 概率增加一个特征
            if mutation_roll < 0.2 and self.feature_genes:
                current_genes_keys = {self._get_gene_key(g) for g in genes}
                potential_additions = [g for g in self.feature_genes if self._get_gene_key(g) not in current_genes_keys]
                if potential_additions:
                    genes.append(random.choice(potential_additions))
                    # 重建 genes 列表，确保添加后没有重复
                    genes = list({self._get_gene_key(g): g for g in genes}.values())
                    print("[变异] 增加基因")
                    return ModelingChromosome(genes=genes)

            # 20% 概率删除一个特征
            elif mutation_roll < 0.4:
                feature_genes_in_chromo = [g for g in genes if isinstance(g, FeatureGene)]
                if len(feature_genes_in_chromo) > 1:  # 至少保留一个特征
                    gene_to_remove = random.choice(feature_genes_in_chromo)
                    genes.remove(gene_to_remove)
                    # 重建 genes 列表，确保删除后没有其他重复
                    genes = list({self._get_gene_key(g): g for g in genes}.values())
                    print("[变异] 删除基因")
                    return ModelingChromosome(genes=genes)

            # 60% 概率替换一个基因 (原逻辑)
            else:
                if not genes: return chromosome
                idx_to_mutate = random.randrange(len(genes))
                gene_to_mutate = genes[idx_to_mutate]

                if isinstance(gene_to_mutate, FeatureGene) and self.feature_genes:
                    # 确保替换的基因是新的
                    current_genes_keys = {self._get_gene_key(g) for g in genes}
                    potential_replacements = [g for g in self.feature_genes if self._get_gene_key(g) not in current_genes_keys]
                    if potential_replacements:
                        genes[idx_to_mutate] = random.choice(potential_replacements)
                        # 重建 genes 列表，确保替换后去重
                        genes = list({self._get_gene_key(g): g for g in genes}.values())
                    print("[变异] 替换特征基因")
                elif isinstance(gene_to_mutate, ModelGene) and self.model_genes:
                    genes[idx_to_mutate] = random.choice(self.model_genes)
                    print("[变异] 替换模型基因")
                return ModelingChromosome(genes=genes)

        return chromosome

    def refine_chromosome(self, chromosome: ModelingChromosome, shap_values: Dict[str, float]) -> ModelingChromosome:
        """
        [V1.4] 使用 SHAP 值来“精炼”一个染色体。
        它会找到贡献度最低的特征，并用基因池中的一个新特征替换它。
        """
        if not shap_values:
            return chromosome

        # 1. 找到 SHAP 值最低的特征名
        # SHAP value 的 key 可能是 'num__LATEST_entity_field' 或 'cat__...'
        min_shap_feature = min(shap_values, key=shap_values.get)
        
        # 2. 从染色体中找到对应的基因
        target_gene_to_remove = None
        original_genes = [g for g in chromosome.genes if isinstance(g, FeatureGene)]

        for gene in original_genes:
            # 构建一个可能的特征名来进行模糊匹配
            # gene.path = "application_train.DAYS_BIRTH" -> "application_train_DAYS_BIRTH"
            gene_feature_part = gene.path.replace('.', '_')
            if gene_feature_part in min_shap_feature:
                target_gene_to_remove = gene
                break
        
        if not target_gene_to_remove:
            print(f"[精炼] 警告: 无法从染色体中匹配到SHAP值最低的特征 {min_shap_feature}，跳过精炼。")
            return chromosome

        # 3. 替换基因
        new_genes_list = [g for g in chromosome.genes if self._get_gene_key(g) != self._get_gene_key(target_gene_to_remove)]
        
        # 寻找一个不在当前染色体中的新基因
        current_genes_keys = {self._get_gene_key(g) for g in new_genes_list} # 使用新列表的键
        potential_additions = [g for g in self.feature_genes if self._get_gene_key(g) not in current_genes_keys]

        if not potential_additions:
            print("[精炼] 警告: 基因池中没有可用的新基因来替换，跳过精炼。")
            return chromosome
            
        new_gene = random.choice(potential_additions)
        new_genes_list.append(new_gene)
        
        print(f"[精炼] ✓ 已将低贡献特征 {target_gene_to_remove.path} 替换为新特征 {new_gene.path}。")
        
        return ModelingChromosome(genes=new_genes_list)


if __name__ == "__main__":
    # 这是一个集成测试，它需要我们 V1.0 的所有依赖
    import semantic_inference
    from data_translator import KnowledgeGraphTranslator
    
    print("--- \"架构师\"模块 (architect.py) V1.0 独立集成测试 ---")
    
    # 1. (模拟) 运行"感知"
    class MockDB: pass
    schema_map = semantic_inference.run_semantic_inference(MockDB())
    
    # 2. (真实) 实例化"翻译官"
    TARGET_VARIABLE = "UserProfile.IsDefault" # 定义标准目标
    translator = KnowledgeGraphTranslator(
        inferred_schema=schema_map,
        physical_target_table='tbl_user_01',
        physical_target_column='is_default'
    )
    
    # 3. (真实) 实例化"基因生成器"
    gene_gen = GeneGenerator(translator, target_variable=TARGET_VARIABLE)
    gene_pool = gene_gen.generate_initial_pool()
    
    # 4. (真实) 实例化"演化引擎"
    evo_engine = EvolutionaryEngine(gene_pool)
    population = evo_engine.initialize_population(size=10) # 创建10个个体
    
    # 5. (真实) 实例化"特征引擎"和"评估器"
    feature_engine = FeatureEngine(translator, standard_target_variable=TARGET_VARIABLE)
    fitness_evaluator = FitnessEvaluator(feature_engine)
    
    # 6. (关键测试) 评估一个"染色体"
    print("\n--- 正在测试评估一个随机染色体 ---")
    test_chromosome = population[0]
    print(f"测试染色体: {test_chromosome.genes}")
    
    evaluation_result = fitness_evaluator.evaluate(test_chromosome)
    
    print("\n--- 评估结果 ---")
    print(json.dumps(evaluation_result, indent=2))
    
    assert evaluation_result['auc'] > 0.0 # 验证模型至少运行成功了
    assert evaluation_result['evaluation_time_ms'] > 0 # 验证时间被测量了
    
    print("\n--- \"架构师\"模块 V1.0 独立测试完毕 ---")
