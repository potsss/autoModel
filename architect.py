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
# 统一禁止进入建模特征的字段（防泄漏）
EXCLUDED_COLUMNS = {"duration"}  # 可按需扩展，比如 {"duration", "y"}
# 新增：兜底所需的 sklearn 组件
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 导入我们项目中的其他模块
from core_structures import ModelingGene, FeatureGene, TransformGene, ModelGene, FilterGene, ModelingChromosome
from knowledge_graph_interface import KnowledgeGraphInterface
from llm_interface import llm_generate_genes


class GeneGenerator:
    """
    负责调用 LLM (占位符) 来动态生成初始基因池。
    """
    def __init__(self, translator: KnowledgeGraphInterface, target_variable: str):
        self.translator = translator
        self.target_variable = target_variable

    def generate_initial_pool(self) -> List[ModelingGene]:
        """
        调用 LLM 占位符获取基因创意；若 LLM 返回无效/为空，则从目标实体自动构造 LATEST 特征。
        始终追加至少一个 ModelGene，保证后续遗传流程能启动。
        """
        print("[架构师-基因] 正在调用 LLM (占位符) 生成初始基因池...")

        # 1) 先尝试从 LLM 拿结果（可能是 list[dict]，也可能异常/无效）
        try:
            standard_schema = self.translator.get_standard_schema()
        except Exception as e:
            standard_schema = {}
            print(f"[架构师-基因][警告] 获取标准模式失败: {e}")

        genes_json = None
        try:
            genes_json = llm_generate_genes(standard_schema, self.target_variable)
        except Exception as e:
            print(f"[架构师-基因][警告] LLM 生成基因异常: {e}")

        # 2) 尝试把 LLM 结果转成合法的 FeatureGene 列表
        gene_pool: List[ModelingGene] = []
        def _safe_add_feature(op: str, path: str, window=None):
            if not isinstance(op, str) or not isinstance(path, str):
                return
            if '.' not in path:
                return
            if op not in ['AVG', 'COUNT', 'SUM', 'LATEST', 'MAX', 'MIN']:
                return
            gene_pool.append(FeatureGene(op=op, path=path, window=window))

        if isinstance(genes_json, list):
            for g in genes_json:
                try:
                    op = g.get('op')
                    path = g.get('path')
                    # 跳过无效记录
                    if not op or not path or '.' not in path:
                        continue
                    # 过滤掉命中 EXCLUDED_COLUMNS 的字段
                    try:
                        _, field = path.split('.', 1)
                    except Exception:
                        continue
                    if field in EXCLUDED_COLUMNS:
                        print(f"[基因过滤] 跳过被禁用字段: {path}")
                        continue
                    if op in ['AVG', 'COUNT', 'SUM', 'LATEST', 'MAX', 'MIN']:
                        gene_pool.append(FeatureGene(
                            op=op,
                            path=path,
                            window=g.get('window')
                        ))
                except Exception:
                    continue
        else:
            # 有些实现会在 llm_interface 里直接记录“无效 JSON”但仍返回部分结构；此处忽略，走兜底
            print("[架构师-基因][提示] LLM 未返回可用的特征列表，准备进入兜底。")

        # 3) 如果 LLM 没产出任何有效 FeatureGene，则从目标实体自动造一批 LATEST 特征
        if len([g for g in gene_pool if isinstance(g, FeatureGene)]) == 0:
            try:
                # 从 "BankRecord" 或 target_variable 指定的实体 选列
                tgt_entity = None
                tgt_field = None
                if isinstance(self.target_variable, str) and '.' in self.target_variable:
                    tgt_entity, tgt_field = self.target_variable.split('.', 1)
                else:
                    # 缺省走 BankRecord.y
                    tgt_entity, tgt_field = 'BankRecord', 'y'

                df_ent = self.translator.get_entity_dataframe(tgt_entity)
                if getattr(df_ent, "empty", True):
                    # 如果目标实体取不到，再从已知的单表里取
                    if hasattr(self.translator, "db_tables") and isinstance(self.translator.db_tables, dict) and len(self.translator.db_tables) == 1:
                        df_ent = list(self.translator.db_tables.values())[0]
                        # 尝试从 schema 找实体名；不行就继续用 BankRecord
                        tgt_entity = tgt_entity or 'BankRecord'

                if df_ent is not None and not getattr(df_ent, "empty", True):
                    # 选前若干列作为 LATEST 特征（排除目标列）
                    cols = [c for c in df_ent.columns if c != tgt_field]
                    # 适当限制数量，避免过多：这里取前 8 列
                    for c in cols[:8]:
                        _safe_add_feature("LATEST", f"{tgt_entity}.{c}", None)
                    print(f"[架构师-基因][兜底] 从实体 {tgt_entity} 自动构造 LATEST 特征 {min(8, len(cols))} 个。")
                else:
                    print("[架构师-基因][兜底-警告] 无法获取任何实体数据，后续将仅追加模型基因。")
            except Exception as e:
                print(f"[架构师-基因][兜底-异常] 自动构造 LATEST 特征失败: {e}")

        # 4) 无论如何，确保有多个模型基因
        model_configs = [
            {"alg": "LogisticRegression", "params": {"solver": "liblinear", "class_weight": "balanced", "random_state": 42}},
            {"alg": "RandomForestClassifier", "params": {"n_estimators": 100, "max_depth": 10, "random_state": 42}},
            {"alg": "LGBMClassifier", "params": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31, "random_state": 42, "verbose": -1}}
        ]
        for config in model_configs:
            gene_pool.append(ModelGene(alg=config["alg"], params=config["params"]))

        # 5) 去重（以 path/op/window 为键），防止重复
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


        # 2) X 初始化
        X = pd.DataFrame(index=base_df.index)

        # 3) 解析 FeatureGene（允许 LATEST 直接从当前目标实体取值）
        feature_genes = [g for g in chromosome.genes if isinstance(g, FeatureGene)]
        if not feature_genes:
            # 没有任何基因：直接走“单表全列”兜底
            return self._build_baseline_X(base_df, self.target_field)

        for gene in feature_genes:
            try:
                entity_name, field_name = gene.path.split('.', 1)
            except Exception:
                continue

            feature_name = f"{gene.op}_{entity_name}_{field_name}"
            if gene.window:
                feature_name += f"_{gene.window}d"
            if feature_name in X.columns:
                continue

            if gene.op == 'LATEST':
                # 允许从当前目标实体直接取值，但先过滤被禁用字段
                if entity_name == self.target_entity and field_name in base_df.columns:
                    if field_name in EXCLUDED_COLUMNS:
                        print(f"[特征引擎] 跳过被禁用字段(LATEST): {field_name}")
                        continue
                    X[feature_name] = base_df[field_name]
                continue
            elif gene.op in ['AVG', 'COUNT', 'SUM', 'MAX', 'MIN']:
                # 跨表（在只有单表的 CSV 场景，这里基本会跳过）
                relation_key = f"{entity_name}_to_UserProfile"
                if relation_key not in self.relationships:
                    print(f"[特征引擎-警告] 找不到关系 {relation_key}，跳过 {gene.path}")
                    continue

                relation = self.relationships[relation_key]
                fk_from = relation['from_key']
                pk_to = relation['to_key']

                fact_df = self.translator.get_entity_dataframe(entity_name)
                if fact_df is None or fact_df.empty or field_name not in fact_df.columns:
                    continue

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
                    X[feature_name] = merged[feature_name].values
                except Exception as merge_error:
                    print(f"[特征引擎-警告] 合并失败: {merge_error}")
                    continue

        # 4) 清理 & 判空
        X = X.dropna(axis=1, how='all')
        if X is None or X.shape[1] == 0:
            # 还是没产出特征 → 兜底到“单表全列”
            return self._build_baseline_X(base_df, self.target_field)

        # 5) 列类型拆分
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

    def evaluate(self, chromosome: ModelingChromosome) -> Dict[str, Any]:
        """
        [V1.1] 执行完整的"评估"流程 (使用5-折交叉验证)，并返回一个包含"分数"和"成本"的字典。
        """
        start_time = time.time()
        
        try:
            # 1) 特征构造
            X, y, num_features, cat_features = self.feature_engine.build_features(chromosome)

            # 检查 y 是否包含多于一个类别
            if y.nunique() < 2:
                print(f"[评估错误] 目标变量 y 只包含一个类别，无法进行评估。")
                return {'auc': 0.0, 'evaluation_time_ms': (time.time() - start_time) * 1000, 'error': 'Target variable has less than 2 classes.'}

            # 双保险：如果上游误入任何被禁用列，这里再一次剔除
            for bad_col in list(EXCLUDED_COLUMNS):
                if hasattr(X, "columns") and bad_col in X.columns:
                    print(f"[评估器] 发现被禁用列残留，已移除: {bad_col}")
                    X = X.drop(columns=[bad_col])
                    if bad_col in num_features:
                        num_features.remove(bad_col)
                    if bad_col in cat_features:
                        cat_features.remove(bad_col)

            # --- DEBUG: feature matrix & target overview ---
            try:
                # 使用文件顶部全局导入的 pandas/numpy
                print(f"[调试] X type={type(X)}, shape={getattr(X, 'shape', None)}")
                y_series = y if isinstance(y, pd.Series) else pd.Series(y)
                print(f"[调试] y分布: {y_series.value_counts(dropna=False).to_dict()}")
                print(f"[调试] num_features={len(num_features)}, cat_features={len(cat_features)}")
                if hasattr(X, "head"):
                    print(f"[调试] X前5列: {list(X.columns)[:5]}")
            except Exception as _dbg_e:
                print(f"[调试] 打印特征信息失败: {str(_dbg_e)}")

            # 运行时断言，确保 duration 已剔除
            print(f"[调试] X列样本(前5): {list(X.columns)[:5]}")
            assert "duration" not in getattr(X, "columns", []), "duration 未被剔除"

            # 2. 预处理流水线
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, num_features),
                    ('cat', categorical_transformer, cat_features)
                ],
                remainder='drop'
            )

            # 3. 模型
            model_gene = next((g for g in chromosome.genes if isinstance(g, ModelGene)), None)
            if model_gene:
                if model_gene.alg == 'LogisticRegression':
                    model = LogisticRegression(**model_gene.params)
                elif model_gene.alg == 'RandomForestClassifier':
                    model = RandomForestClassifier(**model_gene.params)
                elif model_gene.alg == 'LGBMClassifier':
                    model = LGBMClassifier(**model_gene.params)
                else: # Fallback
                    print(f"[警告] 未知的模型算法: {model_gene.alg}，使用默认的 LogisticRegression。")
                    model = LogisticRegression(solver='liblinear', random_state=42)
            else: # Fallback if no model gene
                print(f"[警告] 染色体中没有模型基因，使用默认的 LogisticRegression。")
                model = LogisticRegression(solver='liblinear', random_state=42)

            # 4. Pipeline
            model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])

            # Normalize y dtype if possible（使用全局 np/pd）
            try:
                if not isinstance(y, np.ndarray):
                    y = np.asarray(y)
                if y.dtype.kind not in ('i', 'u'):
                    y_unique = pd.Series(y).dropna().unique().tolist()
                    y_map = {"1":1,"1.0":1,"True":1,"true":1,"Yes":1,"yes":1,
                             "0":0,"0.0":0,"False":0,"false":0,"No":0,"no":0}
                    if set(map(str, y_unique)).issubset(set(y_map.keys())):
                        y = pd.Series(y).map(y_map).fillna(0).astype(int).values
            except Exception as _dtype_e:
                print(f"[调试] y类型规范化失败: {str(_dtype_e)}")

            # 5. [V1.1] K-折交叉验证
            print("[调试] 开始5-折交叉验证...")
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            # 6/7. 训练与评估 (使用 cross_val_score)
            try:
                scores = cross_val_score(model_pipeline, X, y, cv=kfold, scoring='roc_auc')
                auc_mean = np.mean(scores)
                auc_std = np.std(scores)
                print(f"[调试] 交叉验证完成，AUCs={np.round(scores, 4)}, Mean AUC={auc_mean:.4f}, Std={auc_std:.4f}")
            except Exception as _fit_e:
                print(f"[调试] 交叉验证阶段异常: {str(_fit_e)}")
                if 'only one class' in str(_fit_e):
                    return {'auc': 0.0, 'evaluation_time_ms': (time.time() - start_time) * 1000, 'error': str(_fit_e)}
                raise

            evaluation_time_ms = (time.time() - start_time) * 1000
            return {
                'auc': auc_mean,  # 主适应度分数
                'auc_mean': auc_mean,
                'auc_std': auc_std,
                'evaluation_time_ms': evaluation_time_ms, 
                'feature_count': len(num_features) + len(cat_features)
            }

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

    def initialize_population(self, size: int) -> List[ModelingChromosome]:
        """V1.1 智能初始化 (动态特征数量)：确保染色体可被评估"""
        population = []
        if not self.feature_genes or not self.model_genes:
            raise ValueError("基因池中缺少必要的 FeatureGene 或 ModelGene！")

        total_features = len(self.feature_genes)
        # 根据基因池大小，动态决定初始特征数量的范围
        min_count = max(1, int(total_features * 0.1))
        max_count = min(total_features, max(min_count, int(total_features * 0.3)))
        print(f"[演化引擎] 动态初始化特征数范围: [{min_count}, {max_count}] (总特征池: {total_features})")

        for _ in range(size):
            genes = []
            # 从动态范围中随机选择特征数量
            num_features = random.randint(min_count, max_count)
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
        """V1.0 单点交叉（简化版）"""
        # (简化：只交换 FeatureGene，保持 ModelGene 不变)
        p1_features = [g for g in parent1.genes if isinstance(g, FeatureGene)]
        p2_features = [g for g in parent2.genes if isinstance(g, FeatureGene)]
        model_gene = next(g for g in parent1.genes if isinstance(g, ModelGene)) # 继承父1的模型
        
        # 确保有足够的特征进行交叉
        if len(p1_features) <= 1 or len(p2_features) <= 1:
            # 如果特征太少，直接组合所有特征
            child_features = p1_features + p2_features
        else:
            cut = random.randint(1, min(len(p1_features), len(p2_features)) - 1)
            child_features = p1_features[:cut] + p2_features[cut:]
        
        return ModelingChromosome(genes=child_features + [model_gene])

    def mutate(self, chromosome: ModelingChromosome) -> ModelingChromosome:
        """V1.1 变异 (增加/删除/替换)"""
        if random.random() < 0.3:  # 变异率提升到 30%
            genes = chromosome.genes[:]  # 操作副本
            mutation_roll = random.random()

            # 20% 概率增加一个特征
            if mutation_roll < 0.2 and self.feature_genes:
                current_features = {g for g in genes if isinstance(g, FeatureGene)}
                potential_additions = [g for g in self.feature_genes if g not in current_features]
                if potential_additions:
                    genes.append(random.choice(potential_additions))
                    print("[变异] 增加基因")
                    return ModelingChromosome(genes=genes)

            # 20% 概率删除一个特征
            elif mutation_roll < 0.4:
                feature_genes_in_chromo = [g for g in genes if isinstance(g, FeatureGene)]
                if len(feature_genes_in_chromo) > 1:  # 至少保留一个特征
                    gene_to_remove = random.choice(feature_genes_in_chromo)
                    genes.remove(gene_to_remove)
                    print("[变异] 删除基因")
                    return ModelingChromosome(genes=genes)

            # 60% 概率替换一个基因 (原逻辑)
            else:
                if not genes: return chromosome
                idx_to_mutate = random.randrange(len(genes))
                gene_to_mutate = genes[idx_to_mutate]

                if isinstance(gene_to_mutate, FeatureGene) and self.feature_genes:
                    genes[idx_to_mutate] = random.choice(self.feature_genes)
                    print("[变异] 替换特征基因")
                elif isinstance(gene_to_mutate, ModelGene) and self.model_genes:
                    genes[idx_to_mutate] = random.choice(self.model_genes)
                    print("[变异] 替换模型基因")
                return ModelingChromosome(genes=genes)

        return chromosome


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
