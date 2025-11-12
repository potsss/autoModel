"""
对抗性共演化系统 - 主控制单元

这是系统的"大脑"，负责组装并运行完整的"对抗性共演化"系统。
它将所有模块整合在一起，实现最终的"综合得分对抗循环"。

V1.0 架构约定：
- 组装所有V1.0模块（感知、翻译、架构师、破坏者）
- 实现完整的对抗性共演化循环
- 动态调整适应度函数权重
- 提供详细的演化过程日志
"""

import json
import time
import random
import numpy as np
from typing import List, Dict, Any
from pathlib import Path  # 新增
import pandas as pd       # 新增
from schema_config import SchemaConfig  # 已有: 外部Schema配置支持
# 导入我们项目中的所有 V1.0 模块
from data_translator import KnowledgeGraphTranslator
from core_structures import ModelingChromosome
from architect import GeneGenerator, FeatureEngine, FitnessEvaluator, EvolutionaryEngine
from saboteur import EconomicsAttacker, CausalAttacker, SynthesisAttacker


class ControlUnit:
    """
    V1.1 "总指挥部"。
    负责组装并运行完整的"对抗性共演化"系统。
    """
    
    def __init__(self, physical_target_table: str, physical_target_column: str, dataframes: Dict[str, pd.DataFrame] = None, schema_config: SchemaConfig = None, feature_gen_config: Dict[str, Any] = None):
        print("[控制单元] 正在启动...")
        
        self.physical_target_table = physical_target_table
        self.physical_target_column = physical_target_column
        self.dataframes = dataframes or {}
        self.schema_config = schema_config

        # 新增: 运行语义推断以获取 schema_map
        try:
            from semantic_inference import run_semantic_inference
            # 关闭字段自动补全，避免为主表构建无关字段映射
            self.schema_map = run_semantic_inference(self.dataframes, schema_config=self.schema_config, autofill_fields=False)
        except Exception as e:
            print(f"[控制单元][警告] 语义推断失败: {e}")
            self.schema_map = {}

        # --- 1. (感知) 运行"语义推断" ---
        # schema_map = semantic_inference.run_semantic_inference(dataframes=dataframes)
        
        # --- 2. (V1.2 修正) 解耦：先查找标准目标，再实例化翻译官 ---
        target_info = KnowledgeGraphTranslator.get_standard_target_info(
            self.schema_map, physical_target_table, physical_target_column
        )
        self.target_variable = f"{target_info['entity']}.{target_info['field']}"

        # --- 3. (数据) 实例化"翻译官" ---
        self.translator = KnowledgeGraphTranslator(
            inferred_schema=self.schema_map,
            physical_target_table=physical_target_table,
            physical_target_column=physical_target_column,
            dataframes=dataframes,
            disable_entity_fallback=True
        )
        
        # --- 4. (创造) 实例化"架构师"的所有组件 ---
        print("[控制单元] 正在初始化\"架构师\"...")
        # 特征生成配置（外参化风格）：仅控制主表 LGBM 预筛选 Top-K
        self.feature_gen_config = feature_gen_config or {}
        self.gene_generator = GeneGenerator(self.translator, self.target_variable, feature_config=self.feature_gen_config)
        self.gene_pool = self.gene_generator.generate_initial_pool()
        
        self.evo_engine = EvolutionaryEngine(self.gene_pool)
        
        # (V1.1 修改) 传入目标变量
        self.feature_engine = FeatureEngine(self.translator, self.target_variable)
        
        self.fitness_evaluator = FitnessEvaluator(self.feature_engine)
        
        # --- 5. (批判) 实例化"破坏者"的所有组件 ---
        print("[控制单元] 正在初始化\"破坏者\"...")
        self.attackers = {
            "economics": EconomicsAttacker(self.translator, self.target_variable),
            "causal": CausalAttacker(self.translator, self.target_variable),
            "synthesis": SynthesisAttacker(self.translator, self.target_variable)
        }
        # --- 6. (控制) 初始化"适应度函数"的权重 ---
        self.fitness_weights = {
            'auc': 1.0, # AUC 的基础权重
            'economics': 0.0, # 经济惩罚的初始权重 (负数)
            'causal': 0.0, # 因果惩罚的初始权重
            'synthesis': 0.0 # 泛化惩罚的初始权重
        }
        
        print("[控制单元] 系统初始化完毕，准备就绪。")

    def run(self, generations: int, population_size: int, challenge_interval: int, evo_config: Dict[str, Any]):
        """
        运行"对抗性共演化"V1.0 主循环
        """
        print("\n--- V1.0 \"对抗性共演化\"开始 ---")

    # 1. 初始化种群（从外部传入 evo_config）
        population = self.evo_engine.initialize_population(population_size, config=evo_config)

        for gen in range(generations):
            print(f"\n--- 世代 {gen+1}/{generations} ---")

            evaluation_results = []  # 存储评估器返回的"评估字典"
            penalty_scores = []      # 存储破坏者返回的"惩罚字典"
            comprehensive_scores = []  # 存储最终的"综合得分"

            # 2. (核心) 评估与批判阶段
            #    我们对种群中的每一个个体进行"全面体检"
            print(f"正在评估和批判 {len(population)} 个个体...")
            start_gen_time = time.time()

            for chromosome in population:
                # 2a. (创造) 评估基础性能 (来自 FitnessEvaluator)
                eval_result = self.fitness_evaluator.evaluate(chromosome)
                evaluation_results.append(eval_result)

                # 2b. (批判) 计算所有惩罚分 (来自 Saboteur)
                penalties = {}
                for attacker_name, attacker_instance in self.attackers.items():
                    if self.fitness_weights.get(attacker_name, 0.0) != 0.0:
                        penalties[attacker_name] = attacker_instance.challenge(chromosome, eval_result)
                    else:
                        # 如果权重为0，则跳过API调用，直接赋0分
                        penalties[attacker_name] = 0.0
                penalty_scores.append(penalties)

                # 2c. (控制) 计算最终"综合得分"
                final_score = (
                    eval_result.get('auc', 0.0) * self.fitness_weights['auc']
                    + penalties['economics'] * self.fitness_weights['economics']
                    + penalties['causal'] * self.fitness_weights['causal']
                    + penalties['synthesis'] * self.fitness_weights['synthesis']
                )
                comprehensive_scores.append(final_score)

            gen_time = time.time() - start_gen_time
            print(f"世代评估完成，耗时: {gen_time:.2f} 秒")

            # 3. 打印本世代的详细日志
            all_base_aucs = [r.get('auc', 0.0) for r in evaluation_results]
            print(f"  [统计] 平均基础AUC: {np.mean(all_base_aucs):.4f} (最高: {max(all_base_aucs):.4f})")
            print(f"  [统计] 平均综合得分: {np.mean(comprehensive_scores):.4f} (最高: {max(comprehensive_scores):.4f})")

            # 4. (创造) 选择与演化阶段
            # (关键!) "选择"操作基于"综合得分"
            selected = self.evo_engine.select(population, comprehensive_scores)

            # 5. (创造) 生成下一代 (交叉与变异)
            # 修复：确保子代数量为 population_size-1，再加精英=population_size
            next_population = []
            while len(next_population) < (population_size - 1):
                p1, p2 = random.sample(selected, 2)
                child = self.evo_engine.crossover(p1, p2)
                child = self.evo_engine.mutate(child)
                next_population.append(child)

            # (精英保留策略)
            best_individual_index = np.argmax(comprehensive_scores)
            elite_chromosome = population[best_individual_index]

            # [V1.4] 精英学习：对本代冠军进行 SHAP 分析和精炼
            print(f"\n[精英学习] 对本代冠军 (AUC: {all_base_aucs[best_individual_index]:.4f}) 进行 SHAP 精炼...")
            elite_eval_result = self.fitness_evaluator.evaluate(elite_chromosome, calculate_shap=True)
            shap_values = elite_eval_result.get('shap_values')
            
            if shap_values:
                refined_elite = self.evo_engine.refine_chromosome(elite_chromosome, shap_values)
                next_population.append(refined_elite)
            else:
                print("[精英学习] 警告: 未能获取 SHAP 值，直接保留原版精英。")
                next_population.append(elite_chromosome)

            population = next_population

            # 6. (控制) 动态重构适应度函数 (周期性触发)
            if (gen + 1) % challenge_interval == 0:
                print(f"\n[控制单元] 触发\"权重动态更新\"！(对抗压力增加)")
                # 增加对因果和经济性的惩罚权重 (使其更负)
                self.fitness_weights['causal'] = round(self.fitness_weights['causal'] * 1.2, 2)
                self.fitness_weights['economics'] = round(self.fitness_weights['economics'] * 1.1, 2)
                print(f"[控制单元] 新权重: Causal={self.fitness_weights['causal']:.2f}, Economics={self.fitness_weights['economics']:.2f}")

        print("\n--- V1.0 \"对抗性共演化\"结束 ---")

        # 7. 找到并返回最终的冠军
        final_champion_index = np.argmax(comprehensive_scores)
        final_champion = population[final_champion_index]
        final_champion_eval = evaluation_results[final_champion_index]

        print(f"[控制单元] 最终冠军 (综合得分: {max(comprehensive_scores):.4f}):")
        print(f"  - 基础 AUC: {final_champion_eval.get('auc', 0.0):.4f}")
        print(f"  - 评估耗时: {final_champion_eval.get('evaluation_time_ms', 0.0):.0f} ms")
        print(f"  - 特征数量: {final_champion_eval.get('feature_count', 0)}")
        print(f"  - 最终基因:")

        # 使用 json.dumps 来优雅地打印
        print(json.dumps(final_champion, default=lambda o: o.__dict__, indent=4, ensure_ascii=False))
        return final_champion


if __name__ == "__main__":
    from pathlib import Path

    # --- 配置区域 ---
    # 1. 指定数据文件夹路径
    DATA_FOLDER_PATH = Path(r"dateset_test")  # 新增
    PHYSICAL_TARGET_TABLE = "application_train"              # 新增
    PHYSICAL_TARGET_COLUMN = "TARGET"                        # 新增
    SCHEMA_CONFIG_PATH = DATA_FOLDER_PATH / "schema_config.json"  # 新增: 外部Schema路径

    # 新增: 加载数据
    all_dataframes: Dict[str, pd.DataFrame] = {}
    if DATA_FOLDER_PATH.exists():
        for csv_path in DATA_FOLDER_PATH.glob("*.csv"):
            try:
                df = pd.read_csv(csv_path)
                all_dataframes[csv_path.stem] = df
                print(f"[数据] 已加载 {csv_path.stem} shape={df.shape}")
            except Exception as e:
                print(f"[数据][警告] 加载 {csv_path.name} 失败: {e}")
    else:
        raise FileNotFoundError(f"数据目录不存在: {DATA_FOLDER_PATH}")

    # 新增: 加载外部schema配置
    schema_config = None
    if SCHEMA_CONFIG_PATH.exists():
        try:
            schema_config = SchemaConfig(SCHEMA_CONFIG_PATH)
            print(f"[主控] ✓ 已加载外部Schema: {SCHEMA_CONFIG_PATH}")
        except Exception as e:
            print(f"[主控][警告] 外部Schema加载失败: {e}")
    else:
        print(f"[主控] 未找到外部Schema，回退到LLM推断: {SCHEMA_CONFIG_PATH}")

    # 新增: 启动控制单元
    # (V1.2 新增) 特征生成配置，使用比例以获得灵活性
    FEATURE_GEN_CONFIG = {
        "main_table": {
            "lgbm_top_k_single_ratio": 0.40,  # 单表场景下，筛选前 25% 的特征
            "lgbm_top_k_multi_ratio": 0.25   # 多表场景下，筛选前 15% 的特征
        }
    }
    control_unit = ControlUnit(
        physical_target_table=PHYSICAL_TARGET_TABLE,
        physical_target_column=PHYSICAL_TARGET_COLUMN,
        dataframes=all_dataframes,
        schema_config=schema_config,
        feature_gen_config=FEATURE_GEN_CONFIG
    )
    # 演化配置（示例，可按需修改）
    EVOLUTION_CONFIG = {
        "generations": 20,
        "population_size": 20,
        "challenge_interval": 2,
        "evo_feature_config": {
            # 调整特征比例下限/上限与最小底数，避免染色体过于精简
            # 原来 min_features_ratio=0.10 太小，导致最终染色体只有 few features
            "min_features_ratio": 0.20,
            "max_features_ratio": 0.60,
            "max_features_floor": 30
        }
    }
    control_unit.run(
        generations=EVOLUTION_CONFIG["generations"],
        population_size=EVOLUTION_CONFIG["population_size"],
        challenge_interval=EVOLUTION_CONFIG["challenge_interval"],
        evo_config=EVOLUTION_CONFIG["evo_feature_config"]
    )