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
import pandas as pd
# 导入我们项目中的所有 V1.0 模块
import semantic_inference
from data_translator import KnowledgeGraphTranslator
from core_structures import ModelingChromosome
from architect import GeneGenerator, FeatureEngine, FitnessEvaluator, EvolutionaryEngine
from saboteur import EconomicsAttacker, CausalAttacker, SynthesisAttacker


class ControlUnit:
    """
    V1.1 "总指挥部"。
    负责组装并运行完整的"对抗性共演化"系统。
    """
    
    def __init__(self, physical_target_table: str, physical_target_column: str, dataframes: Dict[str, pd.DataFrame] = None):
        print("[控制单元] 正在启动...")
        
        # --- 1. (感知) 运行"语义推断" ---
        schema_map = semantic_inference.run_semantic_inference(dataframes=dataframes)
        
        # --- 2. (V1.2 修正) 解耦：先查找标准目标，再实例化翻译官 ---
        target_info = KnowledgeGraphTranslator.get_standard_target_info(
            schema_map, physical_target_table, physical_target_column
        )
        self.target_variable = f"{target_info['entity']}.{target_info['field']}"

        # --- 3. (数据) 实例化"翻译官" ---
        self.translator = KnowledgeGraphTranslator(
            inferred_schema=schema_map,
            physical_target_table=physical_target_table,
            physical_target_column=physical_target_column,
            dataframes=dataframes
        )
        
        # --- 4. (创造) 实例化"架构师"的所有组件 ---
        print("[控制单元] 正在初始化\"架构师\"...")
        self.gene_generator = GeneGenerator(self.translator, self.target_variable)
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
            'economics': -0.1, # 经济惩罚的初始权重 (负数)
            'causal': -0.2, # 因果惩罚的初始权重
            'synthesis': -0.1 # 泛化惩罚的初始权重
        }
        
        print("[控制单元] 系统初始化完毕，准备就绪。")

    def run(self, generations: int, population_size: int, challenge_interval: int):
        """
        运行"对抗性共演化"V1.0 主循环
        """
        print("\n--- V1.0 \"对抗性共演化\"开始 ---")
        
        # 1. 初始化种群
        population = self.evo_engine.initialize_population(population_size)
        
        for gen in range(generations):
            print(f"\n--- 世代 {gen+1}/{generations} ---")
            
            evaluation_results = [] # 存储评估器返回的"评估字典"
            penalty_scores = []     # 存储破坏者返回的"惩罚字典"
            comprehensive_scores = [] # 存储最终的"综合得分"

            # 2. (核心) 评估与批判阶段
            #    我们对种群中的每一个个体进行"全面体检"
            print(f"正在评估和批判 {len(population)} 个个体...")
            start_gen_time = time.time()
            
            for chromosome in population:
                
                # 2a. (创造) 评估基础性能 (来自 FitnessEvaluator)
                # eval_result = {'auc': 0.8, 'evaluation_time_ms': 150.0, 'feature_count': 5}
                eval_result = self.fitness_evaluator.evaluate(chromosome)
                evaluation_results.append(eval_result)
                
                # 2b. (批判) 计算所有惩罚分 (来自 Saboteur)
                penalties = {
                    "economics": self.attackers["economics"].challenge(chromosome, eval_result),
                    "causal": self.attackers["causal"].challenge(chromosome, eval_result),
                    "synthesis": self.attackers["synthesis"].challenge(chromosome, eval_result)
                }
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
            next_population = []
            for i in range(0, population_size - 1, 2): # 保留一个空位给精英
                p1, p2 = random.sample(selected, 2)
                child = self.evo_engine.crossover(p1, p2)
                child = self.evo_engine.mutate(child)
                next_population.append(child)
            
            # (精英保留策略)
            best_individual_index = np.argmax(comprehensive_scores)
            next_population.append(population[best_individual_index])
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
    # 1. 指定数据文件夹的路径
    #    使用 Path('.') 表示当前文件夹
    DATA_FOLDER_PATH = Path(__file__).parent / "dataset_bank"

    # 2. 指定包含目标预测列的表名（必须与CSV文件名一致，不含.csv后缀）
    PHYSICAL_TARGET_TABLE = "bank-additional-full"
    
    # 3. 指定目标预测列的列名
    PHYSICAL_TARGET_COLUMN = "y"

    # --- 数据自动加载逻辑 ---
    all_dataframes = {}
    print(f"[数据] 正在从文件夹 '{DATA_FOLDER_PATH}' 加载所有CSV文件...")

    csv_files = list(DATA_FOLDER_PATH.glob("*.csv"))
    assert len(csv_files) > 0, f"在文件夹 '{DATA_FOLDER_PATH}' 中未找到任何CSV文件。"

    for csv_path in csv_files:
        table_name = csv_path.stem  # 使用文件名（不含后缀）作为表名
        try:
            # 尝试用不同的分隔符读取，以提高兼容性
            df = pd.read_csv(csv_path, sep=";", encoding="utf-8")
        except Exception:
            df = pd.read_csv(csv_path, encoding="utf-8")

        # 如果当前表是目标表，特殊处理目标列
        if table_name == PHYSICAL_TARGET_TABLE:
            if PHYSICAL_TARGET_COLUMN in df.columns and df[PHYSICAL_TARGET_COLUMN].dtype == "object":
                df[PHYSICAL_TARGET_COLUMN] = (df[PHYSICAL_TARGET_COLUMN].astype(str).str.lower() == "yes").astype(int)
        
        all_dataframes[table_name] = df
        print(f"  [数据] 已加载表 '{table_name}'，shape={df.shape}")

    # --- 运行前检查 ---
    assert PHYSICAL_TARGET_TABLE in all_dataframes, f"目标表 '{PHYSICAL_TARGET_TABLE}' 未在数据文件夹中找到。"
    main_df = all_dataframes[PHYSICAL_TARGET_TABLE]
    assert PHYSICAL_TARGET_COLUMN in main_df.columns, f"目标列 '{PHYSICAL_TARGET_COLUMN}' 在表 '{PHYSICAL_TARGET_TABLE}' 中不存在。"

    # 打印目标列的分布
    print(f"[数据] 目标表 '{PHYSICAL_TARGET_TABLE}' 中，目标列 '{PHYSICAL_TARGET_COLUMN}' 的分布情况：")
    print(main_df[PHYSICAL_TARGET_COLUMN].value_counts().to_dict())

    # --- 系统启动 ---
    # 实例化"总指挥部"，并传入包含所有表的字典
    control_unit = ControlUnit(
        physical_target_table=PHYSICAL_TARGET_TABLE, 
        physical_target_column=PHYSICAL_TARGET_COLUMN,
        dataframes=all_dataframes
    )
    
    # 运行共演化
    control_unit.run(
        generations=3,          # 运行 3 个世代 (用于快速测试)
        population_size=10,      # 每代 10 个个体
        challenge_interval=2     # 每 2 个世代增加一次对抗压力
    )