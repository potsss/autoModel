"""
对抗性共演化系统 - 破坏者模块

这是我们的"判别器"智能体（"破坏者"），它将实现V1.0的真实批判逻辑。
它不再返回随机数，而是基于真实的时间和特征数量进行经济学批判，
并通过调用LLM占位符实现因果批判。

V1.0 架构约定：
- 实现真实的经济学攻击（基于时间和特征成本）
- 通过LLM占位符实现因果批判
- 提供标准化的惩罚分数
- 支持多种攻击策略
"""

import abc
import time
from typing import List, Dict, Any

# 导入我们的"DNA"和"接口"
from core_structures import ModelingChromosome, FeatureGene
from knowledge_graph_interface import KnowledgeGraphInterface

# (关键) 导入我们的"假LLM"
from llm_interface import llm_critique_causality


class BaseAttacker(abc.ABC):
    """
    V1.0 "破坏者"攻击模块的抽象基类（规范）。
    """
    def __init__(self, translator: KnowledgeGraphInterface, target_variable: str):
        """
        :param translator: "翻译官"接口，用于理解数据
        :param target_variable: 目标变量的标准名称, e.g., 'UserProfile.IsDefault'
        """
        self.translator = translator
        self.target_variable = target_variable
        super().__init__()

    @abc.abstractmethod
    def challenge(
        self, 
        chromosome: ModelingChromosome, 
        evaluation_result: Dict[str, Any]
    ) -> float:
        """
        [V1.0 抽象方法] 对一个已评估的策略发起挑战。
        
        :param chromosome: "架构师"提交的最优建模策略。
        :param evaluation_result: "评估器"返回的真实结果字典
               (e.g., {'auc': 0.8, 'evaluation_time_ms': 150.0, 'feature_count': 5})
        :return: 一个标准化的"惩罚分数" (0.0 为无惩罚, 1.0+ 为高惩罚)。
        """
        raise NotImplementedError


class EconomicsAttacker(BaseAttacker):
    """
    [V1.0 真实实现]
    挑战模型的"性价比"（资源成本效益）。
    """
    def __init__(self, translator: KnowledgeGraphInterface, target_variable: str):
        super().__init__(translator, target_variable)
        # (V1.0 简化：硬编码我们的"SLA/预算")
        self.time_budget_ms = 200.0  # 评估时间超过 200ms 开始惩罚
        self.feature_budget = 4      # 特征数量超过 4 个开始惩罚

    def challenge(
        self, 
        chromosome: ModelingChromosome, 
        evaluation_result: Dict[str, Any]
    ) -> float:
        
        # 1. 从评估结果中获取"真实成本"
        eval_time_ms = evaluation_result.get('evaluation_time_ms', 0)
        feature_count = evaluation_result.get('feature_count', 0)
        
        # 2. 计算"延迟惩罚"
        # (简化逻辑：每超出预算100ms，惩罚分 +0.1)
        time_overrun = max(0, eval_time_ms - self.time_budget_ms)
        time_penalty = (time_overrun / 1000.0) * 0.1 
        
        # 3. 计算"特征成本惩罚"
        # (简化逻辑：每多一个特征，惩罚分 +0.05)
        feature_overrun = max(0, feature_count - self.feature_budget)
        feature_penalty = feature_overrun * 0.05
        
        total_penalty = time_penalty + feature_penalty
        
        # print(f"  [破坏者-经济] 耗时: {eval_time_ms:.0f}ms, 特征: {feature_count}个. 惩罚: {total_penalty:.3f}")
        return total_penalty


class CausalAttacker(BaseAttacker):
    """
    [V1.0 真实实现]
    挑战模型的逻辑合理性，通过调用 LLM (占位符) 来识别伪关联。
    """
    def challenge(
        self, 
        chromosome: ModelingChromosome, 
        evaluation_result: Dict[str, Any]
    ) -> float:
        
        # 1. 从染色体中解析出所有用到的"特征路径"
        feature_genes = [g for g in chromosome.genes if isinstance(g, FeatureGene)]
        feature_paths = [g.path for g in feature_genes]
        
        if not feature_paths:
            return 0.0 # 没有特征，没有因果风险

        # 2. (关键) 真实地调用 LLM 占位符
        try:
            critique = llm_critique_causality(
                feature_list=feature_paths,
                target_variable=self.target_variable
            )
            
            # 3. 返回 LLM 给出的"风险评分"作为惩罚
            penalty = critique.get('risk_score', 0.0)
            # if penalty > 0:
            #     print(f"  [破坏者-因果] 发现风险! 惩罚: {penalty:.3f}. 理由: {critique.get('justification')}")
            return penalty
            
        except Exception as e:
            print(f"[破坏者-因果-错误] LLM 占位符调用失败: {type(e)} - {e}")
            return 0.1 # 给予一个小的默认惩罚


class SynthesisAttacker(BaseAttacker):
    """
    [V1.0 占位符]
    (未来职责) 挑战模型在边缘和未知数据上的泛化能力。
    (当前职责) 暂不实现，返回0。
    """
    def challenge(
        self, 
        chromosome: ModelingChromosome, 
        evaluation_result: Dict[str, Any]
    ) -> float:
        
        # V1.0 暂不实现此高级功能
        return 0.0 


if __name__ == "__main__":
    import semantic_inference
    from data_translator import KnowledgeGraphTranslator # 我们需要一个"翻译官"
    
    print("--- \"破坏者\"模块 (saboteur.py) V1.0 独立集成测试 ---")

    # 1. (准备) 实例化一个"翻译官" (因为 BaseAttacker 的 __init__ 需要)
    class MockDB: pass
    schema_map = semantic_inference.run_semantic_inference(MockDB())
    translator = KnowledgeGraphTranslator(inferred_schema=schema_map)
    
    # 2. (准备) 定义目标变量
    TARGET = "UserProfile.IsDefault"

    # 3. (准备) 实例化所有 V1.0 攻击者
    econ_attacker = EconomicsAttacker(translator, TARGET)
    causal_attacker = CausalAttacker(translator, TARGET)
    synth_attacker = SynthesisAttacker(translator, TARGET)

    # 4. (测试用例 1) 一个"廉价且合理"的染色体
    print("\n--- 测试用例 1: 廉价且合理 ---")
    chromo_good = ModelingChromosome(genes=[
        FeatureGene(op="LATEST", path="UserProfile.AnnualIncome")
    ])
    eval_good = {'auc': 0.8, 'evaluation_time_ms': 50.0, 'feature_count': 1}
    
    p_econ_1 = econ_attacker.challenge(chromo_good, eval_good)
    p_causal_1 = causal_attacker.challenge(chromo_good, eval_good)
    print(f"  -> 经济惩罚: {p_econ_1:.3f} (预期: 0.0)")
    print(f"  -> 因果惩罚: {p_causal_1:.3f} (预期: 0.1 左右)")
    assert p_econ_1 == 0.0

    # 5. (测试用例 2) 一个"昂贵且有风险"的染色体
    print("\n--- 测试用例 2: 昂贵且有风险 ---")
    chromo_bad = ModelingChromosome(genes=[
        FeatureGene(op="LATEST", path="UserProfile.AnnualIncome"),
        FeatureGene(op="LATEST", path="UserProfile.Gender"), # 'Gender' 会触发因果惩罚
        FeatureGene(op="AVG", path="UserTransaction.TransactionAmount", window=30),
        FeatureGene(op="COUNT", path="UserTransaction.TransactionID", window=30),
        FeatureGene(op="LATEST", path="UserProfile.RegistrationDate"),
    ])
    eval_bad = {'auc': 0.85, 'evaluation_time_ms': 350.0, 'feature_count': 5} # 耗时且特征多
    
    p_econ_2 = econ_attacker.challenge(chromo_bad, eval_bad)
    p_causal_2 = causal_attacker.challenge(chromo_bad, eval_bad)
    print(f"  -> 经济惩罚: {p_econ_2:.3f} (预期: > 0.0)")
    print(f"  -> 因果惩罚: {p_causal_2:.3f} (预期: 0.4)")
    assert p_econ_2 > 0.0
    assert p_causal_2 == 0.4
    
    print("\n--- \"破坏者\"模块 V1.0 独立测试完毕 ---")
