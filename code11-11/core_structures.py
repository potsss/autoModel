"""
对抗性共演化系统 - 核心数据结构定义

这个文件定义了系统的"DNA"，即 ModelingGene (基因) 和 ModelingChromosome (染色体)。
这些结构构成了整个演化系统的基础数据单位。

V1.0 架构约定：
- 所有数据路径使用标准业务名称（如 UserProfile.AnnualIncome）
- 基因按逻辑顺序排列，形成完整的建模流程
- 支持特征工程、数据变换、模型选择和过滤操作
"""

import dataclasses
import abc
from typing import List, Dict, Optional, Any


@dataclasses.dataclass(frozen=True)
class ModelingGene(abc.ABC):
    """建模基因的抽象基类"""
    pass


@dataclasses.dataclass(frozen=True)
class FeatureGene(ModelingGene):
    """
    特征基因：定义一个特征的提取方式。
    这是特征工程的核心。
    """
    op: str  # 操作, 例如: 'LATEST' (直接获取), 'AVG', 'COUNT', 'SUM' (聚合)
    path: str  # 标准业务路径, 例如 'UserProfile.AnnualIncome' 或 'UserTransaction.TransactionAmount'
    window: Optional[int] = None  # 可选的时间窗口, 例如 30 (天)，用于聚合操作

    def __repr__(self):
        if self.window:
            return f"{self.op}({self.path}, window={self.window}d)"
        return f"{self.op}({self.path})"


@dataclasses.dataclass(frozen=True)
class TransformGene(ModelingGene):
    """
    变换基因：定义一个数据变换操作。
    它作用于一个或多个 FeatureGene 的输出。
    """
    op: str  # 操作, 例如 'Logarithm', 'StandardScaler', 'OneHotEncoder'
    inputs: List[str]  # 该变换所作用的输入特征的标准路径 (例如 ['UserProfile.AnnualIncome'])

    def __repr__(self):
        return f"{self.op}(on {self.inputs})"


@dataclasses.dataclass(frozen=True)
class ModelGene(ModelingGene):
    """模型基因：定义使用的算法和超参数"""
    alg: str  # 算法名称, 例如 'LogisticRegression', 'XGBoost'
    params: Optional[Dict[str, Any]] = dataclasses.field(default_factory=dict)

    def __repr__(self):
        return f"Model({self.alg})"


@dataclasses.dataclass(frozen=True)
class FilterGene(ModelingGene):
    """
    过滤基因：定义数据的筛选条件。
    这将作为特征工程的第一步。
    """
    # 条件也使用标准业务名称
    condition: str  # 例如 "UserProfile.Age > 18" 或 "UserTransaction.Timestamp > '2024-01-01'"

    def __repr__(self):
        return f"Filter({self.condition})"


@dataclasses.dataclass
class ModelingChromosome:
    """
    建模策略染色体：一个完整的、可执行的建模流程。
    它由一系列按逻辑顺序排列的基因构成。
    """
    genes: List[ModelingGene]

    def __repr__(self):
        gene_reprs = [str(gene) for gene in self.genes]
        return f"ModelingChromosome([{', '.join(gene_reprs)}])"


if __name__ == "__main__":
    # 演示 V1.0 染色体:
    # 1. 筛选18岁以上的用户
    # 2. 获取他们的"年收入"
    # 3. 计算他们"近30天平均交易额"
    # 4. 对"年收入"进行对数变换
    # 5. 使用XGBoost建模

    gene_filter = FilterGene(condition="UserProfile.Age > 18")
    gene_f1 = FeatureGene(op="LATEST", path="UserProfile.AnnualIncome")
    gene_f2 = FeatureGene(op="AVG", path="UserTransaction.TransactionAmount", window=30)
    gene_t1 = TransformGene(op="Logarithm", inputs=["UserProfile.AnnualIncome"])
    gene_model = ModelGene(alg="XGBoost", params={"n_estimators": 100})

    my_chromosome = ModelingChromosome(
        genes=[gene_filter, gene_f1, gene_f2, gene_t1, gene_model]
    )

    print("--- V1.0 核心数据结构演示 ---")
    print("创建了一个 V1.0 建模策略染色体:")
    print(my_chromosome)

    print("\n基因列表:")
    for gene in my_chromosome.genes:
        print(f"  - {gene}")

    print("\n--- 基因类型验证 ---")
    print(f"基因总数: {len(my_chromosome.genes)}")
    print(f"过滤基因: {isinstance(gene_filter, FilterGene)}")
    print(f"特征基因: {isinstance(gene_f1, FeatureGene)}")
    print(f"变换基因: {isinstance(gene_t1, TransformGene)}")
    print(f"模型基因: {isinstance(gene_model, ModelGene)}")
    print(f"所有基因都是 ModelingGene 的子类: {all(isinstance(gene, ModelingGene) for gene in my_chromosome.genes)}")
