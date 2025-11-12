from pathlib import Path
import json
import sys

def main():
    root = Path(__file__).parent
    data_dir = Path(r"F:\dataset\hcdr_subset\test")
    schema_path = data_dir / "schema_config.json"

    print(f"[HC] 项目根: {root}")
    print(f"[HC] 数据目录: {data_dir}")
    print(f"[HC] Schema文件: {schema_path}")

    # 1) 模块导入
    try:
        import pandas as pd
        print("[HC] ✓ pandas 可用")
    except Exception as e:
        print(f"[HC] ✗ pandas 不可用: {e}")
        sys.exit(1)

    try:
        from schema_config import SchemaConfig
        print("[HC] ✓ schema_config.SchemaConfig 可导入")
    except Exception as e:
        print(f"[HC] ✗ 无法导入 SchemaConfig: {e}")
        sys.exit(1)

    try:
        from semantic_inference import run_semantic_inference
        print("[HC] ✓ semantic_inference.run_semantic_inference 可导入")
    except Exception as e:
        print(f"[HC] ✗ 无法导入 run_semantic_inference: {e}")
        sys.exit(1)

    # 2) 加载数据（只取前几千行加速）
    if not data_dir.exists():
        print(f"[HC] ✗ 数据目录不存在: {data_dir}")
        sys.exit(1)

    all_dataframes = {}
    import pandas as pd
    for csv in data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv, nrows=5000)
            all_dataframes[csv.stem] = df
            print(f"[HC] ✓ 载入 {csv.name} shape={df.shape}")
        except Exception as e:
            print(f"[HC] ✗ 载入失败 {csv.name}: {e}")

    if not all_dataframes:
        print("[HC] ✗ 未加载到任何CSV")
        sys.exit(1)

    # 3) 加载外部schema（如果存在）
    schema_cfg = None
    if schema_path.exists():
        try:
            schema_cfg = SchemaConfig(schema_path)
            print("[HC] ✓ 已加载外部Schema配置")
        except Exception as e:
            print(f"[HC] ✗ 外部Schema加载失败: {e}")

    # 4) 运行语义推断
    try:
        schema_map = run_semantic_inference(all_dataframes, schema_config=schema_cfg)
        print(f"[HC] ✓ 语义推断完成，实体数: {len(schema_map)}")
        print("[HC] 实体预览:", list(schema_map.keys())[:5])
        # 基本校验：目标表是否存在
        if "application_train" not in schema_map:
            print("[HC] ! 警告: schema_map 中没有 application_train")
        # 打印一个表的字段数
        k = next(iter(schema_map.keys()))
        fields = schema_map[k].get("fields", {})
        print(f"[HC] 实体 `{k}` 字段数: {len(fields)}")
    except Exception as e:
        print(f"[HC] ✗ 语义推断失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()