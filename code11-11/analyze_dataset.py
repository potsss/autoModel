
import pandas as pd
from pathlib import Path

def analyze_sub_table_distribution(data_folder: Path):
    """
    Analyzes the distribution of records per customer (SK_ID_CURR) in the sub-tables.
    """
    print("--- Analyzing Sub-Table Record Distribution ---")

    # --- Analyze bureau.csv ---
    bureau_path = data_folder / "bureau.csv"
    if bureau_path.exists():
        try:
            print(f"\n[Analysis] Loading {bureau_path.name}...")
            bureau_df = pd.read_csv(bureau_path)
            
            if "SK_ID_CURR" in bureau_df.columns:
                bureau_counts = bureau_df.groupby("SK_ID_CURR").size()
                
                print("\n--- Distribution of records per customer in bureau.csv ---")
                print(bureau_counts.describe(percentiles=[.25, .5, .75, .9, .99]))
                
                # Check how many customers from bureau are in the main table sample
                app_train_path = data_folder / "application_train.csv"
                if app_train_path.exists():
                    app_train_df = pd.read_csv(app_train_path, usecols=["SK_ID_CURR"])
                    
                    bureau_customers = set(bureau_df["SK_ID_CURR"].unique())
                    train_customers = set(app_train_df["SK_ID_CURR"].unique())
                    
                    overlap = len(train_customers.intersection(bureau_customers))
                    print(f"\nCoverage in application_train (sample of 50k):")
                    print(f"  - Customers in bureau.csv: {len(bureau_customers)}")
                    print(f"  - Customers in application_train.csv: {len(train_customers)}")
                    print(f"  - Customers present in BOTH tables: {overlap} ({overlap / len(train_customers) * 100:.2f}%)")

            else:
                print("[Analysis] 'SK_ID_CURR' not found in bureau.csv")

        except Exception as e:
            print(f"[Analysis] Error processing bureau.csv: {e}")
    else:
        print(f"[Analysis] {bureau_path.name} not found.")


    # --- Analyze installments_payments.csv ---
    installments_path = data_folder / "installments_payments.csv"
    if installments_path.exists():
        try:
            print(f"\n[Analysis] Loading {installments_path.name}...")
            installments_df = pd.read_csv(installments_path)

            if "SK_ID_CURR" in installments_df.columns:
                installments_counts = installments_df.groupby("SK_ID_CURR").size()

                print("\n--- Distribution of records per customer in installments_payments.csv ---")
                print(installments_counts.describe(percentiles=[.25, .5, .75, .9, .99]))

            else:
                print("[Analysis] 'SK_ID_CURR' not found in installments_payments.csv")

        except Exception as e:
            print(f"[Analysis] Error processing installments_payments.csv: {e}")
    else:
        print(f"[Analysis] {installments_path.name} not found.")


if __name__ == "__main__":
    data_folder_path = Path(__file__).parent / "dateset_test"
    analyze_sub_table_distribution(data_folder_path)
