import pandas as pd
import numpy as np
from pathlib import Path
from TSB_AD.model_wrapper import run_Unsupervise_AD, run_Semisupervise_AD
from TSB_AD.evaluation.metrics import get_metrics

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


Unsupervise_AD_Pool = ['FFT', 'SR', 'NORMA', 'Series2Graph', 'Sub_IForest', 'IForest', 'LOF', 'Sub_LOF', 'POLY', 'MatrixProfile', 'Sub_PCA', 'PCA', 'HBOS',
                        'Sub_HBOS', 'KNN', 'Sub_KNN','KMeansAD', 'KMeansAD_U', 'KShapeAD', 'COPOD', 'CBLOF', 'COF', 'EIF', 'RobustPCA', 'MMPAD', 'Lag_Llama', 'TimesFM', 'Chronos', 'MOMENT_ZS', 'TSPulse_ZS', 'Time_RCD', 'HSF_U', 'HSF_Causal']
Semisupervise_AD_Pool = ['Left_STAMPi', 'SAND', 'MCD', 'Sub_MCD', 'OCSVM', 'Sub_OCSVM', 'AutoEncoder', 'CNN', 'LSTMAD', 'TranAD', 'USAD', 'OmniAnomaly', 'PatchTST',
                        'AnomalyTransformer', 'TimesNet', 'FITS', 'Donut', 'OFA', 'MOMENT_FT', 'M2N2', 'TSPulse_FT', 'xLSTMAD', 'CHARM', 'StreamVAE', 'HSF', 'PaAno_PAI', 'SHADE', 'TimeRCD_MAFT']


def run_Naive_ZScore(data, window=50):
    """Naive rule-based baseline: rolling z-score per channel, max across channels. No learning."""
    df_data = pd.DataFrame(data)
    roll_mean = df_data.rolling(window=window, min_periods=1, center=True).mean()
    roll_std = df_data.rolling(window=window, min_periods=1, center=True).std().replace(0, 1e-8)
    zscores = ((df_data - roll_mean) / roll_std).abs()
    score = zscores.max(axis=1).to_numpy()
    return score.ravel()


def append_result_row(row: dict, path: Path):
    """Append one row to CSV immediately — survives a crash/kill mid-run."""
    df_row = pd.DataFrame([row])
    write_header = not path.exists()
    df_row.to_csv(path, mode='a', header=write_header, index=False)


# --- Config: EDIT THESE FOR EACH RUN ---
AD_NAME = 'PCA'   # <-- Naive_ZScore, KNN, LOF, EIF, CNN, LSTMAD - Others: MMPAD, StreamVAE, PCA
N_ITER = 5

data_direc = Path('../Datasets/raw/ScenariosV4_lite/')
csv_files = sorted(data_direc.glob('*.csv'))   # use '*TA3_..._0.csv' to test on one file only

output_dir = Path('../results')
output_dir.mkdir(parents=True, exist_ok=True)

# One fixed filename per method (no timestamp) so re-runs extend/overwrite predictably
results_csv_path = output_dir / f'results_{AD_NAME}.csv'

# If retrying after a crash/change, delete the old partial file first so rows aren't duplicated:
#   rm /home/nicol/TSB-AD/results/results_CNN.csv
print(f"Running method: {AD_NAME}")
print(f"Saving to: {results_csv_path}")


for file_path in csv_files:
    df = pd.read_csv(file_path).dropna()

    data = df.iloc[:, 1:-1].values.astype(float)
    label_arr = df['ANOMALY?'].astype(int).to_numpy()

    train_ratio = 0.4
    split_idx = int(len(data) * train_ratio)

    data_train = data[:split_idx]
    data_test = data[split_idx:]
    label_test = label_arr[split_idx:]

    scenario_name = file_path.name.split("_")[0]

    for i in range(N_ITER):
        try:
            if AD_NAME == 'Naive_ZScore':
                output = run_Naive_ZScore(data)
                metrics = get_metrics(output, label_arr)

            elif AD_NAME in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(AD_NAME, data)
                metrics = get_metrics(output, label_arr)

            elif AD_NAME in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(AD_NAME, data_train, data_test)
                metrics = get_metrics(output, label_test)

            else:
                raise ValueError(f"'{AD_NAME}' not found in either AD pool")

            raw_pr_auc_wtd = metrics.get("AUC-PR", None)
            vus_pr = metrics.get("VUS-PR", None)

        except Exception as e:
            print(f"Error running {AD_NAME} on {file_path.name} (iter {i}): {e}")
            metrics = {}
            raw_pr_auc_wtd = None
            vus_pr = None

        row = {
            "iteration": i,
            "scenario": scenario_name,
            "method": AD_NAME,
            "file": file_path.name,
            "raw_pr_auc_wtd": raw_pr_auc_wtd,
            "vus_pr": vus_pr,
            **metrics,
        }
        append_result_row(row, results_csv_path)
        print(f"  [{scenario_name}] iter {i}: AUC-PR={raw_pr_auc_wtd}, VUS-PR={vus_pr}")

print(f"\nDone. Results saved to: {results_csv_path}")