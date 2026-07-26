import pandas as pd
from pathlib import Path

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

output_dir = Path('/home/nicol/TSB-AD/results')

# --- Load every per-method results file ---
all_files = sorted(output_dir.glob('results_*.csv'))
if not all_files:
    raise FileNotFoundError(f"No 'results_*.csv' files found in {output_dir}")

print(f"Found {len(all_files)} method result files:")
for f in all_files:
    print(f"  - {f.name}")

combined = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)

# --- Build mean + std pivot tables for each metric ---
metrics_to_report = {
    "raw_pr_auc_wtd": "AUC-PR WTD",
    "vus_pr": "VUS-PR",
}

summary_path = output_dir / 'final_summary.txt'

with open(summary_path, 'w') as out:
    for col, label in metrics_to_report.items():
        header = f"\n######################## {label} ########################\n"
        print(header)
        out.write(header)

        pivot_mean = combined.pivot_table(values=[col], columns=['scenario'], index=['method'], aggfunc='mean')
        pivot_mean['Avg'] = pivot_mean.mean(axis=1)
        pivot_mean = pivot_mean.round(3).sort_values(by='Avg', ascending=False)
        print("Mean:\n", pivot_mean)
        out.write("Mean:\n" + pivot_mean.to_string() + "\n")

        pivot_std = combined.pivot_table(values=[col], columns=['scenario'], index=['method'], aggfunc='std')
        pivot_std['Avg'] = pivot_std.mean(axis=1)
        pivot_std = pivot_std.sort_values(by='Avg', ascending=False)
        print("\nStd:\n", pivot_std)
        out.write("\nStd:\n" + pivot_std.to_string() + "\n")

print(f"\nFinal combined summary saved to: {summary_path}")