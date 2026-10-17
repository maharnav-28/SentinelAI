"""Download and verify official UCI AI4I 2020 Predictive Maintenance dataset."""

import sys
from pathlib import Path

# Add project root to sys.path so script can be run from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import download_uci_dataset, load_raw_data, get_dataset_summary


def main():
    print("=== SentinelAI: Downloading UCI AI4I 2020 Dataset ===")
    try:
        csv_path = download_uci_dataset()
        df = load_raw_data(csv_path)
        summary = get_dataset_summary(df)

        print("\n--- Dataset Summary ---")
        print(f"File location:    {csv_path.resolve()}")
        print(f"Total records:    {summary['total_samples']:,}")
        print(f"Total columns:    {summary['features_count']}")
        print(f"Normal samples:   {summary['normal']:,}")
        print(f"Failure samples:  {summary['failures']:,}")
        print(f"Failure rate:     {summary['failure_rate_pct']}%")
        print("\nDataset successfully acquired and verified!")

    except Exception as exc:
        print(f"\n[ERROR] Failed to obtain or verify dataset: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
