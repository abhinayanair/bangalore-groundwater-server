#!/usr/bin/env python3
"""
Merge all Bengaluru groundwater JSON datasets into a single CSV file.
"""
import os
import json
import pandas as pd

DATASET_DIR = "dataset"
OUTPUT_CSV = "bengaluru_groundwater_merged.csv"

def flatten_data_time(data_time):
    """Flatten the nested dataTime dictionary into a flat dict."""
    if not isinstance(data_time, dict):
        return {}
    flat = {}
    for k, v in data_time.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat[f"{k}_{subk}"] = subv
        else:
            flat[k] = v
    return flat

def main():
    print("Merging all groundwater JSON files into one DataFrame...")
    all_records = []
    json_files = sorted([f for f in os.listdir(DATASET_DIR) if f.endswith('.json')])
    for filename in json_files:
        path = os.path.join(DATASET_DIR, filename)
        print(f"Reading {filename}...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for rec in data:
                # Flatten dataTime
                if 'dataTime' in rec:
                    flat_time = flatten_data_time(rec['dataTime'])
                    for k, v in flat_time.items():
                        rec[f"dataTime_{k}"] = v
                    del rec['dataTime']
                all_records.append(rec)
    print(f"Total records: {len(all_records)}")
    df = pd.DataFrame(all_records)
    # Optional: sort by year, month, day if available
    if 'dataTime_year' in df.columns and 'dataTime_monthValue' in df.columns and 'dataTime_dayOfMonth' in df.columns:
        df = df.sort_values(['dataTime_year', 'dataTime_monthValue', 'dataTime_dayOfMonth'])
    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Merged dataset saved to {OUTPUT_CSV} (shape: {df.shape})")

if __name__ == "__main__":
    main() 