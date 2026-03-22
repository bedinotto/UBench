import pandas as pd
import glob
import re
from pathlib import Path

for s in range(1, 11):
    ds = f"S{s}"
    bbox_file = f"data/{ds}_bounding_boxes.csv"
    if not Path(bbox_file).exists(): continue
    df = pd.read_csv(bbox_file)
    ids = df['ID'].astype(str).tolist()
    tiffs = [f.name for f in Path(f"data/{ds}").glob("*.tiff")]
    print(f"--- {ds} ---")
    print(f"Sample IDs: {ids[:3]}")
    print(f"Sample Tiffs: {tiffs[:3]}")
