import pandas as pd
import re
from pathlib import Path

for s in range(1, 11):
    ds = f"S{s}"
    bbox_file = f"data/{ds}_bounding_boxes.csv"
    if not Path(bbox_file).exists(): continue
    df = pd.read_csv(bbox_file)
    ids = df['ID'].astype(str).tolist()
    tiffs = [f.name for f in Path(f"data/{ds}").glob("*.tiff")]
    tiff_set = set(tiffs)
    missing = []
    
    for img_id in ids:
        digits = "".join(re.findall(r'\d+', img_id))
        expected_tiff = f"R{digits}.tiff"
        if expected_tiff not in tiff_set:
            missing.append(img_id)
            
    if missing:
        print(f"{ds} missing {len(missing)} out of {len(ids)}. Examples: {missing[:5]}")
