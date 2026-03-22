import pandas as pd
import re
from pathlib import Path

success = 0
failed = 0

for s in range(1, 11):
    ds = f"S{s}"
    bbox_file = f"data/{ds}_bounding_boxes.csv"
    if not Path(bbox_file).exists(): continue
    df = pd.read_csv(bbox_file)
    ids = df['ID'].astype(str).tolist()
    
    tiffs = [f.name for f in Path(f"data/{ds}").glob("*.tiff")]
    tiff_set = set(tiffs)
    
    ds_failed = 0
    
    for img_id in ids:
        # Extract all digits
        digits = "".join(re.findall(r'\d+', img_id))
        expected_tiff = f"R{digits}.tiff"
        if expected_tiff in tiff_set:
            success += 1
        else:
            if ds_failed < 3:
                print(f"Failed {ds}: {img_id} -> {expected_tiff} not found")
            failed += 1
            ds_failed += 1

print(f"Success: {success}, Failed: {failed}")
