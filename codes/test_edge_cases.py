import pandas as pd
from pathlib import Path

# Edge case: S3
tiffs_s3 = [f.name for f in Path("data/S3").glob("*.tiff")]
print("S3 missing examples (are they in dir?):")
print("R351010.tiff in dir?", "R351010.tiff" in tiffs_s3)
print("R351011.tiff in dir?", "R351011.tiff" in tiffs_s3)

# Edge case: S7
tiffs_s7 = [f.name for f in Path("data/S7").glob("*.tiff")]
print("S7 missing examples (are they in dir?):")
print("R71141.tiff in dir?", "R71141.tiff" in tiffs_s7)
print("Related files in S7 to RS71141: ", [t for t in tiffs_s7 if '7114' in t][:5])

