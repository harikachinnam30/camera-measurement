import glob, os
import cv2

folder = r"data\calibration_images"
paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
print("Images found:", len(paths))
if not paths:
    raise SystemExit("No JPG images found in data/calibration_images")

# Limit search to common sizes to make it fast
candidates = []
for rows in range(4, 12):      # 4..11
    for cols in range(4, 15):  # 4..14
        candidates.append((cols, rows))

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

def detect(gray, size):
    cols, rows = size
    # Try the normal detector
    found, _ = cv2.findChessboardCorners(gray, (cols, rows), flags=flags)
    if found:
        return True
    # Try the more robust SB detector (if available)
    if hasattr(cv2, "findChessboardCornersSB"):
        found_sb, _ = cv2.findChessboardCornersSB(gray, (cols, rows))
        return bool(found_sb)
    return False

hits_table = []
for (cols, rows) in candidates:
    hits = 0
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if detect(gray, (cols, rows)):
            hits += 1
    hits_table.append((hits, cols, rows))

hits_table.sort(reverse=True)

print("\nTop candidates (hits, cols, rows):")
for h, c, r in hits_table[:10]:
    print(h, c, r)

best = hits_table[0]
print("\nBEST:", best)
if best[0] == 0:
    print("\nNo chessboard detected for any tested size.")
    print("This usually means: wrong pattern, board is cropped, too blurry, glare, too small, or not a true chessboard.")
