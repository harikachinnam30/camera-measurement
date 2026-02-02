import argparse, glob, os
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--rows", type=int, required=True)
ap.add_argument("--cols", type=int, required=True)
args = ap.parse_args()

pattern_size = (args.cols, args.rows)

paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
print("Found JPG files:", len(paths))

good = 0
for p in paths:
    img = cv2.imread(p)
    if img is None:
        print("[READ FAIL]", os.path.basename(p))
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)

    print(os.path.basename(p), "-> corners found?", found)
    if found:
        good += 1
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern_size, corners, found)
        cv2.imshow("Detected corners (press any key)", vis)
        cv2.waitKey(0)

cv2.destroyAllWindows()
print("Total valid images with corners:", good)
