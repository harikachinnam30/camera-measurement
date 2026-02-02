import argparse
import glob
import os
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--rows", type=int, required=True)
ap.add_argument("--cols", type=int, required=True)
ap.add_argument("--square_mm", type=float, required=True)
args = ap.parse_args()

pattern_size = (args.cols, args.rows)

objp = np.zeros((args.rows * args.cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
objp *= args.square_mm

objpoints = []
imgpoints = []

images = glob.glob(os.path.join(args.images, "*.*"))
image_size = None

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = (gray.shape[1], gray.shape[0])

    ret, corners = cv2.findChessboardCorners(gray, pattern_size)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

print("RMS error:", rms)
print("Camera Matrix K:\n", K)
print("Distortion:\n", dist)

os.makedirs("outputs", exist_ok=True)
np.savez("outputs/camera_params.npz", K=K, dist=dist, rms=rms)
print("Saved to outputs/camera_params.npz")
