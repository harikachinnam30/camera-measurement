import argparse
import os
import cv2
import numpy as np
import csv

CLICK_POINTS = []

def click_callback(event, x, y, flags, param):
    global CLICK_POINTS
    if event == cv2.EVENT_LBUTTONDOWN:
        CLICK_POINTS.append((x, y))
        print(f"Point {len(CLICK_POINTS)}: ({x}, {y})")

def warp_point(Hinv, pt):
    x, y = pt
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = Hinv @ p
    q /= q[2]
    return (float(q[0]), float(q[1]))

def dist2(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ref_w_mm", type=float, required=True)
    ap.add_argument("--ref_h_mm", type=float, required=True)
    args = ap.parse_args()

    # Load camera calibration output (Step 1)
    params = np.load("outputs/camera_params.npz")
    K = params["K"]
    dist = params["dist"]

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit("Could not read the image path.")

    h, w = img.shape[:2]

    # Undistort image
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undist = cv2.undistort(img, K, dist, None, newK)

    global CLICK_POINTS
    CLICK_POINTS = []

    cv2.namedWindow("Undistorted - Click 8 points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Undistorted - Click 8 points", click_callback)

    print("Click 4 corners of REFERENCE rectangle in order: TL, TR, BR, BL")
    print("Then click 4 corners of OBJECT rectangle in order: TL, TR, BR, BL")
    print("Press 'q' after 8 clicks.")

    while True:
        vis = undist.copy()
        for i, (x, y) in enumerate(CLICK_POINTS):
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(vis, str(i+1), (x+8, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow("Undistorted - Click 8 points", vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(CLICK_POINTS) != 8:
        raise SystemExit(f"Expected 8 points, got {len(CLICK_POINTS)}. Click exactly 8 points then press q.")

    ref_img = np.array(CLICK_POINTS[:4], dtype=np.float32)
    obj_img = np.array(CLICK_POINTS[4:], dtype=np.float32)

    # Reference rectangle coordinates in mm (world plane)
    ref_world = np.array([
        [0, 0],
        [args.ref_w_mm, 0],
        [args.ref_w_mm, args.ref_h_mm],
        [0, args.ref_h_mm]
    ], dtype=np.float32)

    # Homography WORLD -> IMAGE, invert to map IMAGE -> WORLD
    H, _ = cv2.findHomography(ref_world, ref_img, method=0)
    Hinv = np.linalg.inv(H)

    obj_world = [warp_point(Hinv, p) for p in obj_img]

    w_top = dist2(obj_world[0], obj_world[1])
    w_bot = dist2(obj_world[3], obj_world[2])
    obj_w = 0.5 * (w_top + w_bot)

    h_left = dist2(obj_world[0], obj_world[3])
    h_right = dist2(obj_world[1], obj_world[2])
    obj_h = 0.5 * (h_left + h_right)

    print("\n=== Object Dimensions (mm) ===")
    print(f"Width  = {obj_w:.2f} mm")
    print(f"Height = {obj_h:.2f} mm")

    os.makedirs("outputs", exist_ok=True)
    out_csv = "outputs/results.csv"
    write_header = not os.path.exists(out_csv)

    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["image", "ref_w_mm", "ref_h_mm", "obj_w_mm", "obj_h_mm"])
        writer.writerow([args.image, args.ref_w_mm, args.ref_h_mm, f"{obj_w:.2f}", f"{obj_h:.2f}"])

    print(f"Saved results to {out_csv}")

if __name__ == "__main__":
    main()
