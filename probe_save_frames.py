"""
Save the raw BGR frame, yellow mask, and final grayscale to /tmp so we can
see what the pipeline actually processes.  Runs for 5 simulation steps.
"""
import sys, os
sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python')
os.environ['WEBOTS_HOME'] = '/Applications/Webots.app'

import numpy as np
import cv2
from vehicle import Car

robot = Car()
timestep = int(robot.getBasicTimeStep())
camera = robot.getDevice("camera")
camera.enable(timestep)

step = 0
while robot.step() != -1:
    step += 1
    if step < 3:
        continue

    raw = camera.getImage()
    img4 = np.frombuffer(raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    bgr = img4[:, :, :3]          # BGRA → BGR

    # -- Yellow mask pipeline (current code) --
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo, hi = np.array([15, 80, 80]), np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lo, hi)
    masked_bgr = cv2.bitwise_and(bgr, bgr, mask=mask)
    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # ROI trapezoid (same logic as apply_roi)
    h, w = gray.shape
    roi_mask = np.zeros_like(gray)
    poly = np.array([[
        (int(w*0.25), int(h*0.50)),
        (int(w*0.75), int(h*0.50)),
        (w, h), (0, h)
    ]], dtype=np.int32)
    cv2.fillPoly(roi_mask, poly, 255)
    edges_roi = cv2.bitwise_and(edges, roi_mask)

    # Also: Hough on the ROI
    lines = cv2.HoughLinesP(edges_roi, 1, np.pi/180, threshold=30,
                            minLineLength=15, maxLineGap=20)

    # Save images (upscaled 4× for visibility)
    scale = 4
    out_bgr    = cv2.resize(bgr,        (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
    out_mask   = cv2.resize(mask,       (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
    out_gray   = cv2.resize(gray,       (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
    out_edges  = cv2.resize(edges,      (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
    out_eroi   = cv2.resize(edges_roi,  (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(f'/tmp/step{step}_1_bgr.png',       out_bgr)
    cv2.imwrite(f'/tmp/step{step}_2_ymask.png',     out_mask)
    cv2.imwrite(f'/tmp/step{step}_3_gray.png',      out_gray)
    cv2.imwrite(f'/tmp/step{step}_4_edges.png',     out_edges)
    cv2.imwrite(f'/tmp/step{step}_5_edges_roi.png', out_eroi)

    n_lines = len(lines) if lines is not None else 0
    n_yellow = int(np.count_nonzero(mask))
    # HSV values where mask is active
    yellow_hsv = hsv[mask > 0]
    print(f"Step {step}: yellow_px={n_yellow}  lines_after_hough={n_lines}")
    if n_yellow > 0:
        print(f"  Yellow pixels HSV mean: H={yellow_hsv[:,0].mean():.1f} "
              f"S={yellow_hsv[:,1].mean():.1f} V={yellow_hsv[:,2].mean():.1f}")
        print(f"  Yellow pixels HSV range: "
              f"H=[{yellow_hsv[:,0].min()},{yellow_hsv[:,0].max()}] "
              f"S=[{yellow_hsv[:,1].min()},{yellow_hsv[:,1].max()}] "
              f"V=[{yellow_hsv[:,2].min()},{yellow_hsv[:,2].max()}]")
        # Y-coordinate distribution of yellow pixels
        ys, xs = np.where(mask > 0)
        print(f"  Yellow pixel Y range: {ys.min()}-{ys.max()}  "
              f"(ROI starts at y={int(h*0.5)})")
        print(f"  Yellow pixel X range: {xs.min()}-{xs.max()}")
    print(f"  Saved: /tmp/step{step}_*.png")

    if step >= 5:
        print("Done.")
        break
