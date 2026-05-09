"""
One-shot camera probe: runs for 5 steps, prints pixel stats for rows
in the bottom half of the frame (where the road should be), then exits.
Tells us the actual channel order and HSV values coming from the camera.
"""
import sys
sys.path.insert(0, '/Applications/Webots.app/Contents/lib/controller/python')

import os
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
        continue  # skip first couple frames for camera warmup

    raw = camera.getImage()
    img = np.frombuffer(raw, np.uint8).reshape((camera.getHeight(), camera.getWidth(), 4))
    # img shape: (H, W, 4) — channels are in order they arrive from Webots

    # Sample the bottom-center strip where the road / yellow line should be
    h, w = img.shape[:2]
    roi = img[h//2:, w//4: 3*w//4, :]   # bottom half, center columns

    # Stats for each channel
    for ch, name in enumerate(['Ch0', 'Ch1', 'Ch2', 'Ch3(alpha)']):
        vals = roi[:, :, ch].flatten()
        print(f"  {name}: mean={vals.mean():.1f} max={vals.max()} min={vals.min()}")

    # Try both BGR and RGB interpretations → HSV → count pixels in yellow range
    bgr3 = img[:, :, :3]          # assume BGRA → discard alpha → BGR
    rgb3 = img[:, :, [2, 1, 0]]   # swap Ch0/Ch2 → if RGBA, this becomes BGR

    def count_yellow(bgr_img, lo, hi, label):
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        n = np.count_nonzero(mask)
        # Sample a few pixel HSV values from bottom-center
        roi_hsv = hsv[h//2:, w//4: 3*w//4]
        sample = roi_hsv.reshape(-1, 3)[:10]
        print(f"  [{label}] yellow pixels (H15-40,S80,V80): {n}  |  "
              f"sample HSV (H,S,V): {sample[0]} {sample[1]} {sample[2]}")

    print(f"\n--- Step {step} ---")
    count_yellow(bgr3,  [15, 80, 80], [40, 255, 255], "BGRA→BGR (current)")
    count_yellow(rgb3,  [15, 80, 80], [40, 255, 255], "swap R↔B (if RGBA)")

    # Also try a wide yellow range to catch any yellow regardless of order
    count_yellow(bgr3,  [10, 30, 100], [50, 255, 255], "BGRA→BGR wide")
    count_yellow(rgb3,  [10, 30, 100], [50, 255, 255], "swap R↔B wide")

    if step >= 5:
        print("\nProbe done.")
        break
