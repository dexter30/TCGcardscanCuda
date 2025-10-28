import cv2
import numpy as np
import time
import os
import json

# === CONFIGURATION ===
CARDS_FOLDER = "cards/"
CACHE_FILE = "card_cache.npz"
MANIFEST_FILE = "card_manifest.json"
MIN_GOOD_MATCHES = 0
MIN_INLIERS = 15
MATCH_COOLDOWN = 3
CHECK_EVERY_N_FRAMES = 10
# =======================

# --- Detect CUDA availability ---
use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
if use_cuda:
    print(f"✅ CUDA detected: {cv2.cuda.getCudaEnabledDeviceCount()} GPU(s) available.")
    orb = cv2.cuda_ORB.create(nfeatures=1000)
    matcher = cv2.cuda_DescriptorMatcher.createBFMatcher(cv2.NORM_HAMMING)
else:
    print("⚠️ CUDA not found — using CPU mode.")
    orb = cv2.ORB_create(nfeatures=1000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ---------- Cache logic ----------
def build_card_cache():
    cards = {}
    all_descriptors = []
    card_ids = []
    print(f"Building card cache from '{CARDS_FOLDER}'...")
    for root, _, files in os.walk(CARDS_FOLDER):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, filename)
                img = cv2.imread(path)
                if img is None:
                    print(f"⚠️ Could not read {path}")
                    continue
                kp, des = cv2.ORB_create(nfeatures=1000).detectAndCompute(img, None)
                if des is None:
                    print(f"⚠️ No features found in {path}")
                    continue
                card_index = len(cards)
                cards[card_index] = (path, kp, des, img)
                all_descriptors.append(des)
                card_ids += [card_index] * len(des)

    np.savez_compressed(CACHE_FILE,
                        descriptors=np.vstack(all_descriptors).astype(np.uint8),
                        card_ids=np.array(card_ids, dtype=np.int32))
    with open(MANIFEST_FILE, "w") as f:
        json.dump({idx: path for idx, (path, _, _, _) in cards.items()}, f)
    print(f"✅ Cache saved to '{CACHE_FILE}' and '{MANIFEST_FILE}'")
    return cards, all_descriptors, card_ids

def load_card_cache():
    print(f"Loading card cache from '{CACHE_FILE}'...")
    data = np.load(CACHE_FILE)
    card_ids = data["card_ids"]
    all_descriptors = [data["descriptors"]]
    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    cards = {}
    for idx, path in manifest.items():
        img = cv2.imread(path)
        if img is None:
            continue
        kp, des = cv2.ORB_create(nfeatures=1000).detectAndCompute(img, None)
        cards[int(idx)] = (path, kp, des, img)
    print(f"✅ Loaded {len(cards)} cards from cache.\n")
    return cards, all_descriptors, card_ids

# --- Load or build cache ---
if os.path.exists(CACHE_FILE) and os.path.exists(MANIFEST_FILE):
    cards, all_descriptors, card_ids = load_card_cache()
else:
    cards, all_descriptors, card_ids = build_card_cache()

all_descriptors = np.vstack(all_descriptors).astype(np.uint8)
card_ids = np.array(card_ids)
print(f"Matcher index built: {len(all_descriptors)} total descriptors.\n")

# --- Pre-upload descriptor DB to GPU ---
if use_cuda:
    print("Uploading descriptor database to GPU (this may take a few seconds)...")
    gpu_db = cv2.cuda_GpuMat()
    gpu_db.upload(all_descriptors)
    print("✅ Descriptor DB uploaded to GPU memory.\n")
else:
    gpu_db = None

# ---------- Canny tuner ----------
def nothing(x): pass
cv2.namedWindow("Canny Tuner")
cv2.createTrackbar("Low Threshold", "Canny Tuner", 0, 255, nothing)
cv2.createTrackbar("High Threshold", "Canny Tuner", 57, 255, nothing)
cv2.createTrackbar("edge", "Canny Tuner", 6, 10, nothing)
# ---------- Helper functions ----------
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    if maxW < 10 or maxH < 10:
        return None
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (maxW, maxH))
    return warp if warp is not None and warp.size > 0 else None

def preprocess_for_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_16S, 1, 0)
    grad_y = cv2.Sobel(blur, cv2.CV_16S, 0, 1)
    grad = cv2.convertScaleAbs(cv2.addWeighted(cv2.absdiff(grad_x,0),0.5,
                                               cv2.absdiff(grad_y,0),0.5,0))
    return grad

def detect_card(frame):
    grad = preprocess_for_edges(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    low = cv2.getTrackbarPos("Low Threshold", "Canny Tuner")
    high = cv2.getTrackbarPos("High Threshold", "Canny Tuner")
    edgeDetect = (cv2.getTrackbarPos("edge", "Canny Tuner")/100)
    edges = cv2.Canny(blur, low, high)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    cv2.imshow("Edges", edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_contour, max_area = None, 0
    frame_area = frame.shape[0] * frame.shape[1]
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, edgeDetect * peri, True)
        area = cv2.contourArea(c)
        if 4 <= len(approx) <= 6 and (0.01 * frame_area) < area < (0.95 * frame_area):
            if cv2.isContourConvex(approx) and area > max_area:
                best_contour, max_area = approx, area
    return best_contour

# ---------- Main loop ----------
# ---------- Main loop ----------
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

last_match_time = 0
frame_count = 0
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        continue
    frame_count += 1

    contour = detect_card(frame)
    low = cv2.getTrackbarPos("Low Threshold", "Canny Tuner")
    high = cv2.getTrackbarPos("High Threshold", "Canny Tuner")

    if contour is not None:
        cv2.drawContours(frame, [contour], -1, (0,255,0), 2)
        cv2.putText(frame, "Card detected", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # --- Ensure valid contour shape ---
        if len(contour) != 4:
            hull = cv2.convexHull(contour)
            peri = cv2.arcLength(hull, True)
            contour = cv2.approxPolyDP(hull, 0.05 * peri, True)
        if len(contour) != 4:
            x,y,w,h = cv2.boundingRect(contour)
            contour = np.array([[[x,y]],[[x+w,y]],[[x+w,y+h]],[[x,y+h]]], dtype=np.float32)

        warped = four_point_transform(frame, contour.reshape(4,2))
        if warped is None:
            continue

        cv2.imshow("Detected Card Region", warped)

        # --- Run match every N frames ---
        if frame_count % CHECK_EVERY_N_FRAMES == 0 and time.time() - last_match_time > MATCH_COOLDOWN:
            print(f"[+] Frame {frame_count}: Running GPU match...")

            scale = 0.7
            small = cv2.resize(warped, (0,0), fx=scale, fy=scale)
            gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            # Skip invalid frames
            if gray_small is None or gray_small.size == 0:
                print("⚠️ Empty frame — skipping.")
                continue
            h, w = gray_small.shape[:2]
            if h < 10 or w < 10:
                print(f"⚠️ Invalid warped size ({w}x{h}) — skipping.")
                continue

            gray_small = np.ascontiguousarray(gray_small)

            # ---------- CUDA ORB with fallback ----------
            try:
                if use_cuda:
                    gpu_frame = cv2.cuda_GpuMat()
                    print(f"⚠️ cuda_GpuMat 01.")
                    gpu_frame.upload(gray_small)

                    # CUDA ORB compute
                    kp_frame, des_frame = orb.detectAndComputeAsync(gpu_frame, None)
                    if isinstance(des_frame, cv2.cuda_GpuMat):
                        print(f"⚠️ cuda_GpuMat 02.")
                        des_frame = des_frame.download()
                    if not isinstance(kp_frame, list):
                        kp_frame = []
                else:
                    kp_frame, des_frame = orb.detectAndCompute(gray_small, None)

            except cv2.error as e:
                print(f"⚠️ CUDA ORB failed: {e}")
                kp_frame, des_frame = cv2.ORB_create(nfeatures=1000).detectAndCompute(gray_small, None)

            #if des_frame is None or len(kp_frame) < 10:
            #    continue

            # ---------- Matching phase ----------
            try:
                if use_cuda:
                    gpu_query = cv2.cuda_GpuMat()
                    print(f"⚠️ cuda_GpuMat 03.")
                    gpu_query.upload(des_frame)
                    matches = matcher.knnMatch(gpu_query, gpu_db, k=2)
                else:
                    matches = matcher.knnMatch(des_frame, all_descriptors, k=2)

            except cv2.error as e:
                print(f"⚠️ Matcher error: {e}")
                continue

            good_matches, matched_ids = [], []
            for m_n in matches:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
                    matched_ids.append(card_ids[m.trainIdx])

            # ---------- Voting and verification ----------
            print(f"⚠️ good matches: {len(good_matches)}")
            if len(good_matches) > MIN_GOOD_MATCHES:
                unique_ids, counts = np.unique(matched_ids, return_counts=True)
                best_card_idx = unique_ids[np.argmax(counts)]
                best_card_path, kp_db, des_db, img_db = cards[best_card_idx]
                print(f"✅ Likely match: {os.path.basename(best_card_path)} ({counts.max()} votes)")
                                # Comparison window
                h1, w1 = img_db.shape[:2]
                h2, w2 = warped.shape[:2]
                target_h = 300
                scale1 = target_h / h1
                scale2 = target_h / h2
                img_db_resized = cv2.resize(img_db, (int(w1 * scale1), target_h))
                warped_resized = cv2.resize(warped, (int(w2 * scale2), target_h))
                comparison = np.hstack((img_db_resized, warped_resized))
                label = f"{os.path.basename(best_card_path)} | votes:{counts.max()} good matches:{good_matches}"
                cv2.putText(comparison, label, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.imshow("Match Comparison", comparison)


                # Geometric verification
                src_pts, dst_pts = [], []
                for m in good_matches:
                    if card_ids[m.trainIdx] == best_card_idx and m.queryIdx < len(kp_frame):
                        local_idx = m.trainIdx % len(kp_db)
                        if 0 <= local_idx < len(kp_db):
                            src_pts.append(kp_db[local_idx].pt)
                            dst_pts.append(kp_frame[m.queryIdx].pt)

                if len(src_pts) >= 4:
                    src_pts = np.float32(src_pts).reshape(-1,1,2)
                    dst_pts = np.float32(dst_pts).reshape(-1,1,2)
                    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    inliers = int(mask.sum()) if mask is not None else 0

                    if inliers > MIN_INLIERS:
                        last_match_time = time.time()
                        print(f"✅ Confirmed: {os.path.basename(best_card_path)} ({inliers} inliers)")
                        cv2.putText(frame, f"Matched: {os.path.basename(best_card_path)}",
                                    (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)


    # --- Overlay debug info ---
    cv2.putText(frame, f"Canny: {low}/{high}", (20,120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("Card Scanner", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
