# -*- coding: gbk -*-
import cv2
import numpy as np
import time
import subprocess

# === 参数配置 ===
HISTORY_LEN = 3  # 历史平滑长度
outer_vertical_pixels = []
outer_axis_pixels = []
current_mode = "normal"

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_outer_frame_params(approx_outer):
    pts = approx_outer.reshape(4, 2).astype(np.float32)
    x_sorted = pts[np.argsort(pts[:, 0])]
    left_pts, right_pts = x_sorted[:2], x_sorted[2:]
    left_pixels = distance(*left_pts)
    right_pixels = distance(*right_pts)
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_pts, bottom_pts = y_sorted[:2], y_sorted[2:]
    top_mid = ((top_pts[0][0]+top_pts[1][0])/2, (top_pts[0][1]+top_pts[1][1])/2)
    bottom_mid = ((bottom_pts[0][0]+bottom_pts[1][0])/2, (bottom_pts[0][1]+bottom_pts[1][1])/2)
    axis_pixels = distance(top_mid, bottom_mid)
    return left_pixels, right_pixels, axis_pixels

def process_frame(frame):
    """全分辨率处理，只做轮廓检测和像素计算，不缩放"""
    global outer_vertical_pixels, outer_axis_pixels, current_mode

    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        cv2.putText(img, "未检测到外框", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        return img

    a4 = max(contours, key=cv2.contourArea)
    cv2.drawContours(img, [a4], -1, (0,255,0), 3)
    peri = cv2.arcLength(a4, True)
    approx = cv2.approxPolyDP(a4, 0.02*peri, True)
    if len(approx) != 4:
        cv2.putText(img, "外框非四边形", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        return img

    left_pix, right_pix, axis_pix = get_outer_frame_params(approx)
    outer_vertical_pixels.append((left_pix, right_pix))
    outer_axis_pixels.append(axis_pix)
    if len(outer_vertical_pixels) > HISTORY_LEN:
        outer_vertical_pixels.pop(0)
        outer_axis_pixels.pop(0)

    avg_left = np.mean([p[0] for p in outer_vertical_pixels])
    avg_right = np.mean([p[1] for p in outer_vertical_pixels])
    avg_axis = np.mean(outer_axis_pixels)
    diff_ratio = abs(avg_left - avg_right) / max(avg_left, avg_right, 1e-6)

    # 模式判断
    if diff_ratio <= 0.02:
        current_mode = "normal"; mode_text = "正视模式"; effective = (avg_left+avg_right)/2
    else:
        if diff_ratio <= 0.15: current_mode, mode_text = "slight", "小幅斜视"
        elif diff_ratio <= 0.22: current_mode, mode_text = "moderate", "中幅斜视"
        else: current_mode, mode_text = "large", "大幅斜视"
        effective = avg_axis

    # 标注
    cv2.putText(img, f"模式: {mode_text}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(img, f"有效像素: {effective:.1f}", (20,80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
    print(f"[{time.strftime('%H:%M:%S')}] 模式: {mode_text}, 有效像素: {effective:.1f}")
    return img

def capture_image():
    """
    用 libcamera-still 拍摄一帧全分辨率 JPEG，保存到 image.jpg。
    """
    cmd = [
    "rpicam-still",
    "-o", "image.jpg",
    "--width", "2028",    # 改为 2028 像素宽
    "--height", "1520",   # 改为 1520 像素高
    "--nopreview",
    # 如果要原始 DNG 一起输出，可取消下一行注释：
    # "--raw",
]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    # 1. 拍照
    capture_image()
    # 2. 读取
    img = cv2.imread("image.jpg")
    if img is None:
        raise RuntimeError("读取 image.jpg 失败")

    # 3. 顺时针旋转 90° → (H, W) 互换
    rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    # 4. 全分辨率处理
    processed = process_frame(rot)
    # 5. 逆时针旋转 90° 还原方向
    restored = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # 6. 最终压缩到 640×480 展示
    output = cv2.resize(restored, (640, 480))

    cv2.imshow("单帧外框测量", output)
    while cv2.waitKey(1) & 0xFF != 27:
        pass
    cv2.destroyAllWindows()
