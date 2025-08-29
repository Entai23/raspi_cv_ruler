# -*- coding: gbk -*-
# 适用于斜45度放置的正方形测量（以对角线为中轴）
import cv2
import numpy as np
import time
import math
import subprocess

# === 新增：测距功能参数 ===
# 经验数据：(像素高度, 实际距离cm)，可根据实际情况扩充
DISTANCE_CALIBRATION = [
    (761.0, 200), (781.5, 195), (801.0, 190), (822.0, 185), (844.0, 180),
    (868.0, 175), (893.0, 170), (919.0, 165), (946.5, 160),
    (975.0, 155), (1006.5, 150), (1042.0, 145), (1075.5, 140),
    (1114.5, 135), (1156.5, 130), (1200.5, 125), (1249.0, 120),
    (1301.0, 115), (1357.0, 110), (1419.5, 105), (1488.5, 100)
]
DISTANCE_CALIBRATION.sort(key=lambda x: x[0])
CALIB_PIXELS    = np.array([p for p, _ in DISTANCE_CALIBRATION], dtype=np.float32)
CALIB_DISTANCES = np.array([d for _, d in DISTANCE_CALIBRATION], dtype=np.float32)


ENABLE_CONTINUOUS_VIDEO = 0
ENABLE_SINGLE_FRAME = 1
ENABLE_LOCAL_IMAGE = 0

# === 各距离段的补偿系数表（单位：cm）===
# 新增：针对斜45度正方形的补偿系数
COMPENSATION_TABLE = {
    (100, 120): {'normal': 0.994, 'slight': 1.000, 'moderate': 1.007, 'large': 1.005},
    (120, 140): {'normal': 0.995, 'slight': 1.001, 'moderate': 1.002, 'large': 1.003},
    (140, 160): {'normal': 0.995, 'slight': 1.002, 'moderate': 1.004, 'large': 1.005},
    (160, 180): {'normal': 0.995, 'slight': 1.002, 'moderate': 1.007, 'large': 1.000},
    (180, 200): {'normal': 0.997, 'slight': 1.001, 'moderate': 1.004, 'large': 1.006},
    (200, 300): {'normal': 0.995, 'slight': 1.00, 'moderate': 1.004, 'large': 1.005},
}

# 分段阈值（像素差比例）- 优化单帧判断灵敏度
NORMAL_THRESHOLD = 0.02    # 3% 以内为正视
SLIGHT_THRESHOLD = 0.07    # 3-10% 为小幅旋转
MODERATE_THRESHOLD = 0.9  # 10-20% 为中幅旋转
# >20% 为大幅旋转

# 其他参数
OUTLIER_THRESHOLD = 0.2  # 异常值过滤阈值
HISTORY_LEN = 1  # 单帧模式下仅保留当前帧数据


def get_contour_center(contour):
    """计算轮廓中心点"""
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return (0, 0)
    return (M['m10'] / M['m00'], M['m01'] / M['m00'])


def distance(p1, p2):
    """计算两点间距离"""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_filtered_average(values):
    """计算过滤异常值后的平均值"""
    if len(values) <= 2:
        return np.mean(values), values

    mean_val = np.mean(values)
    filtered = [v for v in values if abs(v - mean_val) <= OUTLIER_THRESHOLD * mean_val]

    if len(filtered) < 2:
        return mean_val, values

    return np.mean(filtered), filtered


# === 新增：测距核心函数 ===
def estimate_distance(pixel_height):
    """根据外框高度插值估算距离（cm）"""
    if pixel_height <= CALIB_PIXELS[0]:
        k = (CALIB_DISTANCES[1] - CALIB_DISTANCES[0]) / (CALIB_PIXELS[1] - CALIB_PIXELS[0])
        return CALIB_DISTANCES[0] + k * (pixel_height - CALIB_PIXELS[0])
    elif pixel_height >= CALIB_PIXELS[-1]:
        k = (CALIB_DISTANCES[-1] - CALIB_DISTANCES[-2]) / (CALIB_PIXELS[-1] - CALIB_PIXELS[-2])
        return CALIB_DISTANCES[-1] + k * (pixel_height - CALIB_PIXELS[-1])
    else:
        return float(np.interp(pixel_height, CALIB_PIXELS, CALIB_DISTANCES))


def get_compensation(distance_cm):
    """根据测得距离选择对应段的补偿系数"""
    for (d_min, d_max), tbl in COMPENSATION_TABLE.items():
        if d_min <= distance_cm < d_max:
            return tbl
    # 超出范围时用最近段
    return COMPENSATION_TABLE[max(COMPENSATION_TABLE.keys(), key=lambda r: r[0])]


def get_outer_frame_params(approx_outer):
    """获取外框关键参数：严格计算上下短边中点距离作为长边中轴"""
    pts = approx_outer.reshape(4, 2).astype(np.float32)
    
    # 1. 严格区分上下短边（水平方向）
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_short_edge = y_sorted[:2]  # 上边（短边）两点
    bottom_short_edge = y_sorted[2:]  # 下边（短边）两点
    
    # 2. 计算上下短边各自的中点
    top_mid = np.mean(top_short_edge, axis=0)  # 上边中点
    bottom_mid = np.mean(bottom_short_edge, axis=0)  # 下边中点
    
    # 3. 计算长边中轴：上下短边中点的几何距离（严格定义）
    vertical_axis_pixels = distance(top_mid, bottom_mid)
    
    # 4. 计算上下短边的长度
    top_width = distance(top_short_edge[0], top_short_edge[1])
    bottom_width = distance(bottom_short_edge[0], bottom_short_edge[1])
    
    # 5. 计算左右长边的长度（用于旋转判断）
    x_sorted = pts[np.argsort(pts[:, 0])]
    left_long_edge = x_sorted[:2]  # 左边（长边）两点
    right_long_edge = x_sorted[2:]  # 右边（长边）两点
    left_height = distance(left_long_edge[0], left_long_edge[1])
    right_height = distance(right_long_edge[0], right_long_edge[1])
    
    # 返回：上下短边信息、左右长边信息、严格计算的长边中轴
    return (top_width, bottom_width, top_mid, bottom_mid), \
           (left_height, right_height), \
           vertical_axis_pixels


def get_inner_square_diagonal(approx_inner):
    """获取内部正方形的对角线长度（最远两点距离）"""
    pts = approx_inner.reshape(-1, 2).astype(np.float32)
    max_dist = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dist = distance(pts[i], pts[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist


def process_frame(frame):
    """处理单帧图像的核心函数"""
    global current_mode  # 仅保留当前模式变量
    NORMAL_COMPENSATION = SLIGHT_ROTATION_COMPENSATION = MODERATE_ROTATION_COMPENSATION = LARGE_ROTATION_COMPENSATION = 1.0

    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # 外框检测
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    a4_outer = contours[0]
    cv2.drawContours(img, [a4_outer], -1, (0, 255, 0), 3)

    # 外框参数计算
    peri_outer = cv2.arcLength(a4_outer, True)
    approx_outer = cv2.approxPolyDP(a4_outer, 0.02 * peri_outer, True)
    rect = cv2.minAreaRect(a4_outer)
    rect_w, rect_h = rect[1]
    
    # 初始化变量
    ppcm = None
    vertical_axis_pix = 0.0  # 长边中轴像素值
    top_mid = None
    bottom_mid = None
    current_left = 0.0  # 当前帧左边竖边长度
    current_right = 0.0  # 当前帧右边竖边长度

    outer_height = 0.0
    if len(approx_outer) == 4:
        # 获取外框参数：严格计算长边中轴
        (top_width, bottom_width, top_mid, bottom_mid), height_pairs, vertical_axis_pix = get_outer_frame_params(approx_outer)
        
        # 单帧模式：仅保留当前帧的左右竖边数据
        current_left, current_right = height_pairs
        
        # 确定当前外框高度（使用严格计算的长边中轴）
        outer_height = vertical_axis_pix
        
        # 计算ppcm（单帧模式直接使用当前中轴）
        ppcm = vertical_axis_pix / 29.7  # A4长边实际为29.7cm

        # 可视化中轴（调试用）
        if top_mid is not None and bottom_mid is not None:
            cv2.line(img, (int(top_mid[0]), int(top_mid[1])), 
                    (int(bottom_mid[0]), int(bottom_mid[1])), (255, 0, 0), 2)
            cv2.circle(img, (int(top_mid[0]), int(top_mid[1])), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(bottom_mid[0]), int(bottom_mid[1])), 5, (0, 0, 255), -1)

    # 估算距离并显示
    distance_cm = 0.0
    if outer_height > 0:
        distance_cm = estimate_distance(outer_height)
        # 选取本次用的补偿系数
        comp = get_compensation(distance_cm)
        NORMAL_COMPENSATION = comp['normal']
        SLIGHT_ROTATION_COMPENSATION = comp['slight']
        MODERATE_ROTATION_COMPENSATION = comp['moderate']
        LARGE_ROTATION_COMPENSATION = comp['large']

        # 在图上显示距离
        cv2.putText(frame, f"Dist: {distance_cm:.1f}cm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        # 打印距离和ppcm信息
        print(f"外框长边中轴: {outer_height:.1f}像素 -> 距离估算: {distance_cm:.1f}cm")
        if ppcm:
            print(f"基于中轴的ppcm: {ppcm:.2f} 像素/厘米")

    # 如果ppcm仍未计算成功，使用备选方案
    if ppcm is None:
        if rect_h > rect_w:
            ppcm = rect_h / 29.7
        else:
            ppcm = rect_w / 29.7
        print(f"使用备选ppcm: {ppcm:.2f} 像素/厘米")

    # 内框检测
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [a4_outer], -1, 255, -1)
    inner_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    contours_inner, _ = cv2.findContours(inner_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    a4_inner = None
    for cnt in contours_inner:
        area = cv2.contourArea(cnt)
        if area < 0.5 * cv2.contourArea(a4_outer) or area > 0.95 * cv2.contourArea(a4_outer):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            a4_inner = cnt
            break
    if a4_inner is not None:
        cv2.drawContours(img, [a4_inner], -1, (0, 0, 255), 2)

        # 内部物体检测
        mask_inner = np.zeros_like(gray)
        cv2.drawContours(mask_inner, [a4_inner], -1, 255, -1)
        shrink = cv2.erode(mask_inner, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=1)
        it2 = cv2.bitwise_and(thresh, thresh, mask=shrink)
        ci2, _ = cv2.findContours(it2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in ci2:
            if cv2.contourArea(c) < 100:
                continue
            center_pt = get_contour_center(c)
            if cv2.pointPolygonTest(a4_inner, center_pt, False) != 1:
                continue
            candidates.append(c)
        if candidates:
            cnt = max(candidates, key=cv2.contourArea)
            cv2.drawContours(img, [cnt], -1, (128, 0, 0), 2)
            center = get_contour_center(cnt)

            # 核心测量逻辑
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # 正方形处理
            if len(approx) == 4:
                # 单帧模式：直接基于当前帧计算旋转差异
                rotation_level = "正视"
                compensation = NORMAL_COMPENSATION

                # 计算当前帧的左右竖边差异比例（单帧模式无需历史数据）
                pixel_diff_ratio = abs(current_left - current_right) / max(current_left, current_right, 1e-6)

                # 单帧模式：无迟滞直接判断
                if pixel_diff_ratio > MODERATE_THRESHOLD:
                    current_mode = "large"
                elif pixel_diff_ratio > SLIGHT_THRESHOLD:
                    current_mode = "moderate"
                elif pixel_diff_ratio > NORMAL_THRESHOLD:
                    current_mode = "slight"
                else:
                    current_mode = "normal"

                # 设置对应的补偿系数和旋转等级
                if current_mode == "normal":
                    rotation_level = "正视"
                    compensation = NORMAL_COMPENSATION
                elif current_mode == "slight":
                    rotation_level = "小幅旋转"
                    compensation = SLIGHT_ROTATION_COMPENSATION
                elif current_mode == "moderate":
                    rotation_level = "中幅旋转"
                    compensation = MODERATE_ROTATION_COMPENSATION
                else:
                    rotation_level = "大幅旋转"
                    compensation = LARGE_ROTATION_COMPENSATION

                # 正视模式（直接计算对角线）
                if current_mode == "normal":
                    # 获取正方形对角线长度（像素）
                    diagonal_pixels = get_inner_square_diagonal(approx)
                    # 转换为实际长度（cm）
                    diagonal_cm = diagonal_pixels / ppcm * compensation
                    # 计算边长：对角线 / √2
                    side_length = diagonal_cm / math.sqrt(2)

                    print(f"正方形（{rotation_level}）: 边长 = {side_length:.2f}cm (对角线 = {diagonal_cm:.2f}cm)")
                    org = (int(center[0]), int(center[1]))
                    cv2.putText(img, f"S={side_length:.2f}cm", org,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # 旋转模式（中轴比例法）
                else:
                    # 内部正方形对角线像素数
                    inner_diagonal_pix = get_inner_square_diagonal(approx)
                    # 外框长边中轴像素数（单帧模式直接使用当前值）
                    outer_axis = vertical_axis_pix
                    # 像素比例计算实际对角线长度
                    axis_ratio = inner_diagonal_pix / outer_axis
                    actual_diagonal = 29.7 * axis_ratio * compensation  # 使用A4长边29.7cm作为基准
                    # 计算边长：对角线 / √2
                    side_length = actual_diagonal / math.sqrt(2)

                    print(f"正方形（{rotation_level}）: 边长 = {side_length:.2f}cm (对角线 = {actual_diagonal:.2f}cm)")
                    org = (int(center[0]), int(center[1]))
                    cv2.putText(img, f"S={side_length:.2f}cm", org,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            else:
                print("未检测到正方形")

    return img


if __name__ == "__main__":
    # 初始化模式
    current_mode = "normal"

    # 单帧拍摄模式
    if ENABLE_SINGLE_FRAME:
        print("单帧拍摄模式 (libcamera-still) - 拍摄中...")
        cmd = [
            "rpicam-still",
            "--nopreview",
            "-t", "10",
            "-o", "image.jpg",
            "--width", "2028",
            "--height", "1520",
            "--awbgains", "3.47,1.55",
            "--gain", "40",
            "--shutter", "30000",
        ]
        subprocess.run(cmd, check=True)

        img = cv2.imread("image.jpg")
        if img is None:
            raise RuntimeError("读取 image.jpg 失败")

        # 顺时针旋转 → 处理 → 逆时针旋转 → 压缩
        rot      = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        processed= process_frame(rot)
        restored = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        output   = cv2.resize(restored, (640, 480))

        cv2.imshow("Result", output)
        print("处理完成 —— 5 秒后自动退出，或按 ESC 退出")

        key = cv2.waitKey(5000)  
        
        if key & 0xFF == 27:
            print("检测到 ESC，提前退出")
        else:
            print("5 秒到，自动退出")
        
        cv2.destroyAllWindows()