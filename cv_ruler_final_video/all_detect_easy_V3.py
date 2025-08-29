# -*- coding: gbk -*-
#去除了角度变换,前三问基础题最终版本,修复了cv2.putText 要求 org int格式报错
import cv2
import numpy as np
import time
import subprocess

# === 新增：测距功能参数 ===
# 经验数据：(像素高度, 实际距离cm)，可根据实际情况扩充

DISTANCE_CALIBRATION = [
    (761.0, 200),(781.5, 195), (801.0, 190), (822.0, 185), (844.0, 180),
    (868.0, 175), (893.0, 170), (919.0, 165), (946.5, 160),
    (975.0, 155), (1006.5, 150), (1042.0, 145), (1075.5, 140),
    (1114.5, 135), (1156.5, 130), (1200.5, 125), (1249.0, 120),
    (1301.0, 115), (1357.0, 110), (1419.5, 105), (1488.5, 100)
]
DISTANCE_CALIBRATION.sort(key=lambda x: x[0])
CALIB_PIXELS    = np.array([p for p, _ in DISTANCE_CALIBRATION], dtype=np.float32)
CALIB_DISTANCES = np.array([d for _, d in DISTANCE_CALIBRATION], dtype=np.float32)


ENABLE_CONTINUOUS_VIDEO = 1
ENABLE_SINGLE_FRAME = 0
ENABLE_LOCAL_IMAGE = 0

# === 各距离段的补偿系数表（单位：cm）===
# 格式：{ (min_dist, max_dist): { 'normal':…, 'normal2':三角形 'slight':…, 'moderate':…, 'large':…, 'circle':… } }
COMPENSATION_TABLE = {
    (100, 120): { 'normal':1.006, 'normal2':1.0035, 'slight':1.005, 'moderate':1.000, 'large':1.000, 'circle':0.999 },
    (120, 140): { 'normal':1.005, 'normal2':1.0035, 'slight':1.004, 'moderate':1.009, 'large':1.009, 'circle':0.999 },
    (140, 160): { 'normal':1.005, 'normal2':1.0035, 'slight':1.003, 'moderate':1.008, 'large':1.008, 'circle':0.999 },
    (160, 180): { 'normal':1.005, 'normal2':1.0034, 'slight':1.002, 'moderate':1.007, 'large':1.007, 'circle':0.999 },
    (180, 200): { 'normal':1.0053, 'normal2':1.004, 'slight':1.001, 'moderate':1.006, 'large':1.006, 'circle':0.999 },
    (200, 300): { 'normal':1.006, 'normal2':1.0045, 'slight':1.000, 'moderate':1.005, 'large':1.005, 'circle':0.999 },
}


# 分段阈值
# 像素差比例
NORMAL_THRESHOLD = 0.00000000000002  # 2% 以内为正对
SLIGHT_THRESHOLD = 0.00015  # 2-5% 为小幅旋转
MODERATE_THRESHOLD = 0.22  # 5-10% 为中幅旋转
# >10% 为大幅旋转

# 其他参数
OUTLIER_THRESHOLD = 0.2  # 异常值过滤阈值
CIRCLE_MEASURE_COUNT = 5  # 圆形测量次数

HISTORY_LEN = 3
# 历史数据长度,搭配calculate_filtered_average使用,在视频模式有用,单帧处理一点用没有

# 历史数据存储
circle_history = []
outer_vertical_pixels = []  # 外框左右边像素数历史
outer_axis_pixels = []  # 外框中轴像素数历史
#外框高度历史（用于测距平滑）
outer_height_history = []


def get_contour_center(contour):
    """计算轮廓中心点"""
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return (0, 0)
    return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))


def distance(p1, p2):
    """计算两点间距离"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

#多帧平滑用的,和HISTORY_LEN = 3搭配使用
def calculate_filtered_average(values):
    """改进的异常值过滤平均值计算
    使用中位数和四分位距法替代均值法，减少极端值对过滤结果的影响
    """
    if len(values) <= 4:
        return np.mean(values), values.copy()
    
    # 转换为numpy数组便于计算
    values_np = np.array(values, dtype=np.float64)
    
    # 使用中位数而非均值作为基准，减少极端值影响
    median_val = np.median(values_np)
    
    # 计算四分位距(IQR)
    q1 = np.percentile(values_np, 25)  # 第一四分位
    q3 = np.percentile(values_np, 75)  # 第三四分位
    iqr = q3 - q1
    
    # 根据IQR确定异常值阈值（更稳健的异常值判断）
    # 1.5是统计学中常用的系数，可根据实际情况调整
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # 过滤异常值
    filtered = values_np[(values_np >= lower_bound) & (values_np <= upper_bound)]
    
    # 确保过滤后有足够的数据
    if len(filtered) < 2:
        # 如果过滤后数据太少，使用放宽的阈值
        lower_bound = q1 - 3.0 * iqr
        upper_bound = q3 + 3.0 * iqr
        filtered = values_np[(values_np >= lower_bound) & (values_np <= upper_bound)]
        
        # 仍然不足则返回原始均值
        if len(filtered) < 2:
            return np.mean(values_np), values.copy()
    
    # 计算过滤后的平均值
    filtered_mean = np.mean(filtered)
    return filtered_mean, filtered.tolist()


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
    """获取外框关键参数：左右边像素数、中轴像素数"""
    pts = approx_outer.reshape(4, 2).astype(np.float32)

    # 按x坐标排序区分左右边
    x_sorted = pts[np.argsort(pts[:, 0])]
    left_pts = x_sorted[:2]  # 左边两点
    right_pts = x_sorted[2:]  # 右边两点

    # 计算左右边像素数（竖直方向长度）
    left_pixels = distance(left_pts[0], left_pts[1])
    right_pixels = distance(right_pts[0], right_pts[1])

    # 按y坐标排序区分上下边
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_pts = y_sorted[:2]  # 上边两点
    bottom_pts = y_sorted[2:]  # 下边两点

    # 计算上下边中点（中轴端点）
    top_mid = ((top_pts[0][0] + top_pts[1][0]) / 2, (top_pts[0][1] + top_pts[1][1]) / 2)
    bottom_mid = ((bottom_pts[0][0] + bottom_pts[1][0]) / 2, (bottom_pts[0][1] + bottom_pts[1][1]) / 2)

    # 计算中轴像素数（上下中点连线）
    axis_pixels = distance(top_mid, bottom_mid)

    return left_pixels, right_pixels, axis_pixels


def get_inner_square_axis(approx_inner):
    """获取内部正方形中轴像素数（上下边中点连线）"""
    pts = approx_inner.reshape(4, 2).astype(np.float32)

    # 按y坐标排序区分上下边
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_pts = y_sorted[:2]  # 上边两点
    bottom_pts = y_sorted[2:]  # 下边两点

    # 计算上下边中点
    top_mid = ((top_pts[0][0] + top_pts[1][0]) / 2, (top_pts[0][1] + top_pts[1][1]) / 2)
    bottom_mid = ((bottom_pts[0][0] + bottom_pts[1][0]) / 2, (bottom_pts[0][1] + bottom_pts[1][1]) / 2)

    # 计算中轴像素数
    return distance(top_mid, bottom_mid)


def process_frame(frame):
    """处理单帧图像的核心函数"""
    global circle_history, outer_vertical_pixels, outer_axis_pixels, current_mode, NORMAL_COMPENSATION, NORMAL_COMPENSATION2, CIRCLE_COMPENSATION_FACTOR
    global outer_height_history  # 测距历史

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
    ppcm = ((rect_w / 21.0) + (rect_h / 29.7)) / 2  # 像素/厘米比例（基于A4纸尺寸）
    print(f"ppcm值为:{ppcm}")
    # === 新增：计算外框高度（用于测距）===
    outer_height = 0.0
    if len(approx_outer) == 4:
        # 获取外框参数
        left_pix, right_pix, axis_pix = get_outer_frame_params(approx_outer)
        outer_vertical_pixels.append((left_pix, right_pix))
        outer_axis_pixels.append(axis_pix)

        # 保持历史数据长度
        if len(outer_vertical_pixels) > HISTORY_LEN:
            outer_vertical_pixels.pop(0)
        if len(outer_axis_pixels) > HISTORY_LEN:
            outer_axis_pixels.pop(0)

        # 确定当前模式的外框高度
        if current_mode == "normal":
            # 正视：取左右边平均高度
            current_height = (left_pix + right_pix) / 2
        else:
            # 斜视：取中轴高度
            current_height = axis_pix

        # 加入历史并过滤
        outer_height_history.append(current_height)
        if len(outer_height_history) > HISTORY_LEN:
            outer_height_history.pop(0)
        outer_height, _ = calculate_filtered_average(outer_height_history)

    # === 新增：估算距离并显示 ===
    distance_cm = 0.0
    if outer_height > 0:
        distance_cm = estimate_distance(outer_height)
        # 选取本次用的补偿系数
        comp = get_compensation(distance_cm)
        NORMAL_COMPENSATION = comp['normal']
        NORMAL_COMPENSATION2 = comp['normal2']
        CIRCLE_COMPENSATION_FACTOR = comp['circle']

        # 在图上显示距离
        cv2.putText(frame, f"Dist: {distance_cm:.1f}cm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        # 打印距离信息
        print(f"外框高度: {outer_height:.1f}像素 -> 距离估算: {distance_cm:.1f}cm")

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
            lengths = []
            compensation = NORMAL_COMPENSATION


            # 正方形处理
            if len(approx) == 4:
                # 确定当前旋转等级和补偿系数
                rotation_level = "正视"

                # 足够的外框历史数据才进行模式判断
                if len(outer_vertical_pixels) >= HISTORY_LEN:
                    # 计算平均左右像素差比例
                    avg_left = np.mean([p[0] for p in outer_vertical_pixels])
                    avg_right = np.mean([p[1] for p in outer_vertical_pixels])
                    # pixel_diff_ratio = abs(avg_left - avg_right) / max(avg_left, avg_right, 1e-6)
                    # 根据当前模式设置补偿系数

                    compensation = NORMAL_COMPENSATION
                for i in range(4):
                    p1 = tuple(approx[i][0])
                    p2 = tuple(approx[(i + 1) % 4][0])
                    d = distance(p1, p2) / ppcm
                    d_compensated = d * compensation
                    lengths.append(d_compensated)

                avg_length, _ = calculate_filtered_average(lengths)
                print(f"正方形（{rotation_level}）: {avg_length:.2f}cm")
                cv2.putText(img, f"S={avg_length:.2f}cm", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)



            # 三角形处理
            elif len(approx) == 3:
                for i in range(3):
                    p1 = tuple(approx[i][0])
                    p2 = tuple(approx[(i + 1) % 3][0])
                    d = distance(p1, p2) / ppcm
                    d_compensated = d * NORMAL_COMPENSATION2  # 使用正视补偿系数
                    #d_compensated = d
                    lengths.append(d_compensated)

                avg_length, _ = calculate_filtered_average(lengths)
                print(f"三角形: {avg_length:.2f}cm")
                cv2.putText(img, f"S={avg_length:.2f}cm", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 圆形处理
            else:
                (x0, y0), radius = cv2.minEnclosingCircle(cnt)
                diameter = 2 * radius / ppcm
                diameter_compensated = diameter * CIRCLE_COMPENSATION_FACTOR
                #diameter_compensated = diameter
                circle_history.append(diameter_compensated)
                if len(circle_history) > CIRCLE_MEASURE_COUNT:
                    circle_history.pop(0)

                avg_diameter, _ = calculate_filtered_average(circle_history)
                print(f"圆形: {avg_diameter:.2f}cm")
                cv2.putText(img, f"D={avg_diameter:.2f}cm", (int(x0), int(y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img
if __name__ == "__main__":
    # 初始化模式状态变量
    current_mode = "normal"  # "normal", "slight", "moderate", "large"

    # 连续视频流模式
    if ENABLE_CONTINUOUS_VIDEO:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("无法打开视频流")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(fps / 10))
        frame_count = 0

        print("连续视频流模式 - 按ESC退出")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_interval != 0:
                continue

            processed_frame = process_frame(frame)
            cv2.imshow("Result", processed_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
                break

        cap.release()
        cv2.destroyAllWindows()

    # 单帧拍摄模式
    elif ENABLE_SINGLE_FRAME:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        # 设置分辨率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1600)

        print("单帧拍摄模式 - 等待摄像头启动...")
        # 等待1秒确保摄像头启动
        time.sleep(1)

        # 捕获一帧图像（可能需要丢弃前几帧无效数据）
        for _ in range(3):
            cap.read()
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError("无法捕获图像")

        # 释放摄像头资源
        cap.release()

        print("已捕获图像 - 按ESC退出")
        processed_frame = process_frame(frame)
        cv2.imshow("Result", processed_frame)
        # 等待ESC键退出
        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    # 本地图片模式
    elif ENABLE_LOCAL_IMAGE:
        # 请替换为你的本地图片路径
        image_path = "../3.png"  # 支持jpg、png格式
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")

        print("本地图片模式 - 按ESC退出")
        processed_frame = process_frame(frame)
        cv2.imshow("Result", processed_frame)

        # 等待ESC键退出
        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
