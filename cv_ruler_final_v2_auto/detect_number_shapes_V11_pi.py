# -*- coding: gbk -*-
import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple
import subprocess
import sys
import argparse
# === 模式控制开关 ===
ENABLE_CONTINUOUS_VIDEO = 0  # 连续视频流模式
ENABLE_SINGLE_FRAME = 1  # 单帧拍摄模式
ENABLE_LOCAL_IMAGE = 0  # 本地图片模式
# 提示：仅能开启一种模式，本地图片请确保路径正确（jpg/png格式）

# === 核心参数设置 ===
NORMAL_COMPENSATION = 1.013  # 正视模式补偿系数
OUTLIER_THRESHOLD = 0.2  # 异常值过滤阈值
HISTORY_LEN = 3  # 历史数据长度
ORPHAN_ANGLE_THRESHOLD = 15  # 落单角斜率匹配阈值（度）
SIDE_LENGTH_TOLERANCE = 0.3  # 边长容忍度（10%）

# 历史数据存储（仅保留外框中轴像素数用于比例计算）
outer_axis_pixels = []
outer_height_history = []

global input_num
input_num = None

outer_vertical_pixels = []
def save_squares_to_file():
    squares = get_all_squares()
    with open('squares_data.txt', 'w') as f:
        for square in squares:
            f.write(f"Vertices: {square.vertices}, Avg Side Length: {square.avg_side_length}\n")


def get_contour_center(contour):
    """计算轮廓中心点"""
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return (0, 0)
    return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))


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


def calculate_vectors(p_prev, p_curr, p_next):
    """计算当前点与前后点的向量"""
    v_prev = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])  # 前向量
    v_next = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])  # 后向量
    return v_prev, v_next

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

def read_button():
    # 假设你从某个输入设备读取按钮状态（如 GPIO）
    # 示例：返回按钮编号（1~4）
    return int(sys.argv[1])  # 获取命令行参数（假设你传入按钮编号）

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

# 定义数据类存储正方形信息
@dataclass
class SquareInfo:
    vertices: List[Tuple[int, int]]  # 四个顶点坐标 (x,y)
    center: Tuple[int, int]  # 几何中心坐标
    avg_side_length: float  # 平均边长（像素）
    detection_method: str  # 检测方法："perfect_edge" 或 "orphan_corner"
    is_valid: bool = True  # 是否为有效正方形

    def __post_init__(self):
        # 计算几何中心（如果未指定）
        if not self.center and self.vertices:
            x_coords = [v[0] for v in self.vertices]
            y_coords = [v[1] for v in self.vertices]

            # 确保 sum() 得到的结果是单一数值，避免类型错误
            self.center = (int(sum(x_coords) / len(x_coords)), int(sum(y_coords) / len(y_coords)))


# 全局列表存储所有检测到的正方形
all_squares: List[SquareInfo] = []


def reset_square_collection():
    """重置正方形收集列表，用于处理新帧"""
    global all_squares
    all_squares = []


def add_square_to_collection(vertices, avg_side, method, center=None):
    """将检测到的正方形添加到收集列表"""
    square = SquareInfo(
        vertices=vertices,
        avg_side_length=avg_side,
        detection_method=method,
        center=center
    )
    all_squares.append(square)
    return square


def get_all_squares() -> List[SquareInfo]:
    """获取所有收集到的正方形信息"""
    return all_squares


def angle_between_vectors(v1, v2):
    """计算两个向量之间的夹角（度）"""
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    norm_v1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
    norm_v2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)  # 限制范围避免计算误差
    return abs(np.degrees(np.arccos(cos_theta)))


def sort_square_vertices(p1, p2, p3, p4):
    """将四个点按顺时针顺序排序，确保连接顺序正确"""
    # 计算中心点
    center = np.mean([p1, p2, p3, p4], axis=0)

    # 计算每个点相对于中心点的角度
    def get_angle(point):
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return np.arctan2(dy, dx)  # 角度范围：-π 到 π

    # 按角度排序（顺时针）
    points = [p1, p2, p3, p4]
    points.sort(key=lambda p: get_angle(p), reverse=True)  # reverse=True 转为顺时针
    return points


def detect_right_angles(contour, img, return_details=False):
    """识别并标记图形中的直角点（85-95度），返回直角点及相关细节"""
    # 1. 轮廓逼近（保留更多顶点，确保锐角不被合并）
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.003 * perimeter  # 更精细的逼近，保留更多顶点
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = approx.reshape(-1, 2)  # 转换为顶点列表
    num_vertices = len(vertices)

    if num_vertices < 3:
        return img, [] if not return_details else (img, [])

    right_angles = []

    # 2. 遍历所有顶点检测直角
    for i in range(num_vertices):
        # 获取当前顶点及相邻顶点
        p_curr = vertices[i]
        p_prev = vertices[(i - 1) % num_vertices]  # 前一个顶点
        p_next = vertices[(i + 1) % num_vertices]  # 后一个顶点

        # 计算向量和夹角
        v_prev, v_next = calculate_vectors(p_prev, p_curr, p_next)
        angle = angle_between_vectors(v_prev, v_next)

        # 3. 判断是否为直角（85-95度）
        if 82 <= angle <= 97:
            # 计算两边的斜率
            slope_prev = calculate_slope(p_prev, p_curr)
            slope_next = calculate_slope(p_curr, p_next)

            right_angles.append({
                'point': p_curr,
                'index': i,
                'angle': angle,
                'slopes': [slope_prev, slope_next],
                'neighbors': [p_prev, p_next]
            })

            if not return_details:  # 仅在不需要详细信息时绘制
                # 标记直角点（红色圆点）
                cv2.circle(img, tuple(p_curr), 8, (0, 0, 255), -1)
                # 显示角度值
                cv2.putText(img, f"{angle:.1f}°",
                            (p_curr[0] + 10, p_curr[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 显示检测到的直角数量
    if not return_details:
        cv2.putText(img, f"直角点: {len(right_angles)}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if return_details:
        return img, right_angles
    return img, right_angles


def calculate_slope(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if abs(dx) < 1e-6:
        return float('inf')
    return dy / dx


def get_line_orientation(slope):
    if slope == float('inf'):
        return "vertical"
    if abs(slope) < 0.1:
        return "horizontal"
    return "diagonal"


def angle_between_slopes(s1, s2):
    if s1 == float('inf'):
        s1 = 1e10
    if s2 == float('inf'):
        s2 = 1e10
    tan_theta = abs((s2 - s1) / (1 + s1 * s2))
    return np.degrees(np.arctan(tan_theta))


def check_direction_consistency(p0, p1, p2, p3):
    s_prev = calculate_slope(p0, p1)
    s_curr = calculate_slope(p1, p2)
    s_next = calculate_slope(p2, p3)

    orient_curr = get_line_orientation(s_curr)

    def is_on_side(point, line_p1, line_p2):
        return (line_p2[0] - line_p1[0]) * (point[1] - line_p1[1]) - \
            (line_p2[1] - line_p1[1]) * (point[0] - line_p1[0])

    prev_ref = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
    next_ref = ((p2[0] + p3[0]) // 2, (p2[1] + p3[1]) // 2)

    prev_side = np.sign(is_on_side(prev_ref, p1, p2))
    next_side = np.sign(is_on_side(next_ref, p1, p2))
    same_side = prev_side == next_side and prev_side != 0

    if orient_curr == "horizontal":
        direction = "up" if prev_side > 0 else "down"
    elif orient_curr == "vertical":
        direction = "right" if prev_side > 0 else "left"
    else:
        direction = "same_side"

    return same_side, direction, prev_side


def check_black_region(img, p1, p2, prev_side, step_pixels=30, threshold=0.70):
    """改进的黑色区域检测：根据边的长度动态调整采样点数量"""
    # 计算边的长度和方向向量
    curr_dx = p2[0] - p1[0]
    curr_dy = p2[1] - p1[1]
    edge_length = np.hypot(curr_dx, curr_dy)  # 边的实际长度

    if edge_length < 1e-6:  # 避免除以零
        return False, 0.0

    # 根据边的长度确定采样点数量和分布
    sample_points = []
    if edge_length < 50:  # 超短边
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        sample_points.append(mid)
        required_passed = 1
    elif 50 <= edge_length < 100:  # 短边
        for i in [1, 2]:
            x = p1[0] + int((curr_dx * i) / 3)
            y = p1[1] + int((curr_dy * i) / 3)
            sample_points.append((x, y))
        required_passed = 1
    else:  # 长边
        for i in [1, 2, 3, 4, 5]:
            x = p1[0] + int((curr_dx * i) / 6)
            y = p1[1] + int((curr_dy * i) / 6)
            sample_points.append((x, y))
        required_passed = 3

    # 计算垂直方向单位向量
    raw_normal_dx = -curr_dy  # 垂直方向x分量
    raw_normal_dy = curr_dx  # 垂直方向y分量

    # 归一化垂直向量（确保长度为1）
    normal_len = np.hypot(raw_normal_dx, raw_normal_dy)
    unit_normal_dx = raw_normal_dx / normal_len
    unit_normal_dy = raw_normal_dy / normal_len

    # 应用方向（prev_side控制左右/上下）
    final_normal_dx = unit_normal_dx * prev_side
    final_normal_dy = unit_normal_dy * prev_side

    # 存储所有采样点的检测结果
    all_ratios = []
    passed_count = 0

    # 对每个采样点进行检测
    for (x, y) in sample_points:
        # 计算偏移检测点
        test_x = int(x + final_normal_dx * step_pixels)
        test_y = int(y + final_normal_dy * step_pixels)
        test_point = (test_x, test_y)

        # 绘制采样点标记和偏移线段（便于调试）
        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)  # 蓝色：原采样点
        cv2.circle(img, test_point, 3, (0, 128, 255), -1)  # 橙色：偏移后检测点
        cv2.line(img, (x, y), test_point, (0, 255, 0), 1)  # 绿色：偏移线段

        # 计算该点周围的黑色比例
        h, w = img.shape[:2]
        size = 20
        half = size // 2
        x1 = max(0, test_x - half)
        x2 = min(w, test_x + half + 1)
        y1 = max(0, test_y - half)
        y2 = min(h, test_y + half + 1)

        region = img[y1:y2, x1:x2]
        if len(region.shape) == 3:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(region, 100, 255, cv2.THRESH_BINARY)

        total = binary.size
        dark = total - cv2.countNonZero(binary)
        ratio = dark / total if total > 0 else 0.0
        all_ratios.append(ratio)

        cv2.putText(img, f"{int(ratio * 100)}%",
                    (test_x + 5, test_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)

        if ratio >= threshold:
            passed_count += 1

    avg_ratio = np.mean(all_ratios) if all_ratios else 0.0
    return passed_count >= required_passed, avg_ratio


def infer_square_points(p1, p2, prev_side):
    """严格按照相邻边方向推测正方形顶点，不反向"""
    edge_length = np.hypot(p2[0] - p1[0], p2[1] - p1[1])

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    norm = np.sqrt(dx ** 2 + dy ** 2)
    if norm == 0:
        return None, None

    perp_dx = -dy / norm
    perp_dy = dx / norm
    perp_dx = perp_dx * prev_side
    perp_dy = perp_dy * prev_side

    p3 = (
        int(p1[0] + perp_dx * edge_length),
        int(p1[1] + perp_dy * edge_length)
    )
    p4 = (
        int(p2[0] + perp_dx * edge_length),
        int(p2[1] + perp_dy * edge_length)
    )

    return p3, p4


def find_matching_node(point, all_nodes, threshold=20):
    """寻找与推测点最匹配的原始节点"""
    min_distance = float('inf')
    best_match_idx = None
    best_match_node = None

    for idx, node in enumerate(all_nodes):
        dist = np.hypot(node[0] - point[0], node[1] - point[1])

        if dist <= threshold and dist < min_distance:
            min_distance = dist
            best_match_idx = idx
            best_match_node = node

    if best_match_idx is not None:
        return best_match_idx, best_match_node
    else:
        return None, None


def get_square_side_lengths(p1, p2, p3, p4):
    """计算正方形四边长度并返回平均值，确保数值有效"""
    try:
        s1 = distance(p1, p2)
        s2 = distance(p2, p3)
        s3 = distance(p3, p4)
        s4 = distance(p4, p1)
        avg = (s1 + s2 + s3 + s4) / 4
        # 确保返回合理的数值
        return round(avg, 1) if avg > 0 else 0.0
    except Exception as e:
        print(f"计算边长时出错: {e}")
        return 0.0


def is_edge_part_of_existing_square(edge, square_groups, threshold=30):
    """检查边是否已属于已识别的正方形"""
    p1, p2 = edge
    for square in square_groups:
        p1_in = any(np.hypot(p1[0] - sp[0], p1[1] - sp[1]) <= threshold for sp in square)
        p2_in = any(np.hypot(p2[0] - sp[0], p2[1] - sp[1]) <= threshold for sp in square)
        if p1_in and p2_in:
            return True
    return False


def detect_perfect_edges_and_squares(contour, img, step_pixels=30):
    # 基础轮廓处理
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.003 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    original_vertices = approx.reshape(-1, 2)
    num_original = len(original_vertices)
    all_vertices = original_vertices.copy().tolist()
    processed_edges = set()  # 用于避免重复处理边
    square_groups = []

    # 初始状态设置
    vertex_status = {i: "未核验" for i in range(num_original)}
    edge_status = {}

    # 绘制原始顶点
    cv2.drawContours(img, [approx], -1, (147, 112, 219), 2)
    for i in range(num_original):
        cv2.putText(img, f"{i + 1}",
                    (original_vertices[i][0] + 5, original_vertices[i][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(img, (original_vertices[i][0], original_vertices[i][1]), 5, (0, 255, 0), -1)

    # 检测完美边
    perfect_edges = []
    for i in range(num_original):
        p1_idx = i
        p2_idx = (i + 1) % num_original
        p1 = original_vertices[p1_idx]
        p2 = original_vertices[p2_idx]
        edge_name = f"l{p1_idx + 1}l{p2_idx + 1}"

        p0 = original_vertices[(i - 1) % num_original]
        p3 = original_vertices[(i + 2) % num_original]

        # 角度检测
        prev_len = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        next_len = np.hypot(p3[0] - p2[0], p3[1] - p2[1])
        s_prev = calculate_slope(p0, p1)
        s_curr = calculate_slope(p1, p2)
        s_next = calculate_slope(p2, p3)
        angle_p1 = angle_between_slopes(s_prev, s_curr)
        angle_p2 = angle_between_slopes(s_curr, s_next)

        # 极短边放宽阈值
        is_right_p1 = 80 <= angle_p1 <= 100 if prev_len < 100 else 85 <= angle_p1 <= 95
        is_right_p2 = 80 <= angle_p2 <= 100 if next_len < 100 else 85 <= angle_p2 <= 95

        # 方向检测
        dir_consistent, direction, prev_side = check_direction_consistency(p0, p1, p2, p3)

        # 黑色区域检测
        black_valid = False
        if is_right_p1 and is_right_p2 and dir_consistent:
            black_valid, _ = check_black_region(
                img, p1, p2, prev_side,
                step_pixels=step_pixels,
                threshold=0.70
            )

        # 标记完美边
        if is_right_p1 and is_right_p2 and dir_consistent and black_valid:
            perfect_edges.append((p1_idx, p2_idx, p1, p2, direction, prev_side))
            edge_status[edge_name] = "已核验"
            vertex_status[p1_idx] = "已核验"
            vertex_status[p2_idx] = "已核验"
            cv2.line(img, tuple(p1), tuple(p2), (255, 0, 0), 3)
        else:
            edge_status[edge_name] = "未核验"
            cv2.line(img, tuple(p1), tuple(p2), (0, 0, 255), 3)

        # 调整边名称显示位置
        edge_mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)  # 向上偏移避免重叠
        cv2.putText(img, edge_name, edge_mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 基于完美边推测正方形
    for edge in perfect_edges:
        p1_idx, p2_idx, p1, p2, direction, prev_side = edge
        edge_key = frozenset([p1_idx, p2_idx])

        # 跳过已处理的边或已属于正方形的边
        if edge_key in processed_edges or is_edge_part_of_existing_square((p1, p2), square_groups):
            processed_edges.add(edge_key)
            continue

        # 推测正方形顶点（仅使用prev_side）
        p3, p4 = infer_square_points(p1, p2, prev_side)
        if p3 is None or p4 is None:
            continue

        # 匹配已有节点
        match3_idx, match3 = find_matching_node(p3, all_vertices)
        match4_idx, match4 = find_matching_node(p4, all_vertices)

        # 初始化原始节点集合
        original_matched_indices = set([p1_idx, p2_idx])

        # 处理p3
        if match3_idx is not None and match3_idx < num_original:
            p3_final = match3
            original_matched_indices.add(match3_idx)
            vertex_status[match3_idx] = "已核验"
        else:
            all_vertices.append(p3)
            p3_final = p3
            cv2.circle(img, p3_final, 5, (0, 255, 255), -1)

        # 处理p4
        if match4_idx is not None and match4_idx < num_original:
            p4_final = match4
            original_matched_indices.add(match4_idx)
            vertex_status[match4_idx] = "已核验"
        else:
            all_vertices.append(p4)
            p4_final = p4
            cv2.circle(img, p4_final, 5, (0, 255, 255), -1)

        # 对四个顶点进行顺时针排序
        sorted_points = sort_square_vertices(p1, p2, p3_final, p4_final)
        s1, s2, s3, s4 = sorted_points

        # 绘制正方形
        square = (s1, s2, s3, s4)
        square_groups.append(square)
        cv2.line(img, tuple(s1), tuple(s2), (255, 255, 0), 2)
        cv2.line(img, tuple(s2), tuple(s3), (255, 255, 0), 2)
        cv2.line(img, tuple(s3), tuple(s4), (255, 255, 0), 2)
        cv2.line(img, tuple(s4), tuple(s1), (255, 255, 0), 2)

        # 显示平均边长
        avg_len = get_square_side_lengths(*square)
        add_square_to_collection(vertices=[(s1[0], s1[1]), (s2[0], s2[1]), (s3[0], s3[1]), (s4[0], s4[1])],
                                 avg_side=avg_len, method="perfect_edge")
        mid_square = ((s1[0] + s3[0]) // 2, (s1[1] + s3[1]) // 2)
        cv2.putText(img, f"= {avg_len:.1f}", mid_square,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # 更新顶点和边状态
        for idx in original_matched_indices:
            prev_idx = (idx - 1) % num_original
            next_idx = (idx + 1) % num_original
            vertex_status[prev_idx] = "已核验"
            vertex_status[next_idx] = "已核验"

        # 更新线段状态
        for i in range(num_original):
            p1_idx = i
            p2_idx = (i + 1) % num_original
            edge_name = f"l{p1_idx + 1}l{p2_idx + 1}"

            if vertex_status[p1_idx] == "已核验" and vertex_status[p2_idx] == "已核验":
                edge_status[edge_name] = "已核验"
                cv2.line(img,
                         tuple(original_vertices[p1_idx]),
                         tuple(original_vertices[p2_idx]),
                         (255, 0, 0), 3)

        # 标记正方形所有边为已处理
        square_edges = [
            edge_key,
            frozenset([p2_idx, match3_idx]) if (match3_idx is not None and match3_idx < num_original) else None,
            frozenset([match3_idx, match4_idx]) if (match3_idx is not None and match4_idx is not None and
                                                    match3_idx < num_original and match4_idx < num_original) else None,
            frozenset([match4_idx, p1_idx]) if (match4_idx is not None and match4_idx < num_original) else None
        ]
        for se in square_edges:
            if se is not None:
                processed_edges.add(se)

    # 整理未核验节点（仅原始节点）
    unverified_vertices = [i for i, status in vertex_status.items() if status == "未核验" and i < num_original]

    # 显示未核验节点
    if unverified_vertices:
        cv2.putText(img, "未核验: " + ",".join(map(str, [i + 1 for i in unverified_vertices])),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return img, vertex_status, edge_status, square_groups, original_vertices


# === 落单角法实现 ===
def extend_line(p1, p2, length_ratio=2.0):  # 延长线长度调整为2倍
    """延长线段，返回延长长后的端点"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    extended_p = (
        int(p2[0] + dx * length_ratio),
        int(p2[1] + dy * length_ratio)
    )
    return extended_p


def line_intersection(p1, p2, p3, p4):
    """计算两条线段(p1-p2和p3-p4)的交点"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # 平行线，无交点

    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    t = t_num / denom

    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))
    u = u_num / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        return (int(x), int(y))
    return None


def check_black_pixel_percentage(img, point, radius=25):
    """检查指定点周围指定半径内黑色像素的比例，添加边界检查"""
    h, w = img.shape[:2]
    x, y = point

    # 首先检查点是否在图像范围内
    if x < 0 or x >= w or y < 0 or y >= h:
        print(f"警告：点({x},{y})超出图像范围")
        return 0.0  # 超出范围视为不满足条件

    # 计算区域边界（确保在图像范围内）
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    # 检查区域是否有效（至少有1x1像素）
    if x1 >= x2 or y1 >= y2:
        print(f"警告：无效区域 ({x1},{y1})-({x2},{y2})")
        return 0.0

    # 提取区域并转换为灰度
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        print(f"警告：空区域 ({x1},{y1})-({x2},{y2})")
        return 0.0

    if len(region.shape) == 3:
        try:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            print(f"转换颜色空间错误: {e}")
            return 0.0

    # 二值化（黑色视为0，其他视为255）
    _, binary = cv2.threshold(region, 100, 255, cv2.THRESH_BINARY)

    # 计算黑色像素比例
    total = binary.size
    if total == 0:
        return 0.0
    black = total - cv2.countNonZero(binary)
    return black / total


def check_slope_match(slopes1, slopes2):
    """检查两个直角的斜率是否匹配（k1,k2 和 k2,k1）"""
    if len(slopes1) != 2 or len(slopes2) != 2:
        return False

    # 检查两种可能的匹配方式
    match1 = (angle_between_slopes(slopes1[0], slopes2[1]) < ORPHAN_ANGLE_THRESHOLD and
              angle_between_slopes(slopes1[1], slopes2[0]) < ORPHAN_ANGLE_THRESHOLD)

    match2 = (angle_between_slopes(slopes1[0], slopes2[0]) < ORPHAN_ANGLE_THRESHOLD and
              angle_between_slopes(slopes1[1], slopes2[1]) < ORPHAN_ANGLE_THRESHOLD)

    return match1 or match2


def orphan_corner_method(img, contour, unverified_indices, original_vertices, square_groups):
    """落单角法：优化同边点场景判断，增加黑色区域验证"""
    # 获取所有直角点的详细信息
    _, all_right_angles = detect_right_angles(contour, img, return_details=True)

    # 筛选出未核验的直角点
    orphan_corners = [
        corner for corner in all_right_angles
        if corner['index'] in unverified_indices
    ]

    # 调试输出：落单角数量
    print(f"\n===== 落单角法开始 =====")
    print(f"检测到未核验的落单角数量: {len(orphan_corners)}")
    for i, corner in enumerate(orphan_corners):
        print(f"落单角 {i + 1}: 坐标={corner['point']}, 斜率={corner['slopes']}")

    # 至少需要两个落单角才能继续
    if len(orphan_corners) < 2:
        print("落单角数量不足2个，无法进行匹配")
        print("===== 落单角法结束 =====")
        return img, square_groups

    cv2.putText(img, f"落单角数量: {len(orphan_corners)}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 检查所有落单角组合
    for i in range(len(orphan_corners)):
        for j in range(i + 1, len(orphan_corners)):
            corner1 = orphan_corners[i]
            corner2 = orphan_corners[j]
            print(f"\n----- 检查落单角组合: 角{i + 1}与角{j + 1} -----")

            # 检查斜率是否匹配
            slope_match = check_slope_match(corner1['slopes'], corner2['slopes'])
            print(f"斜率匹配结果: {slope_match}")
            print(f"角{i + 1}斜率: {corner1['slopes']}, 角{j + 1}斜率: {corner2['slopes']}")

            if not slope_match:
                print("斜率不匹配，跳过该组合")
                continue

            # 标记匹配的落单角
            cv2.circle(img, tuple(corner1['point']), 10, (255, 0, 255), 2)
            cv2.circle(img, tuple(corner2['point']), 10, (255, 0, 255), 2)
            cv2.line(img, tuple(corner1['point']), tuple(corner2['point']), (255, 0, 255), 1)

            p1 = corner1['point']
            p1a, p1b = corner1['neighbors']
            p2 = corner2['point']
            p2a, p2b = corner2['neighbors']

            # 计算角1两条边的斜率及延长线
            slope1a = calculate_slope(p1, p1a)
            slope1b = calculate_slope(p1, p1b)
            ext_p1a = extend_line(p1, p1a, length_ratio=3.0)
            ext_p1b = extend_line(p1, p1b, length_ratio=3.0)

            # 计算角2两条边的斜率及延长线
            slope2a = calculate_slope(p2, p2a)
            slope2b = calculate_slope(p2, p2b)
            ext_p2a = extend_line(p2, p2a, length_ratio=3.0)
            ext_p2b = extend_line(p2, p2b, length_ratio=3.0)

            # 调试输出：各射线斜率
            print(f"角1射线斜率: {slope1a} (延长至{ext_p1a}), {slope1b} (延长至{ext_p1b})")
            print(f"角2射线斜率: {slope2a} (延长至{ext_p2a}), {slope2b} (延长至{ext_p2b})")

            # 绘制所有延长线（不同颜色便于区分）
            cv2.line(img, tuple(p1), ext_p1a, (0, 165, 255), 1, cv2.LINE_AA)  # 橙色
            cv2.line(img, tuple(p1), ext_p1b, (255, 165, 0), 1, cv2.LINE_AA)  # 橙色反色
            cv2.line(img, tuple(p2), ext_p2a, (0, 165, 255), 1, cv2.LINE_AA)  # 橙色
            cv2.line(img, tuple(p2), ext_p2b, (255, 165, 0), 1, cv2.LINE_AA)  # 橙色反色

            # 按斜率正负特性配对射线
            if (slope1a < 0 and slope1b > 0) or (slope1a > 0 and slope1b < 0):
                neg_slope_ray1 = (p1, ext_p1a) if slope1a < 0 else (p1, ext_p1b)
                pos_slope_ray1 = (p1, ext_p1b) if slope1a < 0 else (p1, ext_p1a)
            else:
                neg_slope_ray1 = (p1, ext_p1a) if abs(slope1a) > abs(slope1b) else (p1, ext_p1b)
                pos_slope_ray1 = (p1, ext_p1b) if abs(slope1a) > abs(slope1b) else (p1, ext_p1a)

            if (slope2a < 0 and slope2b > 0) or (slope2a > 0 and slope2b < 0):
                neg_slope_ray2 = (p2, ext_p2a) if slope2a < 0 else (p2, ext_p2b)
                pos_slope_ray2 = (p2, ext_p2b) if slope2a < 0 else (p2, ext_p2a)
            else:
                neg_slope_ray2 = (p2, ext_p2a) if abs(slope2a) > abs(slope2b) else (p2, ext_p2b)
                pos_slope_ray2 = (p2, ext_p2b) if abs(slope2a) > abs(slope2b) else (p2, ext_p2a)

            # 计算交点
            intersect1 = line_intersection(neg_slope_ray1[0], neg_slope_ray1[1],
                                           pos_slope_ray2[0], pos_slope_ray2[1])
            intersect2 = line_intersection(pos_slope_ray1[0], pos_slope_ray1[1],
                                           neg_slope_ray2[0], neg_slope_ray2[1])

            # 调试输出：交点信息
            print(f"负斜率射线与正斜率射线交点1: {intersect1}")
            print(f"正斜率射线与负斜率射线交点2: {intersect2}")

            # 收集有效交点
            valid_intersections = []
            if intersect1:
                valid_intersections.append(intersect1)
                cv2.circle(img, intersect1, 6, (0, 255, 0), -1)  # 绿色标记有效交点
            if intersect2:
                valid_intersections.append(intersect2)
                cv2.circle(img, intersect2, 6, (0, 255, 0), -1)  # 绿色标记标记有效交点

            # 情况1：对角点场景 - 有两个有效交点
            if len(valid_intersections) >= 2:
                print(f"检测到{len(valid_intersections)}个有效交点，进入对角点场景判断")

                # 取前两个有效交点
                intersect1, intersect2 = valid_intersections[:2]

                # 计算四条边的长度
                s1 = distance(p1, intersect1)
                s2 = distance(intersect1, p2)
                s3 = distance(p2, intersect2)
                s4 = distance(intersect2, p1)

                # 计算长度差异
                lengths = [s1, s2, s3, s4]
                max_len = max(lengths)
                min_len = min(lengths)
                length_diff = (max_len - min_len) / max_len if max_len > 0 else 0

                if length_diff <= SIDE_LENGTH_TOLERANCE:
                    print("边长差异在容忍范围内，判定为对角点构成的正方形")
                    square_points = [p1, intersect1, p2, intersect2]
                    sorted_points = sort_square_vertices(*square_points)

                    # 计算平均边长（确保得到有效数值）
                    avg_length = get_square_side_lengths(*sorted_points)
                    # add_square_to_collection(vertices=[(p1, p2, selected_p3, selected_p4)], avg_side=avg_length,
                    #                          method="orphan_corner")
                    # 绘制正方形
                    cv2.line(img, tuple(sorted_points[0]), tuple(sorted_points[1]), (128, 0, 128), 2)
                    cv2.line(img, tuple(sorted_points[1]), tuple(sorted_points[2]), (128, 0, 128), 2)
                    cv2.line(img, tuple(sorted_points[2]), tuple(sorted_points[3]), (128, 0, 128), 2)
                    cv2.line(img, tuple(sorted_points[3]), tuple(sorted_points[0]), (128, 0, 128), 2)

                    # # 简洁文本输出：仅显示 "=数值"，与完美边格式一致
                    mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    if 0 <= mid_point[0] < img.shape[1] and 0 <= mid_point[1] < img.shape[0]:
                        # 只显示等号和数值，不添加其他文字
                        text = f"= {avg_length:.1f}"
                        cv2.putText(img, text, mid_point,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)  # 使用与完美边相同的颜色和大小

                    add_square_to_collection(vertices=sorted_points, avg_side=avg_length, method="orphan_corner")
                    continue
            # 情况2：同边点场景 - 优化版本，增加黑色区域验证
            print("进入同边点场景判断")
            # 计算两点距离作为边长
            side_length = distance(p1, p2)
            print(f"两点两点距离(作为边长): {side_length:.1f}像素")

            # 计算垂直方向（两种两个可能的方向）
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm = np.sqrt(dx ** 2 + dy ** 2) if (dx ** 2 + dy ** 2) > 0 else 1

            # 生成两组可能的顶点（两个不同方向）
            perp_dx1 = -dy / norm
            perp_dy1 = dx / norm
            p3_1 = (int(p1[0] + perp_dx1 * side_length), int(p1[1] + perp_dy1 * side_length))
            p4_1 = (int(p2[0] + perp_dx1 * side_length), int(p2[1] + perp_dy1 * side_length))

            perp_dx2 = dy / norm
            perp_dy2 = -dx / norm
            p3_2 = (int(p1[0] + perp_dx2 * side_length), int(p1[1] + perp_dy2 * side_length))
            p4_2 = (int(p2[0] + perp_dx2 * side_length), int(p2[1] + perp_dy2 * side_length))

            # 调试输出两组可能的顶点
            print(f"第一组推测顶点: {p3_1}, {p4_1}")
            print(f"第二组推测顶点: {p3_2}, {p4_2}")

            # 检查两组顶点的黑色区域比例
            black_thresh = 0.3  # 30%黑色像素阈值
            p3_1_black = check_black_pixel_percentage(img, p3_1)
            p4_1_black = check_black_pixel_percentage(img, p4_1)
            p3_2_black = check_black_pixel_percentage(img, p3_2)
            p4_2_black = check_black_pixel_percentage(img, p4_2)

            print(f"第一组黑色比例: p3={p3_1_black:.2%}, p4={p4_1_black:.2%}")
            print(f"第二组黑色比例: p3={p3_2_black:.2%}, p4={p4_2_black:.2%}")

            # 判断哪组顶点更符合条件（两点都超过阈值）
            group1_valid = p3_1_black >= black_thresh and p4_1_black >= black_thresh
            group2_valid = p3_2_black >= black_thresh and p4_2_black >= black_thresh

            # 选择有效的顶点组
            if group1_valid and group2_valid:
                # 都有效时选黑色比例更高的组
                group1_avg = (p3_1_black + p4_1_black) / 2
                group2_avg = (p3_2_black + p4_2_black) / 2
                selected_p3, selected_p4 = (p3_1, p4_1) if group1_avg > group2_avg else (p3_2, p4_2)
                print(f"两组都有效，选择黑色比例更高的组")
            elif group1_valid:
                selected_p3, selected_p4 = p3_1, p4_1
                print(f"选择第一组顶点")
            elif group2_valid:
                selected_p3, selected_p4 = p3_2, p4_2
                print(f"选择第二组顶点")
            else:
                # 都无效时选黑色比例较高的组
                group1_avg = (p3_1_black + p4_1_black) / 2
                group2_avg = (p3_2_black + p4_2_black) / 2
                selected_p3, selected_p4 = (p3_1, p4_1) if group1_avg > group2_avg else (p3_2, p4_2)
                print(f"两组都不满足条件，选择相对较好的组")

            # 计算正方形边长（四边平均值）
            square_points = [p1, p2, selected_p4, selected_p3]
            try:
                sorted_points = sort_square_vertices(*square_points)
                avg_length = get_square_side_lengths(*sorted_points)
            except Exception as e:
                print(f"排序顶点或计算边长失败: {e}")
                avg_length = round(side_length, 1)  # 使用备选值

            # 绘制正方形
            cv2.line(img, tuple(sorted_points[0]), tuple(sorted_points[1]), (128, 0, 128), 2)
            cv2.line(img, tuple(sorted_points[1]), tuple(sorted_points[2]), (128, 0, 128), 2)
            cv2.line(img, tuple(sorted_points[2]), tuple(sorted_points[3]), (128, 0, 128), 2)
            cv2.line(img, tuple(sorted_points[3]), tuple(sorted_points[0]), (128, 0, 128), 2)

            # 简洁文本输出：仅显示 "=数值"，与完美边格式一致
            mid_point = ((p1[0] + selected_p3[0]) // 2, (p1[1] + selected_p3[1]) // 2)
            if 0 <= mid_point[0] < img.shape[1] and 0 <= mid_point[1] < img.shape[0]:
                # 只显示等号和数值，不添加其他文字
                text = f"= {avg_length:.1f}"
                cv2.putText(img, text, mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)  # 使用与完美边相同的颜色和大小
            else:
                print(f"文本位置超出图像范围: {mid_point}")

            # # 标记验证结果
            # cv2.circle(img, selected_p3, 5, (0, 255, 0) if group1_valid else (0, 165, 255), -1)
            # cv2.circle(img, selected_p4, 5, (0, 255, 0) if group1_valid else (0, 165, 255), -1)

            square_groups.append(sorted_points)
            print(f"已创建同边点场景的正方形，平均边长: {avg_length:.1f}像素")
            add_square_to_collection(vertices=sorted_points, avg_side=avg_length, method="orphan_corner")
    print("\n===== 落单角法结束 =====")
    return img, square_groups


def process_frame(frame, button_id: int):
    """处理单帧图像的核心函数（包含完美边法和落单角法）"""
    global outer_axis_pixels,outer_vertical_pixels,outer_height_history,current_mode
    global ppcm,outer_height ,distance_cm, input_num
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
    print(rect[1])
    ppcm = ((rect_w / 21.0) + (rect_h / 29.7)) / 2  # 像素/厘米比例（基于A4纸尺寸）

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


        current_height = (left_pix + right_pix) / 2

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

        # 在图上显示距离
        cv2.putText(frame, f"Dist: {distance_cm:.1f}cm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        # 打印距离信息
        #print(f"外框高度: {outer_height:.1f}像素 -> 距离估算: {distance_cm:.1f}cm")

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [a4_outer], -1, 255, -1)
    inner_thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    contours_inner, _ = cv2.findContours(inner_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    a4_inner = None
    for cnt in contours_inner:
        area = cv2.contourArea(cnt)
        if area < 0.7 * cv2.contourArea(a4_outer) or area > 0.93 * cv2.contourArea(a4_outer):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            a4_inner = cnt
            break

    # 处理内框（真实内框或模拟内框）
    if a4_inner is not None:
        # 绘制真实内框（红色）
        cv2.drawContours(img, [a4_inner], -1, (0, 0, 255), 3)
        inner_contour = a4_inner
    else:
        # 内框识别失败，绘制模拟内框（黄色）
        print("内框识别失败，使用模拟内框")

        # 获取外框旋转矩形参数
        outer_rect = cv2.minAreaRect(a4_outer)
        outer_center, outer_size, outer_angle = outer_rect
        outer_w, outer_h = outer_size

        # 确定外框长边和短边
        if outer_w > outer_h:
            outer_long_side = outer_w
            outer_short_side = outer_h
            is_rotated = False
        else:
            outer_long_side = outer_h
            outer_short_side = outer_w
            is_rotated = True

        # 计算内框尺寸
        inner_long_side = outer_long_side * 0.865
        inner_short_side = outer_short_side * 0.762

        # 保持内框与外框同方向
        inner_size = (inner_short_side, inner_long_side) if is_rotated else (inner_long_side, inner_short_side)

        # 生成内框旋转矩形
        inner_rect = (outer_center, inner_size, outer_angle)
        inner_pts = cv2.boxPoints(inner_rect)
        inner_pts = np.int64(inner_pts)

        # 绘制模拟内框
        cv2.drawContours(img, [inner_pts], -1, (0, 255, 255), 3)
        cv2.putText(img, "模拟内框", (int(outer_center[0] - 50), int(outer_center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 创建模拟内框的轮廓
        inner_contour = inner_pts.reshape(-1, 1, 2).astype(np.int32)

    # 内部物体检测
    mask_inner = np.zeros_like(gray)
    cv2.drawContours(mask_inner, [inner_contour], -1, 255, -1)
    shrink = cv2.erode(mask_inner, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)), iterations=1)
    it2 = cv2.bitwise_and(thresh, thresh, mask=shrink)
    ci2, _ = cv2.findContours(it2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in ci2:
        if cv2.contourArea(c) < 100:
            continue
        center_pt = get_contour_center(c)
        if cv2.pointPolygonTest(inner_contour, center_pt, False) != 1:
            continue
        candidates.append(c)
    if candidates:
        # 放宽轮廓筛选条件
        inner_area = cv2.contourArea(inner_contour)
        min_valid_area = max(20, inner_area * 0.001)

        filtered = [c for c in candidates if cv2.contourArea(c) >= min_valid_area]
        if len(filtered) < 1:
            sorted_candidates = sorted(candidates, key=cv2.contourArea, reverse=True)
            filtered = sorted_candidates[:3]

        # 处理每个候选轮廓
        for cnt in filtered:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            if area < inner_area * 0.02:
                epsilon = 0.003 * perimeter
            else:
                epsilon = 0.005 * perimeter

            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # 1. 先用完美边法检测
            img, vertex_status, edge_status, square_groups, original_vertices = detect_perfect_edges_and_squares(approx,
                                                                                                                 img,
                                                                                                                 step_pixels=9)

            # 获取未核验的顶点索引
            unverified_indices = [i for i, status in vertex_status.items() if status == "未核验"]

            # 2. 如果有未核验的顶点，启用落单角法
            if unverified_indices:
                img, square_groups = orphan_corner_method(img, approx, unverified_indices, original_vertices,
                                                          square_groups)

            # 输出分类结果
            print("=== 顶点状态 ===")
            for idx, status in vertex_status.items():
                print(f"顶点{idx + 1}: {status}")

            print("\n=== 线段状态 ===")
            for edge, status in edge_status.items():
                print(f"{edge}: {status}")

            if area < inner_area * 0.02:
                cv2.putText(img, "小目标", (get_contour_center(approx)[0] + 10, get_contour_center(approx)[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    save_squares_to_file()
    
    if all_squares:
        # 1) 从大到小排序
        sorted_sqs = sorted(all_squares, key=lambda s: s.avg_side_length, reverse=True)

        # 2) 按钮编号映射到索引（0-based），超出时指向最后一个
        idx = min(button_id - 1, len(sorted_sqs) - 1)
        chosen = sorted_sqs[idx]

        # 3) 打印所选正方形信息
        real_length = chosen.avg_side_length / ppcm
        print(f"ppcm值为:{ppcm}")
        print(f"外框高度: {outer_height:.1f}像素 -> 距离估算: {distance_cm:.1f}cm")
        # print(f"按钮 {button_id} → 排名 {idx+1} 的正方形:")
        print(f"  编号{input_num}对应的正方形平均边长: {chosen.avg_side_length:.1f} 像素；对应实际尺寸: {real_length:.1f} cm")

    return img


if __name__ == "__main__":
    # 1. 新增：解析命令行参数（先处理曝光参数，再保留原有sys.argv逻辑）
    import argparse
    parser = argparse.ArgumentParser()
    # 添加曝光参数（--gain/--shutter，必须传）
    parser.add_argument('--gain', required=True, help='摄像头增益值')
    parser.add_argument('--shutter', required=True, help='摄像头快门时间（微秒）')
    # 解析曝光参数（parse_known_args() 保留未定义的参数，避免冲突）
    args, remaining_argv = parser.parse_known_args()
    
    # 2. 保留原有逻辑：从剩余参数中读取 button_id 和 input_num（对应原sys.argv[1]和sys.argv[2]）
    # 注意：remaining_argv 会过滤掉 --gain/--shutter 及其值，剩下的就是原有的模式和编号
    if len(remaining_argv) < 2:
        raise RuntimeError("请传入模式（button_id）和编号（input_num）参数")
    button_id = int(remaining_argv[0])  # 对应原 sys.argv[1]
    input_num = remaining_argv[1]       # 对应原 sys.argv[2]

    # 初始化模式
    current_mode = "normal"

    # 连续视频流（略）...

    # 3. 单帧拍摄模式：用解析到的曝光参数替换硬编码值
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
            "--gain", args.gain,    # 用解析的增益参数
            "--shutter", args.shutter,  # 用解析的快门参数
        ]
        print(f"执行拍摄命令：{' '.join(cmd)}")  # 调试用，确认参数正确
        subprocess.run(cmd, check=True)

        img = cv2.imread("image.jpg")
        if img is None:
            raise RuntimeError("读取 image.jpg 失败")

        # 原有逻辑不变（用新获取的 button_id 和 input_num）
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        processed = process_frame(rot, button_id)  # 直接用解析后的 button_id
        restored = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        output = cv2.resize(restored, (640, 480))

        cv2.imshow("Result", output)
        print("处理完成 —— 5 秒后自动退出，或按 ESC 退出")
        key = cv2.waitKey(5000)

        if key & 0xFF == 27:
            print("检测到 ESC，提前退出")
        else:
            print("5 秒到，自动退出")

        cv2.destroyAllWindows()