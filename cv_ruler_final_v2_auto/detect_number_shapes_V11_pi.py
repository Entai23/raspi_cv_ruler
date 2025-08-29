# -*- coding: gbk -*-
import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Tuple
import subprocess
import sys
import argparse
# === ģʽ���ƿ��� ===
ENABLE_CONTINUOUS_VIDEO = 0  # ������Ƶ��ģʽ
ENABLE_SINGLE_FRAME = 1  # ��֡����ģʽ
ENABLE_LOCAL_IMAGE = 0  # ����ͼƬģʽ
# ��ʾ�����ܿ���һ��ģʽ������ͼƬ��ȷ��·����ȷ��jpg/png��ʽ��

# === ���Ĳ������� ===
NORMAL_COMPENSATION = 1.013  # ����ģʽ����ϵ��
OUTLIER_THRESHOLD = 0.2  # �쳣ֵ������ֵ
HISTORY_LEN = 3  # ��ʷ���ݳ���
ORPHAN_ANGLE_THRESHOLD = 15  # �䵥��б��ƥ����ֵ���ȣ�
SIDE_LENGTH_TOLERANCE = 0.3  # �߳����̶ȣ�10%��

# ��ʷ���ݴ洢������������������������ڱ������㣩
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
    """�����������ĵ�"""
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return (0, 0)
    return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))


def distance(p1, p2):
    """������������"""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_filtered_average(values):
    """��������쳣ֵ���ƽ��ֵ"""
    if len(values) <= 2:
        return np.mean(values), values

    mean_val = np.mean(values)
    filtered = [v for v in values if abs(v - mean_val) <= OUTLIER_THRESHOLD * mean_val]

    if len(filtered) < 2:
        return mean_val, values

    return np.mean(filtered), filtered


def calculate_vectors(p_prev, p_curr, p_next):
    """���㵱ǰ����ǰ��������"""
    v_prev = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])  # ǰ����
    v_next = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])  # ������
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



# === ������εĲ���ϵ������λ��cm��===
# ��ʽ��{ (min_dist, max_dist): { 'normal':��, 'normal2':������ 'slight':��, 'moderate':��, 'large':��, 'circle':�� } }
COMPENSATION_TABLE = {
    (100, 120): { 'normal':1.006, 'normal2':1.0035, 'slight':1.005, 'moderate':1.000, 'large':1.000, 'circle':0.999 },
    (120, 140): { 'normal':1.005, 'normal2':1.0035, 'slight':1.004, 'moderate':1.009, 'large':1.009, 'circle':0.999 },
    (140, 160): { 'normal':1.005, 'normal2':1.0035, 'slight':1.003, 'moderate':1.008, 'large':1.008, 'circle':0.999 },
    (160, 180): { 'normal':1.005, 'normal2':1.0034, 'slight':1.002, 'moderate':1.007, 'large':1.007, 'circle':0.999 },
    (180, 200): { 'normal':1.0053, 'normal2':1.004, 'slight':1.001, 'moderate':1.006, 'large':1.006, 'circle':0.999 },
    (200, 300): { 'normal':1.006, 'normal2':1.0045, 'slight':1.000, 'moderate':1.005, 'large':1.005, 'circle':0.999 },
}

# === �����������ĺ��� ===
def estimate_distance(pixel_height):
    """�������߶Ȳ�ֵ������루cm��"""
    if pixel_height <= CALIB_PIXELS[0]:
        k = (CALIB_DISTANCES[1] - CALIB_DISTANCES[0]) / (CALIB_PIXELS[1] - CALIB_PIXELS[0])
        return CALIB_DISTANCES[0] + k * (pixel_height - CALIB_PIXELS[0])
    elif pixel_height >= CALIB_PIXELS[-1]:
        k = (CALIB_DISTANCES[-1] - CALIB_DISTANCES[-2]) / (CALIB_PIXELS[-1] - CALIB_PIXELS[-2])
        return CALIB_DISTANCES[-1] + k * (pixel_height - CALIB_PIXELS[-1])
    else:
        return float(np.interp(pixel_height, CALIB_PIXELS, CALIB_DISTANCES))

def get_compensation(distance_cm):
    """���ݲ�þ���ѡ���Ӧ�εĲ���ϵ��"""
    for (d_min, d_max), tbl in COMPENSATION_TABLE.items():
        if d_min <= distance_cm < d_max:
            return tbl
    # ������Χʱ�������
    return COMPENSATION_TABLE[max(COMPENSATION_TABLE.keys(), key=lambda r: r[0])]

def read_button():
    # �������ĳ�������豸��ȡ��ť״̬���� GPIO��
    # ʾ�������ذ�ť��ţ�1~4��
    return int(sys.argv[1])  # ��ȡ�����в����������㴫�밴ť��ţ�

def get_outer_frame_params(approx_outer):
    """��ȡ���ؼ����������ұ�������������������"""
    pts = approx_outer.reshape(4, 2).astype(np.float32)

    # ��x���������������ұ�
    x_sorted = pts[np.argsort(pts[:, 0])]
    left_pts = x_sorted[:2]  # �������
    right_pts = x_sorted[2:]  # �ұ�����

    # �������ұ�����������ֱ���򳤶ȣ�
    left_pixels = distance(left_pts[0], left_pts[1])
    right_pixels = distance(right_pts[0], right_pts[1])

    # ��y���������������±�
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_pts = y_sorted[:2]  # �ϱ�����
    bottom_pts = y_sorted[2:]  # �±�����

    # �������±��е㣨����˵㣩
    top_mid = ((top_pts[0][0] + top_pts[1][0]) / 2, (top_pts[0][1] + top_pts[1][1]) / 2)
    bottom_mid = ((bottom_pts[0][0] + bottom_pts[1][0]) / 2, (bottom_pts[0][1] + bottom_pts[1][1]) / 2)

    # ���������������������е����ߣ�
    axis_pixels = distance(top_mid, bottom_mid)

    return left_pixels, right_pixels, axis_pixels

# ����������洢��������Ϣ
@dataclass
class SquareInfo:
    vertices: List[Tuple[int, int]]  # �ĸ��������� (x,y)
    center: Tuple[int, int]  # ������������
    avg_side_length: float  # ƽ���߳������أ�
    detection_method: str  # ��ⷽ����"perfect_edge" �� "orphan_corner"
    is_valid: bool = True  # �Ƿ�Ϊ��Ч������

    def __post_init__(self):
        # ���㼸�����ģ����δָ����
        if not self.center and self.vertices:
            x_coords = [v[0] for v in self.vertices]
            y_coords = [v[1] for v in self.vertices]

            # ȷ�� sum() �õ��Ľ���ǵ�һ��ֵ���������ʹ���
            self.center = (int(sum(x_coords) / len(x_coords)), int(sum(y_coords) / len(y_coords)))


# ȫ���б�洢���м�⵽��������
all_squares: List[SquareInfo] = []


def reset_square_collection():
    """�����������ռ��б����ڴ�����֡"""
    global all_squares
    all_squares = []


def add_square_to_collection(vertices, avg_side, method, center=None):
    """����⵽����������ӵ��ռ��б�"""
    square = SquareInfo(
        vertices=vertices,
        avg_side_length=avg_side,
        detection_method=method,
        center=center
    )
    all_squares.append(square)
    return square


def get_all_squares() -> List[SquareInfo]:
    """��ȡ�����ռ�������������Ϣ"""
    return all_squares


def angle_between_vectors(v1, v2):
    """������������֮��ļнǣ��ȣ�"""
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    norm_v1 = (v1[0] ** 2 + v1[1] ** 2) ** 0.5
    norm_v2 = (v2[0] ** 2 + v2[1] ** 2) ** 0.5

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)  # ���Ʒ�Χ����������
    return abs(np.degrees(np.arccos(cos_theta)))


def sort_square_vertices(p1, p2, p3, p4):
    """���ĸ��㰴˳ʱ��˳������ȷ������˳����ȷ"""
    # �������ĵ�
    center = np.mean([p1, p2, p3, p4], axis=0)

    # ����ÿ������������ĵ�ĽǶ�
    def get_angle(point):
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return np.arctan2(dy, dx)  # �Ƕȷ�Χ��-�� �� ��

    # ���Ƕ�����˳ʱ�룩
    points = [p1, p2, p3, p4]
    points.sort(key=lambda p: get_angle(p), reverse=True)  # reverse=True תΪ˳ʱ��
    return points


def detect_right_angles(contour, img, return_details=False):
    """ʶ�𲢱��ͼ���е�ֱ�ǵ㣨85-95�ȣ�������ֱ�ǵ㼰���ϸ��"""
    # 1. �����ƽ����������ඥ�㣬ȷ����ǲ����ϲ���
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.003 * perimeter  # ����ϸ�ıƽ����������ඥ��
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = approx.reshape(-1, 2)  # ת��Ϊ�����б�
    num_vertices = len(vertices)

    if num_vertices < 3:
        return img, [] if not return_details else (img, [])

    right_angles = []

    # 2. �������ж�����ֱ��
    for i in range(num_vertices):
        # ��ȡ��ǰ���㼰���ڶ���
        p_curr = vertices[i]
        p_prev = vertices[(i - 1) % num_vertices]  # ǰһ������
        p_next = vertices[(i + 1) % num_vertices]  # ��һ������

        # ���������ͼн�
        v_prev, v_next = calculate_vectors(p_prev, p_curr, p_next)
        angle = angle_between_vectors(v_prev, v_next)

        # 3. �ж��Ƿ�Ϊֱ�ǣ�85-95�ȣ�
        if 82 <= angle <= 97:
            # �������ߵ�б��
            slope_prev = calculate_slope(p_prev, p_curr)
            slope_next = calculate_slope(p_curr, p_next)

            right_angles.append({
                'point': p_curr,
                'index': i,
                'angle': angle,
                'slopes': [slope_prev, slope_next],
                'neighbors': [p_prev, p_next]
            })

            if not return_details:  # ���ڲ���Ҫ��ϸ��Ϣʱ����
                # ���ֱ�ǵ㣨��ɫԲ�㣩
                cv2.circle(img, tuple(p_curr), 8, (0, 0, 255), -1)
                # ��ʾ�Ƕ�ֵ
                cv2.putText(img, f"{angle:.1f}��",
                            (p_curr[0] + 10, p_curr[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # ��ʾ��⵽��ֱ������
    if not return_details:
        cv2.putText(img, f"ֱ�ǵ�: {len(right_angles)}", (20, 30),
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
    """�Ľ��ĺ�ɫ�����⣺���ݱߵĳ��ȶ�̬��������������"""
    # ����ߵĳ��Ⱥͷ�������
    curr_dx = p2[0] - p1[0]
    curr_dy = p2[1] - p1[1]
    edge_length = np.hypot(curr_dx, curr_dy)  # �ߵ�ʵ�ʳ���

    if edge_length < 1e-6:  # ���������
        return False, 0.0

    # ���ݱߵĳ���ȷ�������������ͷֲ�
    sample_points = []
    if edge_length < 50:  # ���̱�
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        sample_points.append(mid)
        required_passed = 1
    elif 50 <= edge_length < 100:  # �̱�
        for i in [1, 2]:
            x = p1[0] + int((curr_dx * i) / 3)
            y = p1[1] + int((curr_dy * i) / 3)
            sample_points.append((x, y))
        required_passed = 1
    else:  # ����
        for i in [1, 2, 3, 4, 5]:
            x = p1[0] + int((curr_dx * i) / 6)
            y = p1[1] + int((curr_dy * i) / 6)
            sample_points.append((x, y))
        required_passed = 3

    # ���㴹ֱ����λ����
    raw_normal_dx = -curr_dy  # ��ֱ����x����
    raw_normal_dy = curr_dx  # ��ֱ����y����

    # ��һ����ֱ������ȷ������Ϊ1��
    normal_len = np.hypot(raw_normal_dx, raw_normal_dy)
    unit_normal_dx = raw_normal_dx / normal_len
    unit_normal_dy = raw_normal_dy / normal_len

    # Ӧ�÷���prev_side��������/���£�
    final_normal_dx = unit_normal_dx * prev_side
    final_normal_dy = unit_normal_dy * prev_side

    # �洢���в�����ļ����
    all_ratios = []
    passed_count = 0

    # ��ÿ����������м��
    for (x, y) in sample_points:
        # ����ƫ�Ƽ���
        test_x = int(x + final_normal_dx * step_pixels)
        test_y = int(y + final_normal_dy * step_pixels)
        test_point = (test_x, test_y)

        # ���Ʋ������Ǻ�ƫ���߶Σ����ڵ��ԣ�
        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)  # ��ɫ��ԭ������
        cv2.circle(img, test_point, 3, (0, 128, 255), -1)  # ��ɫ��ƫ�ƺ����
        cv2.line(img, (x, y), test_point, (0, 255, 0), 1)  # ��ɫ��ƫ���߶�

        # ����õ���Χ�ĺ�ɫ����
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
    """�ϸ������ڱ߷����Ʋ������ζ��㣬������"""
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
    """Ѱ�����Ʋ����ƥ���ԭʼ�ڵ�"""
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
    """�����������ı߳��Ȳ�����ƽ��ֵ��ȷ����ֵ��Ч"""
    try:
        s1 = distance(p1, p2)
        s2 = distance(p2, p3)
        s3 = distance(p3, p4)
        s4 = distance(p4, p1)
        avg = (s1 + s2 + s3 + s4) / 4
        # ȷ�����غ������ֵ
        return round(avg, 1) if avg > 0 else 0.0
    except Exception as e:
        print(f"����߳�ʱ����: {e}")
        return 0.0


def is_edge_part_of_existing_square(edge, square_groups, threshold=30):
    """�����Ƿ���������ʶ���������"""
    p1, p2 = edge
    for square in square_groups:
        p1_in = any(np.hypot(p1[0] - sp[0], p1[1] - sp[1]) <= threshold for sp in square)
        p2_in = any(np.hypot(p2[0] - sp[0], p2[1] - sp[1]) <= threshold for sp in square)
        if p1_in and p2_in:
            return True
    return False


def detect_perfect_edges_and_squares(contour, img, step_pixels=30):
    # ������������
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.003 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    original_vertices = approx.reshape(-1, 2)
    num_original = len(original_vertices)
    all_vertices = original_vertices.copy().tolist()
    processed_edges = set()  # ���ڱ����ظ������
    square_groups = []

    # ��ʼ״̬����
    vertex_status = {i: "δ����" for i in range(num_original)}
    edge_status = {}

    # ����ԭʼ����
    cv2.drawContours(img, [approx], -1, (147, 112, 219), 2)
    for i in range(num_original):
        cv2.putText(img, f"{i + 1}",
                    (original_vertices[i][0] + 5, original_vertices[i][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(img, (original_vertices[i][0], original_vertices[i][1]), 5, (0, 255, 0), -1)

    # ���������
    perfect_edges = []
    for i in range(num_original):
        p1_idx = i
        p2_idx = (i + 1) % num_original
        p1 = original_vertices[p1_idx]
        p2 = original_vertices[p2_idx]
        edge_name = f"l{p1_idx + 1}l{p2_idx + 1}"

        p0 = original_vertices[(i - 1) % num_original]
        p3 = original_vertices[(i + 2) % num_original]

        # �Ƕȼ��
        prev_len = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        next_len = np.hypot(p3[0] - p2[0], p3[1] - p2[1])
        s_prev = calculate_slope(p0, p1)
        s_curr = calculate_slope(p1, p2)
        s_next = calculate_slope(p2, p3)
        angle_p1 = angle_between_slopes(s_prev, s_curr)
        angle_p2 = angle_between_slopes(s_curr, s_next)

        # ���̱߷ſ���ֵ
        is_right_p1 = 80 <= angle_p1 <= 100 if prev_len < 100 else 85 <= angle_p1 <= 95
        is_right_p2 = 80 <= angle_p2 <= 100 if next_len < 100 else 85 <= angle_p2 <= 95

        # ������
        dir_consistent, direction, prev_side = check_direction_consistency(p0, p1, p2, p3)

        # ��ɫ������
        black_valid = False
        if is_right_p1 and is_right_p2 and dir_consistent:
            black_valid, _ = check_black_region(
                img, p1, p2, prev_side,
                step_pixels=step_pixels,
                threshold=0.70
            )

        # ���������
        if is_right_p1 and is_right_p2 and dir_consistent and black_valid:
            perfect_edges.append((p1_idx, p2_idx, p1, p2, direction, prev_side))
            edge_status[edge_name] = "�Ѻ���"
            vertex_status[p1_idx] = "�Ѻ���"
            vertex_status[p2_idx] = "�Ѻ���"
            cv2.line(img, tuple(p1), tuple(p2), (255, 0, 0), 3)
        else:
            edge_status[edge_name] = "δ����"
            cv2.line(img, tuple(p1), tuple(p2), (0, 0, 255), 3)

        # ������������ʾλ��
        edge_mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2 - 10)  # ����ƫ�Ʊ����ص�
        cv2.putText(img, edge_name, edge_mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # �����������Ʋ�������
    for edge in perfect_edges:
        p1_idx, p2_idx, p1, p2, direction, prev_side = edge
        edge_key = frozenset([p1_idx, p2_idx])

        # �����Ѵ���ı߻������������εı�
        if edge_key in processed_edges or is_edge_part_of_existing_square((p1, p2), square_groups):
            processed_edges.add(edge_key)
            continue

        # �Ʋ������ζ��㣨��ʹ��prev_side��
        p3, p4 = infer_square_points(p1, p2, prev_side)
        if p3 is None or p4 is None:
            continue

        # ƥ�����нڵ�
        match3_idx, match3 = find_matching_node(p3, all_vertices)
        match4_idx, match4 = find_matching_node(p4, all_vertices)

        # ��ʼ��ԭʼ�ڵ㼯��
        original_matched_indices = set([p1_idx, p2_idx])

        # ����p3
        if match3_idx is not None and match3_idx < num_original:
            p3_final = match3
            original_matched_indices.add(match3_idx)
            vertex_status[match3_idx] = "�Ѻ���"
        else:
            all_vertices.append(p3)
            p3_final = p3
            cv2.circle(img, p3_final, 5, (0, 255, 255), -1)

        # ����p4
        if match4_idx is not None and match4_idx < num_original:
            p4_final = match4
            original_matched_indices.add(match4_idx)
            vertex_status[match4_idx] = "�Ѻ���"
        else:
            all_vertices.append(p4)
            p4_final = p4
            cv2.circle(img, p4_final, 5, (0, 255, 255), -1)

        # ���ĸ��������˳ʱ������
        sorted_points = sort_square_vertices(p1, p2, p3_final, p4_final)
        s1, s2, s3, s4 = sorted_points

        # ����������
        square = (s1, s2, s3, s4)
        square_groups.append(square)
        cv2.line(img, tuple(s1), tuple(s2), (255, 255, 0), 2)
        cv2.line(img, tuple(s2), tuple(s3), (255, 255, 0), 2)
        cv2.line(img, tuple(s3), tuple(s4), (255, 255, 0), 2)
        cv2.line(img, tuple(s4), tuple(s1), (255, 255, 0), 2)

        # ��ʾƽ���߳�
        avg_len = get_square_side_lengths(*square)
        add_square_to_collection(vertices=[(s1[0], s1[1]), (s2[0], s2[1]), (s3[0], s3[1]), (s4[0], s4[1])],
                                 avg_side=avg_len, method="perfect_edge")
        mid_square = ((s1[0] + s3[0]) // 2, (s1[1] + s3[1]) // 2)
        cv2.putText(img, f"= {avg_len:.1f}", mid_square,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # ���¶���ͱ�״̬
        for idx in original_matched_indices:
            prev_idx = (idx - 1) % num_original
            next_idx = (idx + 1) % num_original
            vertex_status[prev_idx] = "�Ѻ���"
            vertex_status[next_idx] = "�Ѻ���"

        # �����߶�״̬
        for i in range(num_original):
            p1_idx = i
            p2_idx = (i + 1) % num_original
            edge_name = f"l{p1_idx + 1}l{p2_idx + 1}"

            if vertex_status[p1_idx] == "�Ѻ���" and vertex_status[p2_idx] == "�Ѻ���":
                edge_status[edge_name] = "�Ѻ���"
                cv2.line(img,
                         tuple(original_vertices[p1_idx]),
                         tuple(original_vertices[p2_idx]),
                         (255, 0, 0), 3)

        # ������������б�Ϊ�Ѵ���
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

    # ����δ����ڵ㣨��ԭʼ�ڵ㣩
    unverified_vertices = [i for i, status in vertex_status.items() if status == "δ����" and i < num_original]

    # ��ʾδ����ڵ�
    if unverified_vertices:
        cv2.putText(img, "δ����: " + ",".join(map(str, [i + 1 for i in unverified_vertices])),
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return img, vertex_status, edge_status, square_groups, original_vertices


# === �䵥�Ƿ�ʵ�� ===
def extend_line(p1, p2, length_ratio=2.0):  # �ӳ��߳��ȵ���Ϊ2��
    """�ӳ��߶Σ������ӳ�����Ķ˵�"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    extended_p = (
        int(p2[0] + dx * length_ratio),
        int(p2[1] + dy * length_ratio)
    )
    return extended_p


def line_intersection(p1, p2, p3, p4):
    """���������߶�(p1-p2��p3-p4)�Ľ���"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None  # ƽ���ߣ��޽���

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
    """���ָ������Χָ���뾶�ں�ɫ���صı�������ӱ߽���"""
    h, w = img.shape[:2]
    x, y = point

    # ���ȼ����Ƿ���ͼ��Χ��
    if x < 0 or x >= w or y < 0 or y >= h:
        print(f"���棺��({x},{y})����ͼ��Χ")
        return 0.0  # ������Χ��Ϊ����������

    # ��������߽磨ȷ����ͼ��Χ�ڣ�
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)

    # ��������Ƿ���Ч��������1x1���أ�
    if x1 >= x2 or y1 >= y2:
        print(f"���棺��Ч���� ({x1},{y1})-({x2},{y2})")
        return 0.0

    # ��ȡ����ת��Ϊ�Ҷ�
    region = img[y1:y2, x1:x2]
    if region.size == 0:
        print(f"���棺������ ({x1},{y1})-({x2},{y2})")
        return 0.0

    if len(region.shape) == 3:
        try:
            region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            print(f"ת����ɫ�ռ����: {e}")
            return 0.0

    # ��ֵ������ɫ��Ϊ0��������Ϊ255��
    _, binary = cv2.threshold(region, 100, 255, cv2.THRESH_BINARY)

    # �����ɫ���ر���
    total = binary.size
    if total == 0:
        return 0.0
    black = total - cv2.countNonZero(binary)
    return black / total


def check_slope_match(slopes1, slopes2):
    """�������ֱ�ǵ�б���Ƿ�ƥ�䣨k1,k2 �� k2,k1��"""
    if len(slopes1) != 2 or len(slopes2) != 2:
        return False

    # ������ֿ��ܵ�ƥ�䷽ʽ
    match1 = (angle_between_slopes(slopes1[0], slopes2[1]) < ORPHAN_ANGLE_THRESHOLD and
              angle_between_slopes(slopes1[1], slopes2[0]) < ORPHAN_ANGLE_THRESHOLD)

    match2 = (angle_between_slopes(slopes1[0], slopes2[0]) < ORPHAN_ANGLE_THRESHOLD and
              angle_between_slopes(slopes1[1], slopes2[1]) < ORPHAN_ANGLE_THRESHOLD)

    return match1 or match2


def orphan_corner_method(img, contour, unverified_indices, original_vertices, square_groups):
    """�䵥�Ƿ����Ż�ͬ�ߵ㳡���жϣ����Ӻ�ɫ������֤"""
    # ��ȡ����ֱ�ǵ����ϸ��Ϣ
    _, all_right_angles = detect_right_angles(contour, img, return_details=True)

    # ɸѡ��δ�����ֱ�ǵ�
    orphan_corners = [
        corner for corner in all_right_angles
        if corner['index'] in unverified_indices
    ]

    # ����������䵥������
    print(f"\n===== �䵥�Ƿ���ʼ =====")
    print(f"��⵽δ������䵥������: {len(orphan_corners)}")
    for i, corner in enumerate(orphan_corners):
        print(f"�䵥�� {i + 1}: ����={corner['point']}, б��={corner['slopes']}")

    # ������Ҫ�����䵥�ǲ��ܼ���
    if len(orphan_corners) < 2:
        print("�䵥����������2�����޷�����ƥ��")
        print("===== �䵥�Ƿ����� =====")
        return img, square_groups

    cv2.putText(img, f"�䵥������: {len(orphan_corners)}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # ��������䵥�����
    for i in range(len(orphan_corners)):
        for j in range(i + 1, len(orphan_corners)):
            corner1 = orphan_corners[i]
            corner2 = orphan_corners[j]
            print(f"\n----- ����䵥�����: ��{i + 1}���{j + 1} -----")

            # ���б���Ƿ�ƥ��
            slope_match = check_slope_match(corner1['slopes'], corner2['slopes'])
            print(f"б��ƥ����: {slope_match}")
            print(f"��{i + 1}б��: {corner1['slopes']}, ��{j + 1}б��: {corner2['slopes']}")

            if not slope_match:
                print("б�ʲ�ƥ�䣬���������")
                continue

            # ���ƥ����䵥��
            cv2.circle(img, tuple(corner1['point']), 10, (255, 0, 255), 2)
            cv2.circle(img, tuple(corner2['point']), 10, (255, 0, 255), 2)
            cv2.line(img, tuple(corner1['point']), tuple(corner2['point']), (255, 0, 255), 1)

            p1 = corner1['point']
            p1a, p1b = corner1['neighbors']
            p2 = corner2['point']
            p2a, p2b = corner2['neighbors']

            # �����1�����ߵ�б�ʼ��ӳ���
            slope1a = calculate_slope(p1, p1a)
            slope1b = calculate_slope(p1, p1b)
            ext_p1a = extend_line(p1, p1a, length_ratio=3.0)
            ext_p1b = extend_line(p1, p1b, length_ratio=3.0)

            # �����2�����ߵ�б�ʼ��ӳ���
            slope2a = calculate_slope(p2, p2a)
            slope2b = calculate_slope(p2, p2b)
            ext_p2a = extend_line(p2, p2a, length_ratio=3.0)
            ext_p2b = extend_line(p2, p2b, length_ratio=3.0)

            # ���������������б��
            print(f"��1����б��: {slope1a} (�ӳ���{ext_p1a}), {slope1b} (�ӳ���{ext_p1b})")
            print(f"��2����б��: {slope2a} (�ӳ���{ext_p2a}), {slope2b} (�ӳ���{ext_p2b})")

            # ���������ӳ��ߣ���ͬ��ɫ�������֣�
            cv2.line(img, tuple(p1), ext_p1a, (0, 165, 255), 1, cv2.LINE_AA)  # ��ɫ
            cv2.line(img, tuple(p1), ext_p1b, (255, 165, 0), 1, cv2.LINE_AA)  # ��ɫ��ɫ
            cv2.line(img, tuple(p2), ext_p2a, (0, 165, 255), 1, cv2.LINE_AA)  # ��ɫ
            cv2.line(img, tuple(p2), ext_p2b, (255, 165, 0), 1, cv2.LINE_AA)  # ��ɫ��ɫ

            # ��б�����������������
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

            # ���㽻��
            intersect1 = line_intersection(neg_slope_ray1[0], neg_slope_ray1[1],
                                           pos_slope_ray2[0], pos_slope_ray2[1])
            intersect2 = line_intersection(pos_slope_ray1[0], pos_slope_ray1[1],
                                           neg_slope_ray2[0], neg_slope_ray2[1])

            # ���������������Ϣ
            print(f"��б����������б�����߽���1: {intersect1}")
            print(f"��б�������븺б�����߽���2: {intersect2}")

            # �ռ���Ч����
            valid_intersections = []
            if intersect1:
                valid_intersections.append(intersect1)
                cv2.circle(img, intersect1, 6, (0, 255, 0), -1)  # ��ɫ�����Ч����
            if intersect2:
                valid_intersections.append(intersect2)
                cv2.circle(img, intersect2, 6, (0, 255, 0), -1)  # ��ɫ��Ǳ����Ч����

            # ���1���Խǵ㳡�� - ��������Ч����
            if len(valid_intersections) >= 2:
                print(f"��⵽{len(valid_intersections)}����Ч���㣬����Խǵ㳡���ж�")

                # ȡǰ������Ч����
                intersect1, intersect2 = valid_intersections[:2]

                # ���������ߵĳ���
                s1 = distance(p1, intersect1)
                s2 = distance(intersect1, p2)
                s3 = distance(p2, intersect2)
                s4 = distance(intersect2, p1)

                # ���㳤�Ȳ���
                lengths = [s1, s2, s3, s4]
                max_len = max(lengths)
                min_len = min(lengths)
                length_diff = (max_len - min_len) / max_len if max_len > 0 else 0

                if length_diff <= SIDE_LENGTH_TOLERANCE:
                    print("�߳����������̷�Χ�ڣ��ж�Ϊ�Խǵ㹹�ɵ�������")
                    square_points = [p1, intersect1, p2, intersect2]
                    sorted_points = sort_square_vertices(*square_points)

                    # ����ƽ���߳���ȷ���õ���Ч��ֵ��
                    avg_length = get_square_side_lengths(*sorted_points)
                    # add_square_to_collection(vertices=[(p1, p2, selected_p3, selected_p4)], avg_side=avg_length,
                    #                          method="orphan_corner")
                    # ����������
                    cv2.line(img, tuple(sorted_points[0]), tuple(sorted_points[1]), (128, 0, 128), 2)
                    cv2.line(img, tuple(sorted_points[1]), tuple(sorted_points[2]), (128, 0, 128), 2)
                    cv2.line(img, tuple(sorted_points[2]), tuple(sorted_points[3]), (128, 0, 128), 2)
                    cv2.line(img, tuple(sorted_points[3]), tuple(sorted_points[0]), (128, 0, 128), 2)

                    # # ����ı����������ʾ "=��ֵ"���������߸�ʽһ��
                    mid_point = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                    if 0 <= mid_point[0] < img.shape[1] and 0 <= mid_point[1] < img.shape[0]:
                        # ֻ��ʾ�Ⱥź���ֵ���������������
                        text = f"= {avg_length:.1f}"
                        cv2.putText(img, text, mid_point,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)  # ʹ������������ͬ����ɫ�ʹ�С

                    add_square_to_collection(vertices=sorted_points, avg_side=avg_length, method="orphan_corner")
                    continue
            # ���2��ͬ�ߵ㳡�� - �Ż��汾�����Ӻ�ɫ������֤
            print("����ͬ�ߵ㳡���ж�")
            # �������������Ϊ�߳�
            side_length = distance(p1, p2)
            print(f"�����������(��Ϊ�߳�): {side_length:.1f}����")

            # ���㴹ֱ���������������ܵķ���
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm = np.sqrt(dx ** 2 + dy ** 2) if (dx ** 2 + dy ** 2) > 0 else 1

            # ����������ܵĶ��㣨������ͬ����
            perp_dx1 = -dy / norm
            perp_dy1 = dx / norm
            p3_1 = (int(p1[0] + perp_dx1 * side_length), int(p1[1] + perp_dy1 * side_length))
            p4_1 = (int(p2[0] + perp_dx1 * side_length), int(p2[1] + perp_dy1 * side_length))

            perp_dx2 = dy / norm
            perp_dy2 = -dx / norm
            p3_2 = (int(p1[0] + perp_dx2 * side_length), int(p1[1] + perp_dy2 * side_length))
            p4_2 = (int(p2[0] + perp_dx2 * side_length), int(p2[1] + perp_dy2 * side_length))

            # �������������ܵĶ���
            print(f"��һ���Ʋⶥ��: {p3_1}, {p4_1}")
            print(f"�ڶ����Ʋⶥ��: {p3_2}, {p4_2}")

            # ������鶥��ĺ�ɫ�������
            black_thresh = 0.3  # 30%��ɫ������ֵ
            p3_1_black = check_black_pixel_percentage(img, p3_1)
            p4_1_black = check_black_pixel_percentage(img, p4_1)
            p3_2_black = check_black_pixel_percentage(img, p3_2)
            p4_2_black = check_black_pixel_percentage(img, p4_2)

            print(f"��һ���ɫ����: p3={p3_1_black:.2%}, p4={p4_1_black:.2%}")
            print(f"�ڶ����ɫ����: p3={p3_2_black:.2%}, p4={p4_2_black:.2%}")

            # �ж����鶥����������������㶼������ֵ��
            group1_valid = p3_1_black >= black_thresh and p4_1_black >= black_thresh
            group2_valid = p3_2_black >= black_thresh and p4_2_black >= black_thresh

            # ѡ����Ч�Ķ�����
            if group1_valid and group2_valid:
                # ����Чʱѡ��ɫ�������ߵ���
                group1_avg = (p3_1_black + p4_1_black) / 2
                group2_avg = (p3_2_black + p4_2_black) / 2
                selected_p3, selected_p4 = (p3_1, p4_1) if group1_avg > group2_avg else (p3_2, p4_2)
                print(f"���鶼��Ч��ѡ���ɫ�������ߵ���")
            elif group1_valid:
                selected_p3, selected_p4 = p3_1, p4_1
                print(f"ѡ���һ�鶥��")
            elif group2_valid:
                selected_p3, selected_p4 = p3_2, p4_2
                print(f"ѡ��ڶ��鶥��")
            else:
                # ����Чʱѡ��ɫ�����ϸߵ���
                group1_avg = (p3_1_black + p4_1_black) / 2
                group2_avg = (p3_2_black + p4_2_black) / 2
                selected_p3, selected_p4 = (p3_1, p4_1) if group1_avg > group2_avg else (p3_2, p4_2)
                print(f"���鶼������������ѡ����ԽϺõ���")

            # ���������α߳����ı�ƽ��ֵ��
            square_points = [p1, p2, selected_p4, selected_p3]
            try:
                sorted_points = sort_square_vertices(*square_points)
                avg_length = get_square_side_lengths(*sorted_points)
            except Exception as e:
                print(f"���򶥵�����߳�ʧ��: {e}")
                avg_length = round(side_length, 1)  # ʹ�ñ�ѡֵ

            # ����������
            cv2.line(img, tuple(sorted_points[0]), tuple(sorted_points[1]), (128, 0, 128), 2)
            cv2.line(img, tuple(sorted_points[1]), tuple(sorted_points[2]), (128, 0, 128), 2)
            cv2.line(img, tuple(sorted_points[2]), tuple(sorted_points[3]), (128, 0, 128), 2)
            cv2.line(img, tuple(sorted_points[3]), tuple(sorted_points[0]), (128, 0, 128), 2)

            # ����ı����������ʾ "=��ֵ"���������߸�ʽһ��
            mid_point = ((p1[0] + selected_p3[0]) // 2, (p1[1] + selected_p3[1]) // 2)
            if 0 <= mid_point[0] < img.shape[1] and 0 <= mid_point[1] < img.shape[0]:
                # ֻ��ʾ�Ⱥź���ֵ���������������
                text = f"= {avg_length:.1f}"
                cv2.putText(img, text, mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)  # ʹ������������ͬ����ɫ�ʹ�С
            else:
                print(f"�ı�λ�ó���ͼ��Χ: {mid_point}")

            # # �����֤���
            # cv2.circle(img, selected_p3, 5, (0, 255, 0) if group1_valid else (0, 165, 255), -1)
            # cv2.circle(img, selected_p4, 5, (0, 255, 0) if group1_valid else (0, 165, 255), -1)

            square_groups.append(sorted_points)
            print(f"�Ѵ���ͬ�ߵ㳡���������Σ�ƽ���߳�: {avg_length:.1f}����")
            add_square_to_collection(vertices=sorted_points, avg_side=avg_length, method="orphan_corner")
    print("\n===== �䵥�Ƿ����� =====")
    return img, square_groups


def process_frame(frame, button_id: int):
    """����֡ͼ��ĺ��ĺ��������������߷����䵥�Ƿ���"""
    global outer_axis_pixels,outer_vertical_pixels,outer_height_history,current_mode
    global ppcm,outer_height ,distance_cm, input_num
    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # �����
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    a4_outer = contours[0]
    cv2.drawContours(img, [a4_outer], -1, (0, 255, 0), 3)

    # ����������
    peri_outer = cv2.arcLength(a4_outer, True)
    approx_outer = cv2.approxPolyDP(a4_outer, 0.02 * peri_outer, True)
    rect = cv2.minAreaRect(a4_outer)
    rect_w, rect_h = rect[1]
    print(rect[1])
    ppcm = ((rect_w / 21.0) + (rect_h / 29.7)) / 2  # ����/���ױ���������A4ֽ�ߴ磩

    outer_height = 0.0
    if len(approx_outer) == 4:
        # ��ȡ������
        left_pix, right_pix, axis_pix = get_outer_frame_params(approx_outer)
        outer_vertical_pixels.append((left_pix, right_pix))
        outer_axis_pixels.append(axis_pix)

        # ������ʷ���ݳ���
        if len(outer_vertical_pixels) > HISTORY_LEN:
            outer_vertical_pixels.pop(0)
        if len(outer_axis_pixels) > HISTORY_LEN:
            outer_axis_pixels.pop(0)


        current_height = (left_pix + right_pix) / 2

        # ������ʷ������
        outer_height_history.append(current_height)
        if len(outer_height_history) > HISTORY_LEN:
            outer_height_history.pop(0)
        outer_height, _ = calculate_filtered_average(outer_height_history)



    # === ������������벢��ʾ ===
    distance_cm = 0.0
    if outer_height > 0:
        distance_cm = estimate_distance(outer_height)
        # ѡȡ�����õĲ���ϵ��
        comp = get_compensation(distance_cm)
        NORMAL_COMPENSATION = comp['normal']

        # ��ͼ����ʾ����
        cv2.putText(frame, f"Dist: {distance_cm:.1f}cm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        # ��ӡ������Ϣ
        #print(f"���߶�: {outer_height:.1f}���� -> �������: {distance_cm:.1f}cm")

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

    # �����ڿ���ʵ�ڿ��ģ���ڿ�
    if a4_inner is not None:
        # ������ʵ�ڿ򣨺�ɫ��
        cv2.drawContours(img, [a4_inner], -1, (0, 0, 255), 3)
        inner_contour = a4_inner
    else:
        # �ڿ�ʶ��ʧ�ܣ�����ģ���ڿ򣨻�ɫ��
        print("�ڿ�ʶ��ʧ�ܣ�ʹ��ģ���ڿ�")

        # ��ȡ�����ת���β���
        outer_rect = cv2.minAreaRect(a4_outer)
        outer_center, outer_size, outer_angle = outer_rect
        outer_w, outer_h = outer_size

        # ȷ����򳤱ߺͶ̱�
        if outer_w > outer_h:
            outer_long_side = outer_w
            outer_short_side = outer_h
            is_rotated = False
        else:
            outer_long_side = outer_h
            outer_short_side = outer_w
            is_rotated = True

        # �����ڿ�ߴ�
        inner_long_side = outer_long_side * 0.865
        inner_short_side = outer_short_side * 0.762

        # �����ڿ������ͬ����
        inner_size = (inner_short_side, inner_long_side) if is_rotated else (inner_long_side, inner_short_side)

        # �����ڿ���ת����
        inner_rect = (outer_center, inner_size, outer_angle)
        inner_pts = cv2.boxPoints(inner_rect)
        inner_pts = np.int64(inner_pts)

        # ����ģ���ڿ�
        cv2.drawContours(img, [inner_pts], -1, (0, 255, 255), 3)
        cv2.putText(img, "ģ���ڿ�", (int(outer_center[0] - 50), int(outer_center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ����ģ���ڿ������
        inner_contour = inner_pts.reshape(-1, 1, 2).astype(np.int32)

    # �ڲ�������
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
        # �ſ�����ɸѡ����
        inner_area = cv2.contourArea(inner_contour)
        min_valid_area = max(20, inner_area * 0.001)

        filtered = [c for c in candidates if cv2.contourArea(c) >= min_valid_area]
        if len(filtered) < 1:
            sorted_candidates = sorted(candidates, key=cv2.contourArea, reverse=True)
            filtered = sorted_candidates[:3]

        # ����ÿ����ѡ����
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

            # 1. ���������߷����
            img, vertex_status, edge_status, square_groups, original_vertices = detect_perfect_edges_and_squares(approx,
                                                                                                                 img,
                                                                                                                 step_pixels=9)

            # ��ȡδ����Ķ�������
            unverified_indices = [i for i, status in vertex_status.items() if status == "δ����"]

            # 2. �����δ����Ķ��㣬�����䵥�Ƿ�
            if unverified_indices:
                img, square_groups = orphan_corner_method(img, approx, unverified_indices, original_vertices,
                                                          square_groups)

            # ���������
            print("=== ����״̬ ===")
            for idx, status in vertex_status.items():
                print(f"����{idx + 1}: {status}")

            print("\n=== �߶�״̬ ===")
            for edge, status in edge_status.items():
                print(f"{edge}: {status}")

            if area < inner_area * 0.02:
                cv2.putText(img, "СĿ��", (get_contour_center(approx)[0] + 10, get_contour_center(approx)[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    save_squares_to_file()
    
    if all_squares:
        # 1) �Ӵ�С����
        sorted_sqs = sorted(all_squares, key=lambda s: s.avg_side_length, reverse=True)

        # 2) ��ť���ӳ�䵽������0-based��������ʱָ�����һ��
        idx = min(button_id - 1, len(sorted_sqs) - 1)
        chosen = sorted_sqs[idx]

        # 3) ��ӡ��ѡ��������Ϣ
        real_length = chosen.avg_side_length / ppcm
        print(f"ppcmֵΪ:{ppcm}")
        print(f"���߶�: {outer_height:.1f}���� -> �������: {distance_cm:.1f}cm")
        # print(f"��ť {button_id} �� ���� {idx+1} ��������:")
        print(f"  ���{input_num}��Ӧ��������ƽ���߳�: {chosen.avg_side_length:.1f} ���أ���Ӧʵ�ʳߴ�: {real_length:.1f} cm")

    return img


if __name__ == "__main__":
    # 1. ���������������в������ȴ����ع�������ٱ���ԭ��sys.argv�߼���
    import argparse
    parser = argparse.ArgumentParser()
    # ����ع������--gain/--shutter�����봫��
    parser.add_argument('--gain', required=True, help='����ͷ����ֵ')
    parser.add_argument('--shutter', required=True, help='����ͷ����ʱ�䣨΢�룩')
    # �����ع������parse_known_args() ����δ����Ĳ����������ͻ��
    args, remaining_argv = parser.parse_known_args()
    
    # 2. ����ԭ���߼�����ʣ������ж�ȡ button_id �� input_num����Ӧԭsys.argv[1]��sys.argv[2]��
    # ע�⣺remaining_argv ����˵� --gain/--shutter ����ֵ��ʣ�µľ���ԭ�е�ģʽ�ͱ��
    if len(remaining_argv) < 2:
        raise RuntimeError("�봫��ģʽ��button_id���ͱ�ţ�input_num������")
    button_id = int(remaining_argv[0])  # ��Ӧԭ sys.argv[1]
    input_num = remaining_argv[1]       # ��Ӧԭ sys.argv[2]

    # ��ʼ��ģʽ
    current_mode = "normal"

    # ������Ƶ�����ԣ�...

    # 3. ��֡����ģʽ���ý��������ع�����滻Ӳ����ֵ
    if ENABLE_SINGLE_FRAME:
        print("��֡����ģʽ (libcamera-still) - ������...")
        cmd = [
            "rpicam-still",
            "--nopreview",
            "-t", "10",
            "-o", "image.jpg",
            "--width", "2028",
            "--height", "1520",
            "--awbgains", "3.47,1.55",
            "--gain", args.gain,    # �ý������������
            "--shutter", args.shutter,  # �ý����Ŀ��Ų���
        ]
        print(f"ִ���������{' '.join(cmd)}")  # �����ã�ȷ�ϲ�����ȷ
        subprocess.run(cmd, check=True)

        img = cv2.imread("image.jpg")
        if img is None:
            raise RuntimeError("��ȡ image.jpg ʧ��")

        # ԭ���߼����䣨���»�ȡ�� button_id �� input_num��
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        processed = process_frame(rot, button_id)  # ֱ���ý������ button_id
        restored = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        output = cv2.resize(restored, (640, 480))

        cv2.imshow("Result", output)
        print("������� ���� 5 ����Զ��˳����� ESC �˳�")
        key = cv2.waitKey(5000)

        if key & 0xFF == 27:
            print("��⵽ ESC����ǰ�˳�")
        else:
            print("5 �뵽���Զ��˳�")

        cv2.destroyAllWindows()