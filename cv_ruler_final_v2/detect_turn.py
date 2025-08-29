# -*- coding: gbk -*-
# ������б45�ȷ��õ������β������ԶԽ���Ϊ���ᣩ
import cv2
import numpy as np
import time
import math
import subprocess

# === ��������๦�ܲ��� ===
# �������ݣ�(���ظ߶�, ʵ�ʾ���cm)���ɸ���ʵ���������
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

# === ������εĲ���ϵ������λ��cm��===
# ���������б45�������εĲ���ϵ��
COMPENSATION_TABLE = {
    (100, 120): {'normal': 0.994, 'slight': 1.000, 'moderate': 1.007, 'large': 1.005},
    (120, 140): {'normal': 0.995, 'slight': 1.001, 'moderate': 1.002, 'large': 1.003},
    (140, 160): {'normal': 0.995, 'slight': 1.002, 'moderate': 1.004, 'large': 1.005},
    (160, 180): {'normal': 0.995, 'slight': 1.002, 'moderate': 1.007, 'large': 1.000},
    (180, 200): {'normal': 0.997, 'slight': 1.001, 'moderate': 1.004, 'large': 1.006},
    (200, 300): {'normal': 0.995, 'slight': 1.00, 'moderate': 1.004, 'large': 1.005},
}

# �ֶ���ֵ�����ز������- �Ż���֡�ж�������
NORMAL_THRESHOLD = 0.02    # 3% ����Ϊ����
SLIGHT_THRESHOLD = 0.07    # 3-10% ΪС����ת
MODERATE_THRESHOLD = 0.9  # 10-20% Ϊ�з���ת
# >20% Ϊ�����ת

# ��������
OUTLIER_THRESHOLD = 0.2  # �쳣ֵ������ֵ
HISTORY_LEN = 1  # ��֡ģʽ�½�������ǰ֡����


def get_contour_center(contour):
    """�����������ĵ�"""
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return (0, 0)
    return (M['m10'] / M['m00'], M['m01'] / M['m00'])


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


def get_outer_frame_params(approx_outer):
    """��ȡ���ؼ��������ϸ�������¶̱��е������Ϊ��������"""
    pts = approx_outer.reshape(4, 2).astype(np.float32)
    
    # 1. �ϸ��������¶̱ߣ�ˮƽ����
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_short_edge = y_sorted[:2]  # �ϱߣ��̱ߣ�����
    bottom_short_edge = y_sorted[2:]  # �±ߣ��̱ߣ�����
    
    # 2. �������¶̱߸��Ե��е�
    top_mid = np.mean(top_short_edge, axis=0)  # �ϱ��е�
    bottom_mid = np.mean(bottom_short_edge, axis=0)  # �±��е�
    
    # 3. ���㳤�����᣺���¶̱��е�ļ��ξ��루�ϸ��壩
    vertical_axis_pixels = distance(top_mid, bottom_mid)
    
    # 4. �������¶̱ߵĳ���
    top_width = distance(top_short_edge[0], top_short_edge[1])
    bottom_width = distance(bottom_short_edge[0], bottom_short_edge[1])
    
    # 5. �������ҳ��ߵĳ��ȣ�������ת�жϣ�
    x_sorted = pts[np.argsort(pts[:, 0])]
    left_long_edge = x_sorted[:2]  # ��ߣ����ߣ�����
    right_long_edge = x_sorted[2:]  # �ұߣ����ߣ�����
    left_height = distance(left_long_edge[0], left_long_edge[1])
    right_height = distance(right_long_edge[0], right_long_edge[1])
    
    # ���أ����¶̱���Ϣ�����ҳ�����Ϣ���ϸ����ĳ�������
    return (top_width, bottom_width, top_mid, bottom_mid), \
           (left_height, right_height), \
           vertical_axis_pixels


def get_inner_square_diagonal(approx_inner):
    """��ȡ�ڲ������εĶԽ��߳��ȣ���Զ������룩"""
    pts = approx_inner.reshape(-1, 2).astype(np.float32)
    max_dist = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dist = distance(pts[i], pts[j])
            if dist > max_dist:
                max_dist = dist
    return max_dist


def process_frame(frame):
    """����֡ͼ��ĺ��ĺ���"""
    global current_mode  # ��������ǰģʽ����
    NORMAL_COMPENSATION = SLIGHT_ROTATION_COMPENSATION = MODERATE_ROTATION_COMPENSATION = LARGE_ROTATION_COMPENSATION = 1.0

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
    
    # ��ʼ������
    ppcm = None
    vertical_axis_pix = 0.0  # ������������ֵ
    top_mid = None
    bottom_mid = None
    current_left = 0.0  # ��ǰ֡������߳���
    current_right = 0.0  # ��ǰ֡�ұ����߳���

    outer_height = 0.0
    if len(approx_outer) == 4:
        # ��ȡ���������ϸ���㳤������
        (top_width, bottom_width, top_mid, bottom_mid), height_pairs, vertical_axis_pix = get_outer_frame_params(approx_outer)
        
        # ��֡ģʽ����������ǰ֡��������������
        current_left, current_right = height_pairs
        
        # ȷ����ǰ���߶ȣ�ʹ���ϸ����ĳ������ᣩ
        outer_height = vertical_axis_pix
        
        # ����ppcm����֡ģʽֱ��ʹ�õ�ǰ���ᣩ
        ppcm = vertical_axis_pix / 29.7  # A4����ʵ��Ϊ29.7cm

        # ���ӻ����ᣨ�����ã�
        if top_mid is not None and bottom_mid is not None:
            cv2.line(img, (int(top_mid[0]), int(top_mid[1])), 
                    (int(bottom_mid[0]), int(bottom_mid[1])), (255, 0, 0), 2)
            cv2.circle(img, (int(top_mid[0]), int(top_mid[1])), 5, (0, 0, 255), -1)
            cv2.circle(img, (int(bottom_mid[0]), int(bottom_mid[1])), 5, (0, 0, 255), -1)

    # ������벢��ʾ
    distance_cm = 0.0
    if outer_height > 0:
        distance_cm = estimate_distance(outer_height)
        # ѡȡ�����õĲ���ϵ��
        comp = get_compensation(distance_cm)
        NORMAL_COMPENSATION = comp['normal']
        SLIGHT_ROTATION_COMPENSATION = comp['slight']
        MODERATE_ROTATION_COMPENSATION = comp['moderate']
        LARGE_ROTATION_COMPENSATION = comp['large']

        # ��ͼ����ʾ����
        cv2.putText(frame, f"Dist: {distance_cm:.1f}cm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        # ��ӡ�����ppcm��Ϣ
        print(f"��򳤱�����: {outer_height:.1f}���� -> �������: {distance_cm:.1f}cm")
        if ppcm:
            print(f"���������ppcm: {ppcm:.2f} ����/����")

    # ���ppcm��δ����ɹ���ʹ�ñ�ѡ����
    if ppcm is None:
        if rect_h > rect_w:
            ppcm = rect_h / 29.7
        else:
            ppcm = rect_w / 29.7
        print(f"ʹ�ñ�ѡppcm: {ppcm:.2f} ����/����")

    # �ڿ���
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

        # �ڲ�������
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

            # ���Ĳ����߼�
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # �����δ���
            if len(approx) == 4:
                # ��֡ģʽ��ֱ�ӻ��ڵ�ǰ֡������ת����
                rotation_level = "����"
                compensation = NORMAL_COMPENSATION

                # ���㵱ǰ֡���������߲����������֡ģʽ������ʷ���ݣ�
                pixel_diff_ratio = abs(current_left - current_right) / max(current_left, current_right, 1e-6)

                # ��֡ģʽ���޳���ֱ���ж�
                if pixel_diff_ratio > MODERATE_THRESHOLD:
                    current_mode = "large"
                elif pixel_diff_ratio > SLIGHT_THRESHOLD:
                    current_mode = "moderate"
                elif pixel_diff_ratio > NORMAL_THRESHOLD:
                    current_mode = "slight"
                else:
                    current_mode = "normal"

                # ���ö�Ӧ�Ĳ���ϵ������ת�ȼ�
                if current_mode == "normal":
                    rotation_level = "����"
                    compensation = NORMAL_COMPENSATION
                elif current_mode == "slight":
                    rotation_level = "С����ת"
                    compensation = SLIGHT_ROTATION_COMPENSATION
                elif current_mode == "moderate":
                    rotation_level = "�з���ת"
                    compensation = MODERATE_ROTATION_COMPENSATION
                else:
                    rotation_level = "�����ת"
                    compensation = LARGE_ROTATION_COMPENSATION

                # ����ģʽ��ֱ�Ӽ���Խ��ߣ�
                if current_mode == "normal":
                    # ��ȡ�����ζԽ��߳��ȣ����أ�
                    diagonal_pixels = get_inner_square_diagonal(approx)
                    # ת��Ϊʵ�ʳ��ȣ�cm��
                    diagonal_cm = diagonal_pixels / ppcm * compensation
                    # ����߳����Խ��� / ��2
                    side_length = diagonal_cm / math.sqrt(2)

                    print(f"�����Σ�{rotation_level}��: �߳� = {side_length:.2f}cm (�Խ��� = {diagonal_cm:.2f}cm)")
                    org = (int(center[0]), int(center[1]))
                    cv2.putText(img, f"S={side_length:.2f}cm", org,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # ��תģʽ�������������
                else:
                    # �ڲ������ζԽ���������
                    inner_diagonal_pix = get_inner_square_diagonal(approx)
                    # ��򳤱���������������֡ģʽֱ��ʹ�õ�ǰֵ��
                    outer_axis = vertical_axis_pix
                    # ���ر�������ʵ�ʶԽ��߳���
                    axis_ratio = inner_diagonal_pix / outer_axis
                    actual_diagonal = 29.7 * axis_ratio * compensation  # ʹ��A4����29.7cm��Ϊ��׼
                    # ����߳����Խ��� / ��2
                    side_length = actual_diagonal / math.sqrt(2)

                    print(f"�����Σ�{rotation_level}��: �߳� = {side_length:.2f}cm (�Խ��� = {actual_diagonal:.2f}cm)")
                    org = (int(center[0]), int(center[1]))
                    cv2.putText(img, f"S={side_length:.2f}cm", org,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            else:
                print("δ��⵽������")

    return img


if __name__ == "__main__":
    # ��ʼ��ģʽ
    current_mode = "normal"

    # ��֡����ģʽ
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
            "--gain", "40",
            "--shutter", "30000",
        ]
        subprocess.run(cmd, check=True)

        img = cv2.imread("image.jpg")
        if img is None:
            raise RuntimeError("��ȡ image.jpg ʧ��")

        # ˳ʱ����ת �� ���� �� ��ʱ����ת �� ѹ��
        rot      = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        processed= process_frame(rot)
        restored = cv2.rotate(processed, cv2.ROTATE_90_COUNTERCLOCKWISE)
        output   = cv2.resize(restored, (640, 480))

        cv2.imshow("Result", output)
        print("������� ���� 5 ����Զ��˳����� ESC �˳�")

        key = cv2.waitKey(5000)  
        
        if key & 0xFF == 27:
            print("��⵽ ESC����ǰ�˳�")
        else:
            print("5 �뵽���Զ��˳�")
        
        cv2.destroyAllWindows()