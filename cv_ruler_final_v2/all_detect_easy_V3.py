# -*- coding: gbk -*-
#ȥ���˽Ƕȱ任,ǰ���ʻ��������հ汾,�޸���cv2.putText Ҫ�� org int��ʽ����
import cv2
import numpy as np
import time
import subprocess

# === ��������๦�ܲ��� ===
# �������ݣ�(���ظ߶�, ʵ�ʾ���cm)���ɸ���ʵ���������

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


ENABLE_CONTINUOUS_VIDEO = 0
ENABLE_SINGLE_FRAME = 1
ENABLE_LOCAL_IMAGE = 0

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


# �ֶ���ֵ
# ���ز����
NORMAL_THRESHOLD = 0.00000000000002  # 2% ����Ϊ����
SLIGHT_THRESHOLD = 0.00015  # 2-5% ΪС����ת
MODERATE_THRESHOLD = 0.22  # 5-10% Ϊ�з���ת
# >10% Ϊ�����ת

# ��������
OUTLIER_THRESHOLD = 0.2  # �쳣ֵ������ֵ
CIRCLE_MEASURE_COUNT = 5  # Բ�β�������

HISTORY_LEN = 3
# ��ʷ���ݳ���,����calculate_filtered_averageʹ��,����Ƶģʽ����,��֡����һ����û��

# ��ʷ���ݴ洢
circle_history = []
outer_vertical_pixels = []  # ������ұ���������ʷ
outer_axis_pixels = []  # ���������������ʷ
#���߶���ʷ�����ڲ��ƽ����
outer_height_history = []


def get_contour_center(contour):
    """�����������ĵ�"""
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return (0, 0)
    return (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))


def distance(p1, p2):
    """������������"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

#��֡ƽ���õ�,��HISTORY_LEN = 3����ʹ��
def calculate_filtered_average(values):
    """�Ľ����쳣ֵ����ƽ��ֵ����
    ʹ����λ�����ķ�λ�෨�����ֵ�������ټ���ֵ�Թ��˽����Ӱ��
    """
    if len(values) <= 4:
        return np.mean(values), values.copy()
    
    # ת��Ϊnumpy������ڼ���
    values_np = np.array(values, dtype=np.float64)
    
    # ʹ����λ�����Ǿ�ֵ��Ϊ��׼�����ټ���ֵӰ��
    median_val = np.median(values_np)
    
    # �����ķ�λ��(IQR)
    q1 = np.percentile(values_np, 25)  # ��һ�ķ�λ
    q3 = np.percentile(values_np, 75)  # �����ķ�λ
    iqr = q3 - q1
    
    # ����IQRȷ���쳣ֵ��ֵ�����Ƚ����쳣ֵ�жϣ�
    # 1.5��ͳ��ѧ�г��õ�ϵ�����ɸ���ʵ���������
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # �����쳣ֵ
    filtered = values_np[(values_np >= lower_bound) & (values_np <= upper_bound)]
    
    # ȷ�����˺����㹻������
    if len(filtered) < 2:
        # ������˺�����̫�٣�ʹ�÷ſ����ֵ
        lower_bound = q1 - 3.0 * iqr
        upper_bound = q3 + 3.0 * iqr
        filtered = values_np[(values_np >= lower_bound) & (values_np <= upper_bound)]
        
        # ��Ȼ�����򷵻�ԭʼ��ֵ
        if len(filtered) < 2:
            return np.mean(values_np), values.copy()
    
    # ������˺��ƽ��ֵ
    filtered_mean = np.mean(filtered)
    return filtered_mean, filtered.tolist()


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


def get_inner_square_axis(approx_inner):
    """��ȡ�ڲ����������������������±��е����ߣ�"""
    pts = approx_inner.reshape(4, 2).astype(np.float32)

    # ��y���������������±�
    y_sorted = pts[np.argsort(pts[:, 1])]
    top_pts = y_sorted[:2]  # �ϱ�����
    bottom_pts = y_sorted[2:]  # �±�����

    # �������±��е�
    top_mid = ((top_pts[0][0] + top_pts[1][0]) / 2, (top_pts[0][1] + top_pts[1][1]) / 2)
    bottom_mid = ((bottom_pts[0][0] + bottom_pts[1][0]) / 2, (bottom_pts[0][1] + bottom_pts[1][1]) / 2)

    # ��������������
    return distance(top_mid, bottom_mid)


def process_frame(frame):
    """����֡ͼ��ĺ��ĺ���"""
    global circle_history, outer_vertical_pixels, outer_axis_pixels, current_mode
    global outer_height_history  # �����ʷ

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
    ppcm = ((rect_w / 21.0) + (rect_h / 29.7)) / 2  # ����/���ױ���������A4ֽ�ߴ磩
    print(f"ppcmֵΪ:{ppcm}")
    # === �������������߶ȣ����ڲ�ࣩ===
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

        # ȷ����ǰģʽ�����߶�
        if current_mode == "normal":
            # ���ӣ�ȡ���ұ�ƽ���߶�
            current_height = (left_pix + right_pix) / 2
        else:
            # б�ӣ�ȡ����߶�
            current_height = axis_pix

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
        NORMAL_COMPENSATION2 = comp['normal2']
        CIRCLE_COMPENSATION_FACTOR = comp['circle']

        # ��ͼ����ʾ����
        cv2.putText(frame, f"Dist: {distance_cm:.1f}cm", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        # ��ӡ������Ϣ
        print(f"���߶�: {outer_height:.1f}���� -> �������: {distance_cm:.1f}cm")

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
            lengths = []


            # �����δ���
            if len(approx) == 4:
                # ȷ����ǰ��ת�ȼ��Ͳ���ϵ��
                rotation_level = "����"
                compensation = NORMAL_COMPENSATION

                # �㹻�������ʷ���ݲŽ���ģʽ�ж�
                if len(outer_vertical_pixels) >= HISTORY_LEN:
                    # ����ƽ���������ز����
                    avg_left = np.mean([p[0] for p in outer_vertical_pixels])
                    avg_right = np.mean([p[1] for p in outer_vertical_pixels])
                    # pixel_diff_ratio = abs(avg_left - avg_right) / max(avg_left, avg_right, 1e-6)
                    # ���ݵ�ǰģʽ���ò���ϵ��

                    compensation = NORMAL_COMPENSATION
                for i in range(4):
                    p1 = tuple(approx[i][0])
                    p2 = tuple(approx[(i + 1) % 4][0])
                    d = distance(p1, p2) / ppcm
                    d_compensated = d * compensation
                    lengths.append(d_compensated)

                avg_length, _ = calculate_filtered_average(lengths)
                print(f"�����Σ�{rotation_level}��: {avg_length:.2f}cm")
                cv2.putText(img, f"S={avg_length:.2f}cm", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)



            # �����δ���
            elif len(approx) == 3:
                for i in range(3):
                    p1 = tuple(approx[i][0])
                    p2 = tuple(approx[(i + 1) % 3][0])
                    d = distance(p1, p2) / ppcm
                    d_compensated = d * NORMAL_COMPENSATION2  # ʹ�����Ӳ���ϵ��
                    lengths.append(d_compensated)

                avg_length, _ = calculate_filtered_average(lengths)
                print(f"������: {avg_length:.2f}cm")
                cv2.putText(img, f"S={avg_length:.2f}cm", center,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Բ�δ���
            else:
                (x0, y0), radius = cv2.minEnclosingCircle(cnt)
                diameter = 2 * radius / ppcm
                diameter_compensated = diameter * CIRCLE_COMPENSATION_FACTOR

                circle_history.append(diameter_compensated)
                if len(circle_history) > CIRCLE_MEASURE_COUNT:
                    circle_history.pop(0)

                avg_diameter, _ = calculate_filtered_average(circle_history)
                print(f"Բ��: {avg_diameter:.2f}cm")
                cv2.putText(img, f"D={avg_diameter:.2f}cm", (int(x0), int(y0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img

if __name__ == "__main__":
    # ��ʼ��ģʽ
    current_mode = "normal"

    # ������Ƶ�����ԣ�...

    # ��֡����ģʽ��libcamera-still �� 2028��1520��
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
    "--gain", "25",
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

        # �ȴ� 10000 ���루�� 10 �룩
        key = cv2.waitKey(5000)  
        
        # ����� 10 ���ڰ��� ESC��27��������ǰ�˳�
        if key & 0xFF == 27:
            print("��⵽ ESC����ǰ�˳�")
        else:
            print("5 �뵽���Զ��˳�")
        
        cv2.destroyAllWindows()