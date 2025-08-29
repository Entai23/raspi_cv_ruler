# -*- coding: gbk -*-
# -*- coding: gbk -*-
import RPi.GPIO as GPIO
import subprocess
import time
import sys


# ���� �������ű� ӳ�䣨BCM ��ţ� ����������������������������������������������������
# ���У�GPIO10 �� detect_multi_shapes_V12_pi.py
# �������ĸ��û��Զ���� BCM ���� �� detect_number_shapes_V5.py
BUTTON_MAP = {
    21: { 'path': './all_detect_easy_V3.py', 'arg': None },
    20: { 'path': './detect_multi_shapes_V12_pi.py', 'arg': None },       # ����1

    16: { 'path': './detect_number_shapes_V11_pi.py', 'arg': 1 },            # ����2 �� �� 1 ��Ϊģʽ
    12: { 'path': './detect_number_shapes_V11_pi.py', 'arg': 2 },            # ����3 �� �� 2
    26: { 'path': './detect_number_shapes_V11_pi.py', 'arg': 3 },            # ����4 �� �� 3
    19: { 'path': './detect_number_shapes_V11_pi.py', 'arg': 4 },            # ����5 �� �� 4
    13: { 'path': './detect_number_shapes_V11_pi.py', 'arg': 5 },

    5:{ 'path': './detect_turn.py', 'arg': None },
    # ������������ʵ��BCM��ţ�
    18: { 'type': 'number', 'value': 0 },  # ����0
    23: { 'type': 'number', 'value': 1 },  # ����1
    24: { 'type': 'number', 'value': 2 },  # ����2
    25:  { 'type': 'number', 'value': 3 },  # ����3
    17: { 'type': 'number', 'value': 4 },  # ����4
    27: { 'type': 'number', 'value': 5 },  # ����5
    22: { 'type': 'number', 'value': 6 },  # ����6
    10: { 'type': 'number', 'value': 7 },  # ����7
    9: { 'type': 'number', 'value': 8 },  # ����8
    11:  { 'type': 'number', 'value': 9 },  # ����9
    8: { 'type': 'confirm' }              # ȷ����ť
}

# ȫ�ֱ���
debounce_time = 0.2  # ȥ��������룩
last_pressed = {pin: 0 for pin in BUTTON_MAP}
input_mode = False  # �Ƿ�����������ģʽ
current_arg = None  # ����3�ĵ�ǰģʽ����
input_number = ""  # ����������ַ���

# GPIO��ʼ��
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for pin in BUTTON_MAP:
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

print("Waiting for buttons... (Ctrl+C to exit)")

try:
    while True:
        now = time.time()
        for pin, info in BUTTON_MAP.items():
            if GPIO.input(pin) == GPIO.LOW and now - last_pressed[pin] > debounce_time:
                # �����ܰ�����������/ȷ������
                if 'path' in info:
                    # ����Ƿ��ǹ���3����Ҫ���������ģʽ��
                    if info['path'] == './detect_number_shapes_V11_pi.py':
                        input_mode = True
                        current_arg = info['arg']
                        input_number = ""
                        print(f"GPIO{pin} pressed �� ���빦��3-ģʽ{current_arg}�����������ֺ�ȷ��")
                    else:
                        # ��������ֱ��ִ��
                        print(f"GPIO{pin} pressed �� run {info['path']} {info['arg'] or ''}")
                        cmd = ['python3', info['path']]
                        if info['arg'] is not None:
                            cmd.append(str(info['arg']))
                        try:
                            subprocess.run(cmd, check=True)
                        except subprocess.CalledProcessError as e:
                            print(f"Error: {e}")

                # ������������
                elif info.get('type') == 'number' and input_mode:
                    input_number += str(info['value'])
                    print(f"��������: {input_number}")

                # ����ȷ������
                elif info.get('type') == 'confirm' and input_mode:
                    if input_number:  # ȷ��������
                        print(f"ȷ������ �� ִ�й���3-ģʽ{current_arg}�����{input_number}")
                        # ִ�нű�������ģʽ���������ֱ��
                        cmd = [
                            'python3',
                            './detect_number_shapes_V11_pi.py',
                            str(current_arg),
                            input_number
                        ]
                        try:
                            subprocess.run(cmd, check=True)
                        except subprocess.CalledProcessError as e:
                            print(f"Error: {e}")
                    else:
                        print("δ�������֣�ȡ��ִ��")
                    # �˳�����ģʽ
                    input_mode = False
                    current_arg = None
                    input_number = ""

                last_pressed[pin] = time.time()
                # �ȴ��ɿ��ټ���
                while GPIO.input(pin) == GPIO.LOW:
                    time.sleep(0.01)

        time.sleep(0.01)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    GPIO.cleanup()