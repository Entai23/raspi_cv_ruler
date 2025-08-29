# -*- coding: gbk -*-
# -*- coding: gbk -*-
import RPi.GPIO as GPIO
import subprocess
import time
import sys


# —— 按键→脚本 映射（BCM 编号） ——————————————————————————
# 现有：GPIO10 → detect_multi_shapes_V12_pi.py
# 新增：四个用户自定义的 BCM 引脚 → detect_number_shapes_V5.py
BUTTON_MAP = {
    21: { 'path': './all_detect_easy_V3.py', 'arg': None },
    20: { 'path': './detect_multi_shapes_V12_pi.py', 'arg': None },       # 按键1

    16: { 'path': './detect_number_shapes_V11_pi.py', 'arg': 1 },            # 按键2 → 传 1 作为模式
    12: { 'path': './detect_number_shapes_V11_pi.py', 'arg': 2 },            # 按键3 → 传 2
    26: { 'path': './detect_number_shapes_V11_pi.py', 'arg': 3 },            # 按键4 → 传 3
    19: { 'path': './detect_number_shapes_V11_pi.py', 'arg': 4 },            # 按键5 → 传 4
    13: { 'path': './detect_number_shapes_V11_pi.py', 'arg': 5 },

    5:{ 'path': './detect_turn.py', 'arg': None },
    # 数字输入区（实际BCM编号）
    18: { 'type': 'number', 'value': 0 },  # 数字0
    23: { 'type': 'number', 'value': 1 },  # 数字1
    24: { 'type': 'number', 'value': 2 },  # 数字2
    25:  { 'type': 'number', 'value': 3 },  # 数字3
    17: { 'type': 'number', 'value': 4 },  # 数字4
    27: { 'type': 'number', 'value': 5 },  # 数字5
    22: { 'type': 'number', 'value': 6 },  # 数字6
    10: { 'type': 'number', 'value': 7 },  # 数字7
    9: { 'type': 'number', 'value': 8 },  # 数字8
    11:  { 'type': 'number', 'value': 9 },  # 数字9
    8: { 'type': 'confirm' }              # 确定按钮
}

# 全局变量
debounce_time = 0.2  # 去抖间隔（秒）
last_pressed = {pin: 0 for pin in BUTTON_MAP}
input_mode = False  # 是否处于数字输入模式
current_arg = None  # 功能3的当前模式参数
input_number = ""  # 输入的数字字符串

# GPIO初始化
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
                # 处理功能按键（非数字/确定键）
                if 'path' in info:
                    # 检查是否是功能3（需要数字输入的模式）
                    if info['path'] == './detect_number_shapes_V11_pi.py':
                        input_mode = True
                        current_arg = info['arg']
                        input_number = ""
                        print(f"GPIO{pin} pressed → 进入功能3-模式{current_arg}，请输入数字后按确定")
                    else:
                        # 其他功能直接执行
                        print(f"GPIO{pin} pressed → run {info['path']} {info['arg'] or ''}")
                        cmd = ['python3', info['path']]
                        if info['arg'] is not None:
                            cmd.append(str(info['arg']))
                        try:
                            subprocess.run(cmd, check=True)
                        except subprocess.CalledProcessError as e:
                            print(f"Error: {e}")

                # 处理数字输入
                elif info.get('type') == 'number' and input_mode:
                    input_number += str(info['value'])
                    print(f"输入数字: {input_number}")

                # 处理确定按键
                elif info.get('type') == 'confirm' and input_mode:
                    if input_number:  # 确保有输入
                        print(f"确定输入 → 执行功能3-模式{current_arg}，编号{input_number}")
                        # 执行脚本并传递模式参数和数字编号
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
                        print("未输入数字，取消执行")
                    # 退出输入模式
                    input_mode = False
                    current_arg = None
                    input_number = ""

                last_pressed[pin] = time.time()
                # 等待松开再继续
                while GPIO.input(pin) == GPIO.LOW:
                    time.sleep(0.01)

        time.sleep(0.01)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    GPIO.cleanup()