# -*- coding: gbk -*-
# -*- coding: gbk -*-
import RPi.GPIO as GPIO
import subprocess
import time
import sys

EXPOSURE_PARAMS = {
    'gain': '25',       # 增益值
    'shutter': '10000'  # 快门时间
}
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



# —— 然后修改 while True 循环内的功能按键处理逻辑 —————————
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
                        # 其他功能直接执行（修正重复逻辑，添加曝光参数）
                        print(f"GPIO{pin} pressed → run {info['path']} {info['arg'] or ''}")
                        # 初始化命令，先加脚本路径
                        cmd = ['python3', info['path']]
                        # 加入统一曝光参数
                        cmd.extend(['--gain', EXPOSURE_PARAMS['gain'], 
                                    '--shutter', EXPOSURE_PARAMS['shutter']])
                        # 再加入脚本自身的参数（如果有）
                        if info['arg'] is not None:
                            cmd.append(str(info['arg']))
                        # 执行脚本
                        try:
                            subprocess.run(cmd, check=True)
                        except subprocess.CalledProcessError as e:
                            print(f"Error: {e}")

                # —— 以下数字输入、确定按键的原有逻辑保持不变 ——
                elif info.get('type') == 'number' and input_mode:
                    input_number += str(info['value'])
                    print(f"输入数字: {input_number}")

                elif info.get('type') == 'confirm' and input_mode:
                    if input_number:  
                        print(f"确定输入 → 执行功能3-模式{current_arg}，编号{input_number}")
                        # 功能3脚本也要加曝光参数，这里同步修改
                        cmd = [
                            'python3',
                            './detect_number_shapes_V11_pi.py',
                            '--gain', EXPOSURE_PARAMS['gain'],  # 加曝光参数
                            '--shutter', EXPOSURE_PARAMS['shutter'],  # 加曝光参数
                            str(current_arg),
                            input_number
                        ]
                        try:
                            subprocess.run(cmd, check=True)
                        except subprocess.CalledProcessError as e:
                            print(f"Error: {e}")
                    else:
                        print("未输入数字，取消执行")
                    input_mode = False
                    current_arg = None
                    input_number = ""

                last_pressed[pin] = time.time()
                # 等待按键松开
                while GPIO.input(pin) == GPIO.LOW:
                    time.sleep(0.01)

        time.sleep(0.01)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    GPIO.cleanup()