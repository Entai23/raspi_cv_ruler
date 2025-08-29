# -*- coding: gbk -*-
import RPi.GPIO as GPIO
import time
import sys

# --------------------------
# 配置需要测试的GPIO引脚（BCM编号）
# 请根据你的实际连线修改此列表
# --------------------------
TEST_PINS = [21, 20, 16, 12, 26, 19, 13,  # 原有引脚
             17, 27, 22, 5, 6, 18,        # 新增功能引脚
             23, 24, 25, 10, 9, 11, 8]    # 数字输入及控制引脚（可根据需要增减）

# 配置参数
DEBOUNCE_TIME = 0.1  # 去抖时间（秒）
PRINT_INTERVAL = 0.05  # 状态刷新间隔（秒）

# 全局变量
last_state = {}  # 记录每个引脚的上一状态
last_change_time = {}  # 记录状态变化时间

def init_gpio():
    """初始化GPIO设置"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    for pin in TEST_PINS:
        # 配置为输入模式，启用内部上拉电阻（默认高电平，按下为低电平）
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        last_state[pin] = GPIO.input(pin)
        last_change_time[pin] = time.time()
    
    print("GPIO初始化完成，开始监测按钮状态...")
    print("引脚编号 | 状态 (高/低) | 事件")
    print("-" * 40)

def print_state(pin, current_state, event):
    """打印引脚状态变化"""
    state_str = "高" if current_state else "低"
    print(f"  GPIO{pin:2d}   |    {state_str:2s}     | {event}")

def main():
    try:
        init_gpio()
        
        while True:
            current_time = time.time()
            
            for pin in TEST_PINS:
                current_state = GPIO.input(pin)
                # 检测状态变化且超过去抖时间
                if current_state != last_state[pin] and \
                   current_time - last_change_time[pin] > DEBOUNCE_TIME:
                    
                    # 判断是按下还是松开
                    if current_state == GPIO.LOW:
                        print_state(pin, current_state, "按钮按下")
                    else:
                        print_state(pin, current_state, "按钮松开")
                    
                    # 更新状态记录
                    last_state[pin] = current_state
                    last_change_time[pin] = current_time
            
            time.sleep(PRINT_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n程序已退出")
    finally:
        GPIO.cleanup()  # 清理GPIO资源

if __name__ == "__main__":
    main()
    