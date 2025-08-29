# -*- coding: gbk -*-
import RPi.GPIO as GPIO
import time
import sys

# --------------------------
# ������Ҫ���Ե�GPIO���ţ�BCM��ţ�
# ��������ʵ�������޸Ĵ��б�
# --------------------------
TEST_PINS = [21, 20, 16, 12, 26, 19, 13,  # ԭ������
             17, 27, 22, 5, 6, 18,        # ������������
             23, 24, 25, 10, 9, 11, 8]    # �������뼰�������ţ��ɸ�����Ҫ������

# ���ò���
DEBOUNCE_TIME = 0.1  # ȥ��ʱ�䣨�룩
PRINT_INTERVAL = 0.05  # ״̬ˢ�¼�����룩

# ȫ�ֱ���
last_state = {}  # ��¼ÿ�����ŵ���һ״̬
last_change_time = {}  # ��¼״̬�仯ʱ��

def init_gpio():
    """��ʼ��GPIO����"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    
    for pin in TEST_PINS:
        # ����Ϊ����ģʽ�������ڲ��������裨Ĭ�ϸߵ�ƽ������Ϊ�͵�ƽ��
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        last_state[pin] = GPIO.input(pin)
        last_change_time[pin] = time.time()
    
    print("GPIO��ʼ����ɣ���ʼ��ⰴť״̬...")
    print("���ű�� | ״̬ (��/��) | �¼�")
    print("-" * 40)

def print_state(pin, current_state, event):
    """��ӡ����״̬�仯"""
    state_str = "��" if current_state else "��"
    print(f"  GPIO{pin:2d}   |    {state_str:2s}     | {event}")

def main():
    try:
        init_gpio()
        
        while True:
            current_time = time.time()
            
            for pin in TEST_PINS:
                current_state = GPIO.input(pin)
                # ���״̬�仯�ҳ���ȥ��ʱ��
                if current_state != last_state[pin] and \
                   current_time - last_change_time[pin] > DEBOUNCE_TIME:
                    
                    # �ж��ǰ��»����ɿ�
                    if current_state == GPIO.LOW:
                        print_state(pin, current_state, "��ť����")
                    else:
                        print_state(pin, current_state, "��ť�ɿ�")
                    
                    # ����״̬��¼
                    last_state[pin] = current_state
                    last_change_time[pin] = current_time
            
            time.sleep(PRINT_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n�������˳�")
    finally:
        GPIO.cleanup()  # ����GPIO��Դ

if __name__ == "__main__":
    main()
    