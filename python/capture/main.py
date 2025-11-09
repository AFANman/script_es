import argparse
import os
import sys
import time
from typing import Optional

try:
    import easyocr  # type: ignore
except Exception:
    easyocr = None

CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from python.capture.orchestrator import run_capture


def main():
    parser = argparse.ArgumentParser(description='活动页面截图编排 CLI (Python 版)')
    parser.add_argument('--device-id', type=str, default=None, help='ADB 设备 ID；默认自动选择在线设备或连接 127.0.0.1:5555')
    parser.add_argument('--adb', type=str, default='adb', help='ADB 可执行路径，默认 adb')
    parser.add_argument('--mode', type=str, choices=['swipe', 'tap', 'keyevent'], default='swipe', help='页面滚动/翻页方式，默认 swipe')
    parser.add_argument('--prefix', type=str, default='capture', help='输出文件前缀与目录名，默认 capture')
    parser.add_argument('--interval', type=int, default=600, help='每步间隔毫秒，默认 600')
    # 采集参数已默认化，无需额外传入坐标与滑动
    parser.add_argument('--count', type=int, default=6, help='每页截图张数上限，默认 6')
    # keyevent/tap 模式下将使用内置默认值
    parser.add_argument('--langs', type=str, default='ch_sim,en', help='页码 OCR 语言，默认 ch_sim,en')

    args = parser.parse_args()

    # 初始化 ADB 客户端并自动连接
    try:
        from python.capture.adb_client import AdbClient
        client = AdbClient(adb_path=args.adb)
        client.connect(args.device_id)
    except Exception as e:
        print('ADB 客户端初始化/连接失败：', e)
        sys.exit(2)

    # 初始化 OCR
    if easyocr is None:
        print('未安装 easyocr，安装：python -m pip install -r python/requirements.txt')
        sys.exit(2)
    langs = [s.strip() for s in args.langs.split(',') if s.strip()]
    try:
        reader = easyocr.Reader(langs)
    except Exception as e:
        print('初始化 easyocr.Reader 失败：', e)
        sys.exit(2)

    # 开始编排采集
    run_capture(
        client,
        reader,
        device_id=args.device_id,
        mode=args.mode,
        prefix=args.prefix,
        interval=args.interval,
        next_xy=None,
        prev_xy=None,
        swipe=None,
        count=args.count,
        keyevent=None,
        tap_xy=None,
    )


if __name__ == '__main__':
    main()