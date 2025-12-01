import argparse
import os
import sys
from typing import Optional

CURRENT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from python.capture.activity_flows import capture_point_line, capture_trophy_line


def main():
    parser = argparse.ArgumentParser(description='活动积分/奖杯分数线截图执行模板 (Python 版)')
    parser.add_argument('--device-id', type=str, default=None, help='ADB 设备 ID；默认自动选择在线设备或连接 127.0.0.1:5555')
    parser.add_argument('--adb', type=str, default='adb', help='ADB 可执行路径，默认 adb')
    parser.add_argument('--out-root', type=str, default='captures', help='输出根目录，默认 captures')
    parser.add_argument('--wait-seconds', type=float, default=1.0, help='每步等待秒数，默认 1.0')
    args = parser.parse_args()

    # 初始化 ADB 客户端并自动连接
    try:
        from python.capture.adb_client import AdbClient
        client = AdbClient(adb_path=args.adb)
        client.connect(args.device_id)
    except Exception as e:
        print('ADB 客户端初始化/连接失败：', e)
        sys.exit(2)

    # 准备输出目录（各块脚本独立目录）
    point_dir = os.path.join(args.out_root, 'point_line')
    trophy_dir = os.path.join(args.out_root, 'trophy_line')
    os.makedirs(point_dir, exist_ok=True)
    os.makedirs(trophy_dir, exist_ok=True)

    # 执行两块脚本，主方法作为调用模板
    print('开始执行：积分线脚本')
    capture_point_line(client, point_dir, wait_seconds=args.wait_seconds)
    print('完成：积分线脚本，输出目录 =', point_dir)

    print('开始执行：奖杯分数线脚本')
    capture_trophy_line(client, trophy_dir, wait_seconds=args.wait_seconds)
    print('完成：奖杯分数线脚本，输出目录 =', trophy_dir)


if __name__ == '__main__':
    main()