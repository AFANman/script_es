import os
import time
from typing import Protocol


class SupportsAdbClient(Protocol):
    """与 AdbClient 兼容的最小接口。"""

    def tap(self, x: int, y: int) -> None: ...

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None: ...

    def screencap_to_file(self, local_path: str) -> None: ...


def ensure_dir(path: str) -> None:
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)


def wait(seconds: float) -> None:
    """等待指定秒数（非负）。"""
    time.sleep(max(0.0, float(seconds)))


def tap(client: SupportsAdbClient, x: int, y: int) -> None:
    """执行点击。"""
    client.tap(x, y)


def capture(client: SupportsAdbClient, out_path: str) -> None:
    """执行截图保存到 out_path（会自动创建父目录）。"""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    client.screencap_to_file(out_path)


# ----------------------
# 业务原子操作（新增）
# ----------------------

# 约定坐标常量（便于复用与统一维护）
PREV_PAGE_XY = (457, 1260)
NEXT_PAGE_XY = (1986, 1260)
SWIPE_COORDS = (487, 1074, 334, 233)
# 列表区域滑动坐标 (X=1200 是列表中心区域)
SCROLL_DOWN_COORDS = (1200, 1000, 1200, 600) # 手指上滑，查看下方内容
SCROLL_UP_COORDS = (1200, 600, 1200, 1000)   # 手指下滑，查看上方内容

# 活动页面入口与排名坐标
BEST_RECORD_RANK_XY = (167, 573)   # 最高记录排行入口
POINT_RANK_XY = (167, 365)         # 积分排行入口

RANK_100_XY = (2363, 542)
RANK_1000_XY = (2362, 669)
RANK_5000_XY = (2360, 797)
RANK_10000_XY = (2362, 927)
RANK_50000_XY = (2360, 1055)
RANK_1_XY = (2364, 413)


def previous_page(client: SupportsAdbClient) -> None:
    """
    上一页：点击固定坐标。

    参数:
    - client: 具备 tap 方法的 ADB 客户端。
    """
    x, y = PREV_PAGE_XY
    client.tap(x, y)


def next_page(client: SupportsAdbClient) -> None:
    """
    下一页：点击固定坐标。

    参数:
    - client: 具备 tap 方法的 ADB 客户端。
    """
    x, y = NEXT_PAGE_XY
    client.tap(x, y)


def swipe_page(client: SupportsAdbClient, duration_ms: int = 500) -> None:
    """
    页面滑动：按固定坐标执行 input swipe。
    
    参数:
    - client: 具备 swipe 方法的 ADB 客户端。
    - duration_ms: 滑动持续时间毫秒，默认 500ms。
    """
    x1, y1, x2, y2 = SWIPE_COORDS
    client.swipe(x1, y1, x2, y2, duration_ms)


def scroll_list_down(client: SupportsAdbClient, duration_ms: int = 800) -> None:
    """
    列表向下滑动（手指上滑），查看下方内容。
    """
    x1, y1, x2, y2 = SCROLL_DOWN_COORDS
    client.swipe(x1, y1, x2, y2, duration_ms)


def scroll_list_up(client: SupportsAdbClient, duration_ms: int = 800) -> None:
    """
    列表向上滑动（手指下滑），查看上方内容。
    """
    x1, y1, x2, y2 = SCROLL_UP_COORDS
    client.swipe(x1, y1, x2, y2, duration_ms)


def open_best_record_rank(client: SupportsAdbClient) -> None:
    """进入最高记录排行页（点击 167,573）。"""
    x, y = BEST_RECORD_RANK_XY
    client.tap(x, y)


def open_point_rank(client: SupportsAdbClient) -> None:
    """进入积分排行页（点击 167,365）。"""
    x, y = POINT_RANK_XY
    client.tap(x, y)


def select_rank_100(client: SupportsAdbClient) -> None:
    """选择 100 位（点击 2363,747）。"""
    x, y = RANK_100_XY
    client.tap(x, y)


def select_rank_1000(client: SupportsAdbClient) -> None:
    """选择 1000 位（点击 2363,889）。"""
    x, y = RANK_1000_XY
    client.tap(x, y)


def select_rank_5000(client: SupportsAdbClient) -> None:
    """选择 5000 位（点击 2363,1057）。"""
    x, y = RANK_5000_XY
    client.tap(x, y)


def select_rank_10000(client: SupportsAdbClient) -> None:
    """选择 10000 位（点击 2363,1168）。"""
    x, y = RANK_10000_XY
    client.tap(x, y)


def select_rank_50000(client: SupportsAdbClient) -> None:
    """选择 50000 位（点击 2363,1280）。"""
    x, y = RANK_50000_XY
    client.tap(x, y)


def select_rank_1(client: SupportsAdbClient) -> None:
    """选择 1 位（点击 2344,606）。"""
    x, y = RANK_1_XY
    client.tap(x, y)