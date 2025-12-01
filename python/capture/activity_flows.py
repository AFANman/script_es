import os
from typing import Protocol


class SupportsCaptureClient(Protocol):
    """与 abd_command 约束一致的最小 ADB 客户端接口。"""
    def tap(self, x: int, y: int) -> None: ...
    def screencap_to_file(self, local_path: str) -> None: ...


from python.capture.abd_command import (
    ensure_dir,
    wait,
    capture,
    open_best_record_rank,
    open_point_rank,
    select_rank_100,
    select_rank_1000,
    select_rank_5000,
    select_rank_10000,
)




def capture_point_line(client: SupportsCaptureClient, out_dir: str, wait_seconds: float = 1.0) -> None:
    """
    积分线脚本：
    - 点击 171,355，等待 1s
    - 依次点击并截图：
      2363,747；2363,889；2363,1057；2363,1168

    参数:
    - client: 具备 tap 与 screencap_to_file 方法的 ADB 客户端。
    - out_dir: 截图输出目录（该模块会自行创建）。
    - wait_seconds: 每个脚本操作间的等待秒数，默认 1 秒。
    """

    ensure_dir(out_dir)

    # 第一步：进入积分排行入口（包装方法）
    open_point_rank(client)
    wait(wait_seconds)

    # 后续排名选择 + 截图（包装方法）
    for idx, select in enumerate([
        select_rank_100,
        select_rank_1000,
        select_rank_5000,
        select_rank_10000,
    ], start=1):
        select(client)
        # 在点击后给予页面响应时间
        wait(wait_seconds)
        filename = os.path.join(out_dir, f"point_line_{str(idx).zfill(2)}.png")
        capture(client, filename)
        # 截图完成后也做间隔，保证后续动作更稳定
        wait(wait_seconds)


def capture_trophy_line(client: SupportsCaptureClient, out_dir: str, wait_seconds: float = 1.0) -> None:
    """
    奖杯分数线脚本：
    - 点击 167,573，等待 1s
    - 依次点击并截图：
      2363,747；2363,889；2363,1057；2363,1168

    参数:
    - client: 具备 tap 与 screencap_to_file 方法的 ADB 客户端。
    - out_dir: 截图输出目录（该模块会自行创建）。
    - wait_seconds: 每个脚本操作间的等待秒数，默认 1 秒。
    """

    ensure_dir(out_dir)

    # 第一步：进入最高记录排行入口（包装方法）
    open_best_record_rank(client)
    wait(wait_seconds)

    # 后续排名选择 + 截图（包装方法）
    for idx, select in enumerate([
        select_rank_100,
        select_rank_1000,
        select_rank_5000,
        select_rank_10000,
    ], start=1):
        select(client)
        wait(wait_seconds)
        filename = os.path.join(out_dir, f"trophy_line_{str(idx).zfill(2)}.png")
        capture(client, filename)
        wait(wait_seconds)