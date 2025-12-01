import subprocess
from typing import List, Optional, Union
import os
import time


class AdbClient:
    """
    ADB 客户端（Python 版，增强实现）

    - 支持文本/二进制执行，带超时与基础重试
    - 优先使用 `exec-out screencap -p` 直接拉取 PNG 流；回退到远端临时文件
    - 提供 connect/get_screen_size/screencap/swipe/tap/keyevent 与通用 shell
    """

    def __init__(self, adb_path: str = 'adb', device_id: Optional[str] = None, default_timeout: int = 20):
        self.adb_path = adb_path or 'adb'
        self.device_id = device_id
        self.default_timeout = max(1, int(default_timeout))

    def _cmd(self, args: List[str]) -> List[str]:
        """构建 ADB 命令参数列表。

        参数:
            args: 追加的子命令与参数，例如 ['shell', 'wm', 'size']。

        返回:
            包含 adb 可执行、设备选择（-s device）与传入 args 的完整命令列表。
        """
        cmd = [self.adb_path]
        if self.device_id:
            cmd += ['-s', self.device_id]
        cmd += args
        return cmd

    def exec_text(self, args: List[str], timeout: Optional[int] = None) -> str:
        """以文本方式执行 ADB 命令并返回标准输出（解码为 str）。

        参数:
            args: 子命令与参数列表，例如 ['shell', 'wm', 'size']。
            timeout: 超时时间（秒），默认使用 `default_timeout`。

        返回:
            命令标准输出的字符串内容（忽略无法解码的字符）。

        异常:
            RuntimeError: 当命令失败（非零退出）或超时时抛出。
        """
        timeout = timeout or self.default_timeout
        cmd = self._cmd(args)
        print(f"[ADB] {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)
            return proc.stdout.decode(errors='ignore')
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ADB 命令失败: {' '.join(cmd)}\n{e.stderr.decode(errors='ignore')}".strip())
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ADB 命令超时: {' '.join(cmd)}")

    def exec_bytes(self, args: List[str], timeout: Optional[int] = None) -> bytes:
        """以二进制方式执行 ADB 命令并返回原始字节输出。

        参数:
            args: 子命令与参数列表，例如 ['exec-out', 'screencap', '-p']。
            timeout: 超时时间（秒），默认使用 `default_timeout`。

        返回:
            命令标准输出的字节内容。

        异常:
            RuntimeError: 当命令失败（非零退出）或超时时抛出。
        """
        timeout = timeout or self.default_timeout
        cmd = self._cmd(args)
        print(f"[ADB BIN] {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=True)
            return proc.stdout
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ADB 命令失败: {' '.join(cmd)}\n{e.stderr.decode(errors='ignore')}".strip())
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"ADB 命令超时: {' '.join(cmd)}")

    # 兼容旧方法名
    def exec(self, args: List[str]) -> str:
        """兼容旧方法名的文本执行封装，等同于 `exec_text`。"""
        return self.exec_text(args)

    def connect(self, device_id: Optional[str] = None) -> None:
        """
        连接设备：
        - 若指定 device_id，则尝试连接该设备
        - 若未指定，则优先选择 `adb devices` 列出的第一个在线设备
        - 若仍无设备，尝试连接 `127.0.0.1:5555`
        
        参数:
            device_id: 期望连接的设备 ID（如 emulator-5554 或 127.0.0.1:5555）。

        行为:
            - 成功连接后更新 `self.device_id` 并打印连接信息。
            - 连接失败时打印错误；若最终无法确定设备将抛出异常。

        异常:
            RuntimeError: 当无法找到任何可用设备时抛出。
        """
        if device_id:
            self.device_id = device_id
        if not self.device_id:
            # 读取在线设备列表
            try:
                out = self.exec_text(['devices'])
                lines = [l.strip() for l in out.splitlines() if l.strip()]
                candidates = []
                for l in lines:
                    if '\tdevice' in l:
                        did = l.split('\t')[0]
                        if did and did != 'List of devices attached':
                            candidates.append(did)
                if candidates:
                    self.device_id = candidates[0]
            except Exception:
                pass
        if not self.device_id:
            # 尝试连接常见模拟器端口
            self.device_id = '127.0.0.1:5555'
            try:
                self.exec_text(['connect', self.device_id])
            except Exception:
                pass
        if self.device_id:
            try:
                out = self.exec_text(['connect', self.device_id])
                if 'connected' not in out.lower() and 'already' not in out.lower():
                    print(out.strip())
                print(f"ADB 已连接: {self.device_id}")
            except Exception as e:
                print(f"连接设备失败（{self.device_id}）：{e}")
        else:
            raise RuntimeError('未找到可用的 ADB 设备，请手动指定 --device-id 或检查 adb devices')

    def shell(self, command: Union[str, List[str]]) -> str:
        """执行 `adb shell` 命令并返回文本输出。

        参数:
            command: 字符串命令（例如 'wm size'）或参数列表（例如 ['wm', 'size']）。

        返回:
            标准输出文本内容。
        """
        if isinstance(command, str):
            return self.exec_text(['shell', command])
        else:
            return self.exec_text(['shell'] + [str(x) for x in command])

    def get_screen_size(self):
        """获取设备屏幕尺寸。

        返回:
            dict: `{"width": int, "height": int}` 当解析成功时。
            None: 无法解析屏幕尺寸时。
        """
        import re
        out = self.exec_text(['shell', 'wm', 'size'])
        m = re.search(r"Physical size:\s*(\d+)x(\d+)", out)
        if m:
            return {"width": int(m.group(1)), "height": int(m.group(2))}
        # 备用：尝试 dumpsys display
        out2 = self.exec_text(['shell', 'dumpsys', 'display'])
        m2 = re.search(r"mBaseDisplayInfo=.+? (\d+)x(\d+)", out2)
        if m2:
            return {"width": int(m2.group(1)), "height": int(m2.group(2))}
        return None

    def screencap_to_file(self, local_path: str) -> None:
        """截图并保存到本地文件。

        行为:
            - 优先使用 `exec-out screencap -p` 获取 PNG 字节并本地写入。
            - 若失败则回退到远端临时文件 `/data/local/tmp/screencap.png` 再执行 `adb pull`。
            - 归一化 CRLF 为 LF，确保 PNG 数据正确。

        参数:
            local_path: 本地保存路径（会自动创建父目录）。
        """
        def _is_valid_png(b: bytes) -> bool:
            # PNG 签名: 89 50 4E 47 0D 0A 1A 0A
            if not b or len(b) < 8:
                return False
            if not b.startswith(b'\x89PNG\r\n\x1a\n'):
                return False
            # 粗略检查是否包含 IEND chunk
            return b'IEND' in b

        # 优先尝试 exec-out 二进制输出
        try:
            data = self.exec_bytes(['exec-out', 'screencap', '-p'], timeout=max(5, self.default_timeout))
            # 某些设备会返回 CRLF，需归一化
            data = data.replace(b'\r\n', b'\n')
            if _is_valid_png(data):
                os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
                with open(local_path, 'wb') as f:
                    f.write(data)
                return
            else:
                print('exec-out 得到的截图疑似无效PNG，尝试回退到远端临时文件方案')
        except Exception as e:
            print(f"exec-out 方式失败，回退到远端临时文件: {e}")

        # 回退方案：写远端再 pull
        remote_path = '/data/local/tmp/screencap.png'
        self.exec_text(['shell', 'screencap', '-p', remote_path])
        os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
        self.exec_text(['pull', remote_path, local_path])
        # 最后再做一次轻量校验，若仍不合法，抛错提示
        try:
            with open(local_path, 'rb') as f:
                data = f.read()
            if not _is_valid_png(data):
                raise RuntimeError('通过远端临时文件获取的截图仍非有效PNG，请检查设备兼容性与ADB版本')
        except Exception as e:
            raise RuntimeError(f'截图保存失败或文件损坏: {e}')

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        """执行滑动手势。

        参数:
            x1, y1: 起点坐标。
            x2, y2: 终点坐标。
            duration_ms: 持续时间毫秒，默认 500ms。
        """
        self.exec_text(['shell', 'input', 'swipe', str(x1), str(y1), str(x2), str(y2), str(duration_ms)])

    def tap(self, x: int, y: int) -> None:
        """执行点击手势。

        参数:
            x, y: 点击坐标。
        """
        self.exec_text(['shell', 'input', 'tap', str(x), str(y)])

    def keyevent(self, key_code: int) -> None:
        """发送按键事件。

        参数:
            key_code: Android 按键码（如 93 表示 PAGE_DOWN）。
        """
        self.exec_text(['shell', 'input', 'keyevent', str(key_code)])