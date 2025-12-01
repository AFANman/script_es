import os
import shutil
import tempfile
import unittest

from python.capture.activity_flows import capture_point_line, capture_trophy_line


class FakeClient:
    def __init__(self):
        self.calls = []  # 记录调用顺序与参数

    def tap(self, x: int, y: int) -> None:
        self.calls.append(("tap", x, y))

    def screencap_to_file(self, local_path: str) -> None:
        # 写入一个小文本文件，模拟截图存在
        os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
        with open(local_path, 'wb') as f:
            f.write(b'fake_png_data')
        self.calls.append(("shot", local_path))


class ActivityFlowsTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="activity_flows_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_capture_point_line(self):
        client = FakeClient()
        out_dir = os.path.join(self.tmpdir, "point_line")

        # 使用较短等待以加快测试（验证流程即可）
        capture_point_line(client, out_dir, wait_seconds=0.0)

        # 验证目录与文件
        self.assertTrue(os.path.isdir(out_dir))
        shots = sorted([f for f in os.listdir(out_dir) if f.endswith('.png')])
        self.assertEqual(shots, [
            'point_line_01.png',
            'point_line_02.png',
            'point_line_03.png',
            'point_line_04.png',
        ])

        # 验证调用顺序：首次 tap 后，依次 (tap -> shot) * 4
        expected_calls = [
            ("tap", 167, 365),
            ("tap", 2363, 747), ("shot", os.path.join(out_dir, 'point_line_01.png')),
            ("tap", 2363, 889), ("shot", os.path.join(out_dir, 'point_line_02.png')),
            ("tap", 2363, 1057), ("shot", os.path.join(out_dir, 'point_line_03.png')),
            ("tap", 2363, 1168), ("shot", os.path.join(out_dir, 'point_line_04.png')),
        ]
        self.assertEqual(client.calls, expected_calls)

    def test_capture_trophy_line(self):
        client = FakeClient()
        out_dir = os.path.join(self.tmpdir, "trophy_line")

        capture_trophy_line(client, out_dir, wait_seconds=0.0)

        self.assertTrue(os.path.isdir(out_dir))
        shots = sorted([f for f in os.listdir(out_dir) if f.endswith('.png')])
        self.assertEqual(shots, [
            'trophy_line_01.png',
            'trophy_line_02.png',
            'trophy_line_03.png',
            'trophy_line_04.png',
        ])

        expected_calls = [
            ("tap", 167, 573),
            ("tap", 2363, 747), ("shot", os.path.join(out_dir, 'trophy_line_01.png')),
            ("tap", 2363, 889), ("shot", os.path.join(out_dir, 'trophy_line_02.png')),
            ("tap", 2363, 1057), ("shot", os.path.join(out_dir, 'trophy_line_03.png')),
            ("tap", 2363, 1168), ("shot", os.path.join(out_dir, 'trophy_line_04.png')),
        ]
        self.assertEqual(client.calls, expected_calls)


if __name__ == '__main__':
    unittest.main()