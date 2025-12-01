import unittest


class FakeClient:
    def __init__(self):
        self.calls = []

    def tap(self, x: int, y: int) -> None:
        self.calls.append(("tap", x, y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        self.calls.append(("swipe", x1, y1, x2, y2, duration_ms))

    def screencap_to_file(self, local_path: str) -> None:
        self.calls.append(("screencap", local_path))


class TestAbdCommand(unittest.TestCase):
    def test_previous_page_calls_tap_with_expected_coords(self):
        from python.capture.abd_command import previous_page, PREV_PAGE_XY
        client = FakeClient()
        previous_page(client)
        self.assertEqual(client.calls, [("tap", PREV_PAGE_XY[0], PREV_PAGE_XY[1])])

    def test_next_page_calls_tap_with_expected_coords(self):
        from python.capture.abd_command import next_page, NEXT_PAGE_XY
        client = FakeClient()
        next_page(client)
        self.assertEqual(client.calls, [("tap", NEXT_PAGE_XY[0], NEXT_PAGE_XY[1])])

    def test_swipe_page_calls_swipe_with_expected_coords_and_default_duration(self):
        from python.capture.abd_command import swipe_page, SWIPE_COORDS
        client = FakeClient()
        swipe_page(client)
        x1, y1, x2, y2 = SWIPE_COORDS
        self.assertEqual(client.calls, [("swipe", x1, y1, x2, y2, 500)])

    def test_select_rank_1_calls_tap_with_expected_coords(self):
        from python.capture.abd_command import select_rank_1, RANK_1_XY
        client = FakeClient()
        select_rank_1(client)
        self.assertEqual(client.calls, [("tap", RANK_1_XY[0], RANK_1_XY[1])])


if __name__ == '__main__':
    unittest.main()