import os
import time
import re
import hashlib
from typing import Optional, Tuple, Dict

import cv2
import numpy as np
import easyocr


def _hash_file(path: str) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def is_bottom_blank(image_path: str, options: Optional[Dict] = None) -> Tuple[bool, Dict]:
    """
    检测截图底部是否存在大量空白：
    - 统计灰度均值、方差、白像素比例
    - 计算拉普拉斯边缘密度
    返回 (isBlank, stats)
    """
    options = options or {}
    band_ratio = float(options.get('bandRatio', 0.4))
    left_crop_ratio = float(options.get('leftCropRatio', 0.2))
    right_crop_ratio = float(options.get('rightCropRatio', 0.2))
    brightness_threshold = int(options.get('brightnessThreshold', 220))
    white_ratio_threshold = float(options.get('whiteRatioThreshold', 0.60))
    std_threshold = float(options.get('stdThreshold', 25))
    edge_pixel_threshold = int(options.get('edgePixelThreshold', 25))
    edge_density_threshold = float(options.get('edgeDensityThreshold', 0.015))

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f'无法读取图片: {image_path}')
    h, w = img.shape[:2]
    top = max(0, int(h * (1 - band_ratio)))
    left = int(w * left_crop_ratio)
    width = max(1, int(w * (1 - left_crop_ratio - right_crop_ratio)))
    height = max(1, h - top)

    roi = img[top:top + height, left:left + width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 统计
    mean = float(np.mean(gray))
    std = float(np.std(gray))
    white_ratio = float(np.mean(gray >= brightness_threshold))

    # 拉普拉斯近似边缘
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    edge = np.abs(lap)
    edge_density = float(np.mean(edge >= edge_pixel_threshold))

    is_blank = white_ratio >= white_ratio_threshold or (
        mean >= brightness_threshold and std <= std_threshold and edge_density <= edge_density_threshold
    )
    stats = {
        'whiteRatio': white_ratio,
        'mean': mean,
        'std': std,
        'edgeDensity': edge_density,
    }
    return is_blank, stats


def detect_page_number(image_path: str, reader: easyocr.Reader, options: Optional[Dict] = None) -> Optional[Dict]:
    """
    识别页码（形如 "7/7"），裁剪底部右侧区域，用 easyocr 做 OCR。
    返回 {current, total, raw} 或 None。
    """
    options = options or {}
    band_ratio = float(options.get('bandRatio', 0.2))
    right_crop_ratio = float(options.get('rightCropRatio', 0.35))
    left_margin_ratio = float(options.get('leftMarginRatio', 0.45))

    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    top = max(0, int(h * (1 - band_ratio)))
    left = int(w * left_margin_ratio)
    width = max(1, int(w * right_crop_ratio))
    height = max(1, h - top)

    roi = img[top:top + height, left:left + width]
    # 适度增强对比与尺寸
    roi = cv2.resize(roi, (width, max(80, height)), interpolation=cv2.INTER_LINEAR)
    roi = cv2.normalize(roi, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    results = reader.readtext(roi, detail=0, paragraph=True)
    text = ' '.join([t.strip() for t in results if t.strip()])
    m = re.search(r'(\d+)\s*/\s*(\d+)', text)
    if not m:
        return None
    current = int(m.group(1))
    total = int(m.group(2))
    return {'current': current, 'total': total, 'raw': text}


def run_capture(
    client,
    reader: easyocr.Reader,
    *,
    device_id: Optional[str],
    mode: str,
    prefix: str,
    interval: int,
    next_xy: Optional[Tuple[int, int]],
    prev_xy: Optional[Tuple[int, int]],
    swipe: Optional[Tuple[int, int, int, int, int]],
    count: int,
    keyevent: Optional[int],
    tap_xy: Optional[Tuple[int, int]]
):
    if not client:
        raise RuntimeError('缺少 ADB 客户端')

    def pad(n: int) -> str:
        return str(n).zfill(4)

    def sleep(ms: int):
        time.sleep(ms / 1000.0)

    if prefix and not os.path.exists(prefix):
        os.makedirs(prefix, exist_ok=True)

    # 获取屏幕尺寸用于默认参数
    size = client.get_screen_size()

    # 默认 swipe 参数计算
    if mode == 'swipe' and (not swipe or len(swipe) == 0):
        print('未提供滑动参数，自动计算...')
        if not size:
            raise RuntimeError('无法获取屏幕尺寸，请手动提供 --swipe 参数')
        x = int(size['width'] / 2)
        y1 = int(size['height'] * 0.80)
        y2 = int(size['height'] * 0.25)
        swipe = (x, y1, x, y2, 500)
        print(f'使用默认滑动参数: 从 {x},{y1} 到 {x},{y2}')

    # 默认 keyevent 与 tap-xy 设置，以便精简 CLI
    if mode == 'keyevent' and keyevent is None:
        keyevent = 93  # PAGE_DOWN 默认
        print('未提供 keyevent，使用默认 PAGE_DOWN(93)')
    if mode == 'tap' and (not tap_xy or len(tap_xy) != 2):
        if not size:
            raise RuntimeError('无法获取屏幕尺寸，请手动提供 --tap-xy 坐标')
        tap_xy = (int(size['width'] * 0.5), int(size['height'] * 0.85))
        print(f'未提供 tap-xy，使用默认底部中间坐标: {tap_xy[0]},{tap_xy[1]}')

    # 默认翻页按钮坐标（底部左右角），便于“收敛为默认值”
    if not next_xy and size:
        next_xy = (int(size['width'] * 0.92), int(size['height'] * 0.90))
        print(f'未提供下一页坐标，使用默认值: {next_xy[0]},{next_xy[1]}')
    if not prev_xy and size:
        prev_xy = (int(size['width'] * 0.08), int(size['height'] * 0.90))
        print(f'未提供上一页坐标，使用默认值: {prev_xy[0]},{prev_xy[1]}')

    # 准备阶段：识别当前页并回到第一页
    def goto_first_page() -> int:
        first_preview = os.path.join(prefix, f'{prefix}_prep_preview.png') if prefix else 'screenshot_prep_preview.png'
        client.screencap_to_file(first_preview)
        info = None
        try:
            info = detect_page_number(first_preview, reader)
        except Exception:
            info = None
        if info:
            print(f'准备阶段：当前页 {info["current"]}/{info["total"]}')
        else:
            print('准备阶段：未识别到页码，尝试直接进入执行阶段')

        if not prev_xy or len(prev_xy) != 2:
            return info['current'] if info else 1

        safety = 0
        no_change_count = 0
        last_hash = _hash_file(first_preview)
        while info and info['current'] > 1:
            safety += 1
            if safety > 50:
                print('准备阶段：达到安全上限 50 次，停止回到第一页尝试')
                break
            px, py = prev_xy
            print(f'准备阶段：点击上一页 {px},{py}')
            client.tap(px, py)
            if interval > 0:
                sleep(interval)
            client.screencap_to_file(first_preview)
            cur_hash = _hash_file(first_preview)
            if cur_hash == last_hash:
                no_change_count += 1
                print(f'准备阶段：点击上一页后页面未变化 ({no_change_count}/2)')
                if no_change_count >= 2:
                    print('准备阶段：页面无变化，判定已在第一页')
                    break
            else:
                no_change_count = 0
            last_hash = cur_hash
            try:
                info = detect_page_number(first_preview, reader)
            except Exception:
                info = None
            if info:
                print(f'准备阶段：翻至 {info["current"]}/{info["total"]}')
        try:
            os.remove(first_preview)
        except Exception:
            pass
        return info['current'] if info else 1

    start_page = goto_first_page()
    page_index = start_page
    last_page_hash = None
    retry_count = 0

    while True:
        last_scroll_hash = None
        print(f"\n=== 开始采集第 {page_index} 页 ===")
        last_shot_filename = None
        ocr_page = None
        bottom_reached = False

        for i in range(count):
            base_prefix = os.path.join(prefix, prefix) if prefix else 'screenshot'
            filename = f"{base_prefix}_p{page_index}_{pad(i)}.png"
            client.screencap_to_file(filename)
            print(f"[Page {page_index}, {i + 1}/{count}] {filename}")
            last_shot_filename = filename

            if i == 0:
                try:
                    ocr_page = detect_page_number(filename, reader)
                    if ocr_page:
                        print(f"OCR 页码识别：{ocr_page['current']}/{ocr_page['total']} (raw=\"{ocr_page['raw']}\")")
                    else:
                        print('OCR 页码识别失败或未匹配到形如 n/m 的文本')
                except Exception as e:
                    print('OCR 页码识别异常：', e)

            current_hash = _hash_file(filename)
            if i > 0 and current_hash == last_scroll_hash:
                print('滚动到底部，截图未变化。')
                bottom_reached = True
                break
            last_scroll_hash = current_hash

            if i < count - 1:
                print(f'执行滑动操作 ({i + 1} -> {i + 2})')
                if mode == 'swipe':
                    x1, y1, x2, y2, dur = swipe
                    client.swipe(x1, y1, x2, y2, dur)
                elif mode == 'keyevent' and keyevent is not None:
                    client.keyevent(keyevent)
                elif mode == 'tap' and tap_xy:
                    tx, ty = tap_xy
                    client.tap(tx, ty)
                print('滑动后暂停1秒...')
                sleep(1000)
            else:
                print('到达本页最后一张截图，不执行滑动')

            if interval > 0:
                sleep(interval)

        # 底部空白检测
        blank_at_tail = False
        if last_shot_filename:
            try:
                is_blank, stats = is_bottom_blank(last_shot_filename)
                print(f"底部空白检测: blank={is_blank} (whiteRatio={stats['whiteRatio']:.3f}, mean={stats['mean']:.1f}, std={stats['std']:.1f})")
                blank_at_tail = is_blank
            except Exception as e:
                print('底部空白检测失败，继续按翻页逻辑处理。错误:', e)

        # 末页处理
        if ocr_page and ocr_page['current'] == ocr_page['total']:
            print(f"当前为最后一页 {ocr_page['current']}/{ocr_page['total']}，继续本页到底后结束，不再翻页。")
            if bottom_reached or blank_at_tail:
                print('检测到最后一页最底，采集结束。')
                break

            base_prefix = os.path.join(prefix, prefix) if prefix else 'screenshot'
            # 继续补滑直到稳定
            def next_index_from(fn: str) -> int:
                m = re.search(r"_(\d{4})\.png$", fn)
                return int(m.group(1)) + 1 if m else (count or 10)
            idx = next_index_from(last_shot_filename) if last_shot_filename else (count or 10)
            attempts = 0
            while attempts < 8:
                attempts += 1
                print(f'末页补滑第 {attempts} 次')
                if mode == 'swipe':
                    x1, y1, x2, y2, dur = swipe
                    client.swipe(x1, y1, x2, y2, dur)
                elif mode == 'keyevent' and keyevent is not None:
                    client.keyevent(keyevent)
                elif mode == 'tap' and tap_xy:
                    tx, ty = tap_xy
                    client.tap(tx, ty)
                print('滑动后暂停1秒...')
                sleep(1000)
                fn = f"{base_prefix}_p{page_index}_{pad(idx)}.png"
                client.screencap_to_file(fn)
                h = _hash_file(fn)
                if h == last_scroll_hash:
                    print('末页补滑：截图未变化，确认已到底。')
                    bottom_reached = True
                    last_shot_filename = fn
                    break
                last_scroll_hash = h
                try:
                    is_blank, _ = is_bottom_blank(fn)
                    if is_blank:
                        print('末页补滑：检测到底部空白，确认已到底。')
                        bottom_reached = True
                        last_shot_filename = fn
                        break
                except Exception:
                    pass
                idx += 1
                if interval > 0:
                    sleep(interval)
            print('检测到最后一页最底，采集结束。' if bottom_reached else '末页补滑达到上限仍可滑动，出于安全将结束本次采集。')
            break

        # 翻页
        if next_xy:
            x, y = next_xy
            print(f'执行翻页操作，点击坐标: {x},{y}')
            client.tap(x, y)
            sleep(interval * 2)
            after_preview = os.path.join(prefix, f'{prefix}_after_next_preview.png') if prefix else 'screenshot_after_next_preview.png'
            client.screencap_to_file(after_preview)
            current_page_hash = _hash_file(after_preview)
            if current_page_hash == last_page_hash:
                retry_count += 1
                print(f'翻页后截图未变化，重试次数: {retry_count}')
                if retry_count >= 3:
                    print('到达末页，采集结束。')
                    break
            else:
                retry_count = 0
                print(f'成功翻页到第 {page_index + 1} 页')
            last_page_hash = current_page_hash
            try:
                os.remove(after_preview)
            except Exception:
                pass
            page_index += 1
        else:
            print('没有翻页操作，采集结束。')
            break

    print('\n=== 采集完成 ===')