import re
from typing import List, Dict, Tuple

import easyocr
import numpy as np
import cv2


def _bbox_min_xy(bbox: List[Tuple[float, float]]) -> Tuple[float, float]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return float(min(xs)), float(min(ys))


def _bbox_max_xy(bbox: List[Tuple[float, float]]) -> Tuple[float, float]:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return float(max(xs)), float(max(ys))


def _is_noise_text(text: str) -> bool:
    """
    判断文本是否为噪声：
    - 纯数字（如页码、计数）
    - 进度分数形态，如 "1/7"、"12/100"
    - 仅符号或长度过短且不含中文/英文
    """
    t = (text or '').strip()
    if not t:
        return True
    # 进度分数样式（如 1/7）
    if re.fullmatch(r"\d+\s*/\s*\d+", t):
        return True
    # 纯数字
    if re.fullmatch(r"\d+", t):
        return True
    # 带千分位的纯数字（如 80,000）
    if re.fullmatch(r"\d{1,3}(?:,\d{3})+", t):
        return True
    # 仅符号
    if re.fullmatch(r"[^\w\u4e00-\u9fff]+", t):
        return True
    # 长度过短且不含中文或英文
    if len(t) < 2 and not re.search(r"[\u4e00-\u9fffA-Za-z]", t):
        return True
    # 短大写英文噪声
    if re.fullmatch(r"[A-Z]{1,3}", t):
        return True
    # 常见 UI 标签干扰词
    if re.search(r"活动积分|下个奖|下一个奖|下一奖励|还需", t):
        return True
    return False


def parse_image_items(reader: easyocr.Reader, image_path: str, read_options: Dict | None = None) -> List[Dict]:
    """
    解析单张截图，返回 [{score, item}]

    - 使用 easyocr 提取文本与 bbox
    - 按 y 聚合为行，行内按 x 排序
    - 在右半区域识别分数（1000 的倍数），行左侧合并为道具描述
    """
    read_options = read_options or {}
    # 默认小 batch + 单线程，避免 DataLoader 线程在部分 Windows/Python 组合下阻塞
    batch_size = int(read_options.get('batch_size', 4))
    workers = int(read_options.get('workers', 0))
    contrast_ths = read_options.get('contrast_ths', None)
    adjust_contrast = read_options.get('adjust_contrast', None)
    # 处理 ROI 裁剪（比例 0-1 或像素）
    roi = read_options.get('roi') if isinstance(read_options, dict) else None
    img_w = None
    img_to_read = image_path
    if roi and len(roi) == 4:
        try:
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                l, t, r, b = roi
                if max(roi) <= 1.0:  # 按比例
                    L = int(l * w)
                    T = int(t * h)
                    R = int(r * w)
                    B = int(b * h)
                else:  # 像素
                    L, T, R, B = int(l), int(t), int(r), int(b)
                L = max(0, min(L, w - 1))
                R = max(L + 1, min(R, w))
                T = max(0, min(T, h - 1))
                B = max(T + 1, min(B, h))
                crop = img[T:B, L:R]
                if crop is not None and crop.size > 0:
                    img_to_read = crop
                    img_w = R - L
        except Exception:
            # 若裁剪异常则退回整图
            img_to_read = image_path
            img_w = None

    kwargs = {
        'detail': 1,
        'paragraph': False,
        'batch_size': batch_size,
        'workers': workers,
    }
    if isinstance(contrast_ths, (int, float)):
        kwargs['contrast_ths'] = float(contrast_ths)
    if isinstance(adjust_contrast, (int, float)):
        kwargs['adjust_contrast'] = float(adjust_contrast)
    results = reader.readtext(img_to_read, **kwargs)
    if not results:
        return []

    words = []
    # 结果格式: [bbox, text, conf]
    for bbox, text, conf in results:
        t = (text or '').strip()
        if not t:
            continue
        x0, y0 = _bbox_min_xy(bbox)
        x1, y1 = _bbox_max_xy(bbox)
        words.append({
            'text': t,
            'x0': x0,
            'y0': y0,
            'x1': x1,
            'y1': y1,
            'conf': conf,
        })

    if not words:
        return []

    # 估计图像宽度（ROI 裁剪宽度或由 bbox 最大 x 值近似）
    width = img_w if img_w is not None else max(int(w['x1']) for w in words)

    # 调试输出
    if read_options.get('debug'):
        print(f'[DEBUG] tokens={len(words)} width={width}')
        for w in words[:50]:
            print(f"[DEBUG] '{w['text']}' x0={w['x0']:.1f} y0={w['y0']:.1f} x1={w['x1']:.1f} y1={w['y1']:.1f} conf={w['conf']:.2f}")

    # 改为基于分数字段的行对齐：对每个分数字段，在其垂直窗口内收集左侧文本
    right_boundary = width * 0.55
    score_regex = re.compile(r'^(\d{1,3}(?:,\d{3})+|\d{3,})$')

    # 找出所有分数字段
    scores = [w for w in words if w['x0'] >= right_boundary and score_regex.match(w['text'])]
    # 按纵向位置排序，以保持自上而下的顺序
    scores.sort(key=lambda w: (w['y0'], w['x0']))

    # 垂直对齐窗口（像素）；在多分辨率下适度放宽
    y_window = 90
    left_min_x = width * 0.30

    pairs = []
    for score_word in scores:
        score = int(score_word['text'].replace(',', ''))
        if score % 1000 != 0:
            continue

        cy = (score_word['y0'] + score_word['y1']) / 2.0
        # 收集左侧候选并按 x 排序
        candidates = [w for w in words if w['x1'] < score_word['x0'] and w['x0'] >= left_min_x and abs(((w['y0'] + w['y1']) / 2.0) - cy) <= y_window and w['conf'] >= 0.15]
        candidates.sort(key=lambda w: w['x0'])

        left_texts = []
        prev = ""
        for w in candidates:
            txt = (w['text'] or '').strip()
            # 统一乘号
            txt = txt.replace('×', 'x').replace('X', 'x').replace('*', 'x')
            # 将“...xN”尾缀与文字分开："PAIRINGx1" => "PAIRING x1"
            txt = re.sub(r'([\w\u4e00-\u9fff\)\]]+)x(\d+)$', r'\1 x\2', txt)
            # 保留数量：如果当前是纯数字且前一个是 x，则保留
            if re.fullmatch(r"\d+", txt) and prev.lower() == 'x':
                left_texts.append(txt)
                prev = txt
                continue
            # 常规噪声过滤（含千分位纯数字、比值等）
            if _is_noise_text(txt):
                continue
            left_texts.append(txt)
            prev = txt

        item = ' '.join(left_texts)
        item = re.sub(r'\s+', ' ', item).strip()
        if not item:
            continue
        pairs.append({'score': score, 'item': item})

    return pairs


def parse_directory(reader: easyocr.Reader, dir_path: str, max_score: int = 100000000, read_options: Dict | None = None) -> Dict:
    import os
    files = [f for f in os.listdir(dir_path) if f.lower().endswith('.png')]
    files.sort(key=lambda x: x.lower())

    score_map = {}
    for f in files:
        path = os.path.join(dir_path, f)
        pairs = parse_image_items(reader, path, read_options=read_options)
        for p in pairs:
            s = p['score']
            if s not in score_map or len(p['item']) > len(score_map[s]):
                score_map[s] = p['item']

    items = [{'score': s, 'item': score_map[s]} for s in sorted(score_map.keys())]

    missing = []
    for s in range(1000, max_score + 1, 1000):
        if s not in score_map:
            missing.append(s)

    return {'items': items, 'missing': missing}


def save_json(out_path: str, data: Dict):
    import json
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_rank_page(reader: easyocr.Reader, image_path: str, read_options: Dict | None = None) -> List[Dict]:
    """
    解析排名页面截图，返回 [{rank, score}]
    """
    read_options = read_options or {}
    # 简单读取，不使用复杂参数
    results = reader.readtext(image_path, detail=1, paragraph=False)
    if not results:
        return []

    words = []
    for bbox, text, conf in results:
        t = (text or '').strip()
        if not t: continue
        x0, y0 = _bbox_min_xy(bbox)
        x1, y1 = _bbox_max_xy(bbox)
        words.append({'text': t, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1, 'conf': conf})

    rank_regex = re.compile(r'^(\d+)\s*(?:位|位~|Rank)?$')
    score_regex = re.compile(r'^(\d{1,3}(?:,\d{3})+|\d{4,})$')

    ranks = []
    scores = []
    width = max(w['x1'] for w in words) if words else 1000

    for w in words:
        rm = rank_regex.match(w['text'])
        if rm and w['x0'] < width * 0.5:
            ranks.append({'val': int(rm.group(1)), 'y': (w['y0'] + w['y1']) / 2, 'obj': w})
            continue
        
        if score_regex.match(w['text']):
             s_val = int(w['text'].replace(',', ''))
             # 简单过滤：积分通常较大
             if s_val > 0:
                 scores.append({'val': s_val, 'y': (w['y0'] + w['y1']) / 2, 'obj': w})

    pairs = []
    for r in ranks:
        best_s = None
        min_dist = 100  # Y 轴距离阈值
        for s in scores:
            dist = abs(r['y'] - s['y'])
            # 积分通常在排名右侧
            if dist < min_dist and s['obj']['x0'] > r['obj']['x0']:
                min_dist = dist
                best_s = s
        
        if best_s:
            pairs.append({'rank': r['val'], 'score': best_s['val']})
    
    # 按排名排序
    pairs.sort(key=lambda x: x['rank'])
    return pairs


def parse_bottom_start_rank(reader: easyocr.Reader, image_path: str) -> int | None:
    """
    解析页面底部的起始排名（例如 "420位~" -> 420）
    直接识别底部整行，不再硬编码左右边界
    """
    img = cv2.imread(image_path)
    if img is None: return None
    
    h, w = img.shape[:2]
    # 仅截取底部 15% 的高度，宽度取全屏
    # 这样可以避开上面的列表内容，但保留整个底部栏
    t, b = int(h * 0.85), h
    
    crop = img[t:b, :]
    if crop.size == 0: return None
    
    # 预处理：转灰度 + 放大
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    results = reader.readtext(gray, detail=0)
    
    # 调试输出
    # print(f"[DEBUG] Bottom OCR raw results: {results}")
    
    # 寻找包含 "位~" 或 "位" 的文本，或者纯数字
    # 底部排名通常是页面上唯一的独立大数字或带位~的文本
    for text in results:
        clean_text = text.replace(',', '').replace(' ', '')
        
        # 优先匹配带 "位~" 或 "位" 的
        if '位' in clean_text or '~' in clean_text:
            match = re.search(r'(\d+)', clean_text)
            if match:
                return int(match.group(1))
                
    # 如果没找到带单位的，尝试找独立的纯数字（且数值较大，不是页码1/2这种）
    for text in results:
        clean_text = text.replace(',', '').replace(' ', '').replace('~', '')
        if clean_text.isdigit():
            val = int(clean_text)
            if val > 10: # 简单的过滤，假设排名通常大于10
                return val
            
    return None
