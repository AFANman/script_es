import time
import json
import os
import argparse
from typing import List, Dict, Tuple

import easyocr
from python.capture.adb_client import AdbClient
from python.capture import abd_command
from python.parse.activity_parser import parse_rank_page, parse_bottom_start_rank

# 关键排名锚点
ANCHOR_RANKS = [1, 100, 1000, 5000, 10000, 50000]

# 每页固定显示数量
ITEMS_PER_PAGE = 20

# 目标积分
TARGET_SCORES = [
    22000000,
    15000000,
    11000000,
    7500000,
    3500000
]

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_anchors(client: AdbClient, reader: easyocr.Reader, out_dir: str) -> Dict[int, int]:
    """
    获取关键排名对应的积分
    """
    anchors = {}
    ensure_dir(out_dir)
    
    # 映射排名到点击函数
    rank_funcs = {
        1: abd_command.select_rank_1,
        100: abd_command.select_rank_100,
        1000: abd_command.select_rank_1000,
        5000: abd_command.select_rank_5000,
        10000: abd_command.select_rank_10000,
        50000: abd_command.select_rank_50000
    }

    print(">>> 开始采集关键排名锚点...")
    for rank in ANCHOR_RANKS:
        if rank not in rank_funcs:
            print(f"警告: 未定义的排名点击函数: {rank}")
            continue
            
        print(f"正在跳转到排名: {rank}")
        rank_funcs[rank](client)
        time.sleep(2.5)  # 等待加载
        
        screenshot_path = os.path.join(out_dir, f"rank_{rank}.png")
        client.screencap_to_file(screenshot_path)
        print(f"截图已保存: {screenshot_path}")
        
        pairs = parse_rank_page(reader, screenshot_path)
        if not pairs:
            print(f"警告: 无法解析排名 {rank} 的页面")
            continue
            
        # 找到最接近目标 rank 的条目
        best_pair = None
        min_diff = float('inf')
        for p in pairs:
            diff = abs(p['rank'] - rank)
            if diff < min_diff:
                min_diff = diff
                best_pair = p
        
        if best_pair:
            print(f"解析成功: 排名 {best_pair['rank']} -> 积分 {best_pair['score']}")
            anchors[best_pair['rank']] = best_pair['score']
            # 将页面上其他数据也作为参考点
            for p in pairs:
                anchors[p['rank']] = p['score']
        else:
            print(f"警告: 页面中未找到接近 {rank} 的排名数据")

    return anchors

def skip_pages(client: AdbClient, direction: str, pages: int):
    """
    快速翻页，不进行截图和识别。
    direction: 'next' or 'prev'
    """
    if pages <= 0: return
    
    print(f"*** 快速跳过 {pages} 页 ({direction}) ***")
    for i in range(pages):
        if direction == 'next':
            abd_command.next_page(client)
        else:
            abd_command.previous_page(client)
        # 快速翻页间隔可以短一点，但要保证系统响应
        time.sleep(0.8)
    # 最后多等一下确保加载
    time.sleep(2.0)

def estimate_rank_from_anchors(target_score: int, anchors: Dict[int, int]) -> int:
    """
    根据锚点数据，使用分段线性插值估算目标积分对应的排名。
    """
    sorted_ranks = sorted(anchors.keys())
    if not sorted_ranks:
        return 1
        
    # 1. 如果超出范围，使用最近的区间的斜率外推
    # (由于积分随排名增加而减少，我们找区间 [r1, r2] 使得 score[r1] >= target >= score[r2])
    
    # 转换为 (rank, score) 列表
    points = [(r, anchors[r]) for r in sorted_ranks]
    
    # 寻找包含 target_score 的区间
    for i in range(len(points) - 1):
        r1, s1 = points[i]
        r2, s2 = points[i+1]
        
        # 理想情况：s1 >= target >= s2 (降序)
        # 但数据可能不完全单调(极少情况)，这里假设是严格降序或允许小波动
        if s1 >= target_score >= s2:
            # 插值
            if s1 == s2: return r1
            ratio = (s1 - target_score) / (s1 - s2)
            return int(r1 + ratio * (r2 - r1))
            
    # 如果没找到区间，说明在两端
    # Case A: target > max_score (rank < min_rank)
    if target_score > points[0][1]:
        # 使用前两个点外推 (如果只有一个点，无法外推，返回该点rank)
        if len(points) > 1:
            r1, s1 = points[0]
            r2, s2 = points[1]
            slope = (s2 - s1) / (r2 - r1) # 应该是负数
            if slope == 0: return r1
            return int(r1 + (target_score - s1) / slope)
        else:
            return points[0][0]
            
    # Case B: target < min_score (rank > max_rank)
    if target_score < points[-1][1]:
        if len(points) > 1:
            r1, s1 = points[-2]
            r2, s2 = points[-1]
            slope = (s2 - s1) / (r2 - r1)
            if slope == 0: return r2
            return int(r2 + (target_score - s2) / slope)
        else:
            return points[-1][0]
            
    return 1

def find_target_score(client: AdbClient, reader: easyocr.Reader, target_score: int, anchors: Dict[int, int], out_dir: str) -> Dict:
    """
    查找目标积分对应的排名
    """
    print(f"\n>>> 开始查找目标积分: {target_score:,}")
    
    # 1. 选择最佳起点 (绝对距离最近的锚点)
    valid_jumps = [1, 100, 1000, 5000, 10000, 50000]
    
    # 优化：选择离目标积分最近的锚点
    best_jump_rank = 1
    min_score_diff = float('inf')
    
    for jr in valid_jumps:
        if jr in anchors:
            score = anchors[jr]
            diff = abs(score - target_score)
            if diff < min_score_diff:
                min_score_diff = diff
                best_jump_rank = jr
                
    print(f"执行跳转到最近锚点: {best_jump_rank} (积分: {anchors.get(best_jump_rank, 'unknown')})")
    
    if best_jump_rank == 1: abd_command.select_rank_1(client)
    elif best_jump_rank == 100: abd_command.select_rank_100(client)
    elif best_jump_rank == 1000: abd_command.select_rank_1000(client)
    elif best_jump_rank == 5000: abd_command.select_rank_5000(client)
    elif best_jump_rank == 10000: abd_command.select_rank_10000(client)
    elif best_jump_rank == 50000: abd_command.select_rank_50000(client)
    time.sleep(2.5)
    
    # 搜索循环
    max_pages = 200
    
    # 差值估算相关状态
    last_page_avg_score = anchors.get(best_jump_rank, 0)
    score_diffs = [] # 记录每页的积分差值
    items_per_page = ITEMS_PER_PAGE # 固定为20
    just_skipped = False # 标记是否刚进行过跳页
    
    # 预估目标排名
    estimated_target_rank = estimate_rank_from_anchors(target_score, anchors)
    print(f"根据锚点预估目标排名约为: {estimated_target_rank}")
    
    start_time = time.time()
    
    for i in range(max_pages):
        # 识别底部起始排名 (用于精确翻页计算)
        current_bottom_rank = None
        
        # --- 页内滑动搜索逻辑 Start ---
        page_pairs = []
        last_scroll_min_s = None
        
        # 最多滑动 8 次
        for scroll_idx in range(8):
            current_elapsed = time.time() - start_time
            log_prefix = f"[Target: {target_score:,} | {current_elapsed:.1f}s]"
            
            screenshot_path = os.path.join(out_dir, f"search_{target_score}_{i}_{scroll_idx}.png")
            client.screencap_to_file(screenshot_path)
            
            # 尝试识别底部排名 (仅在第一次滑动或未获取到时识别)
            if current_bottom_rank is None:
                 current_bottom_rank = parse_bottom_start_rank(reader, screenshot_path)
                 if current_bottom_rank:
                     print(f"{log_prefix} 识别到底部起始排名: {current_bottom_rank}")
            
            pairs = parse_rank_page(reader, screenshot_path)
            if not pairs:
                print(f"{log_prefix} 解析页面失败，重试...")
                time.sleep(1)
                continue
                
            # 添加到当前页结果
            for p in pairs:
                if not any(existing['rank'] == p['rank'] for existing in page_pairs):
                    page_pairs.append(p)
            
            all_scores = [p['score'] for p in page_pairs]
            if not all_scores: continue
            
            # 过滤异常值：剔除与中位数差异过大的噪声 (e.g. 识别成 5622 而实际是 350w)
            if len(all_scores) >= 3:
                sorted_s = sorted(all_scores)
                median_s = sorted_s[len(sorted_s) // 2]
                # 阈值设定：同一页内积分通常不会跌破中位数的 20% (应对极端的断层)
                # 同时过滤掉过大的离群值 (虽然较少见)
                valid_pairs = []
                for p in page_pairs:
                    if 0.2 * median_s < p['score'] < 5.0 * median_s:
                        valid_pairs.append(p)
                    else:
                        print(f"{log_prefix} 剔除异常积分: {p['score']:,} (Rank: {p['rank']}) | Median: {median_s:,}")
                
                page_pairs = valid_pairs
                all_scores = [p['score'] for p in page_pairs]
                if not all_scores: continue

            max_s = max(all_scores)
            min_s = min(all_scores)
            
            print(f"{log_prefix} Page {i} Scroll {scroll_idx}: Range [{min_s:,} - {max_s:,}]")
            
            # 到底检测：如果当前最小积分没有比上一次更小，说明没滑出新数据（到底了）
            if last_scroll_min_s is not None and min_s >= last_scroll_min_s:
                print(f"{log_prefix} 积分未更新(Min: {min_s:,})，判定为本页到底。")
                break
            last_scroll_min_s = min_s
            
            # 1. 精确命中
            if min_s <= target_score <= max_s:
                best_p = min(page_pairs, key=lambda p: abs(p['score'] - target_score))
                print(f"{log_prefix} 找到最接近目标: 排名 {best_p['rank']}, 积分 {best_p['score']:,}")
                return {'target': target_score, 'rank': best_p['rank'], 'actual_score': best_p['score'], 'elapsed': current_elapsed}
            
            # 2. 页内滑动决策
            if min_s > target_score:
                print(f"{log_prefix} 当前最小积分仍高于目标，尝试页内下滑...")
                abd_command.scroll_list_down(client)
                time.sleep(1.5)
            else:
                break
        
        # --- 页内滑动搜索逻辑 End ---
        
        current_elapsed = time.time() - start_time
        log_prefix = f"[Target: {target_score:,} | {current_elapsed:.1f}s]"
        
        if not page_pairs:
             print(f"{log_prefix} 本页无有效数据，尝试翻页...")
             abd_command.next_page(client)
             time.sleep(2.0)
             continue

        all_scores = [p['score'] for p in page_pairs]
        max_s = max(all_scores)
        min_s = min(all_scores)
        
        # 更新 items_per_page (如果当前页抓取到的数量比较合理)
        # if len(page_pairs) > 5:
        #     items_per_page = len(page_pairs)
        
        # 计算当前页平均分，用于更新差值
        curr_page_avg = sum(all_scores) / len(all_scores)
        
        # 更新页间差值 (只在非第一页计算)
        # 如果刚跳过页，或者第一次进入循环，diff 不可靠（包含跳页的距离），不记录
        if i > 0 and last_page_avg_score > 0 and not just_skipped:
            diff = abs(curr_page_avg - last_page_avg_score)
            if diff > 0:
                score_diffs.append(diff)
                if len(score_diffs) > 5: score_diffs.pop(0)
        
        last_page_avg_score = curr_page_avg
        just_skipped = False # 重置标记
        
        # 估算平均差值
        avg_diff = sum(score_diffs) / len(score_diffs) if score_diffs else 0
        
        print(f"{log_prefix} Page {i} Final Range: [{min_s:,} - {max_s:,}] | Avg Diff: {int(avg_diff):,}")
        
        if min_s <= target_score <= max_s:
             best_p = min(page_pairs, key=lambda p: abs(p['score'] - target_score))
             return {'target': target_score, 'rank': best_p['rank'], 'actual_score': best_p['score'], 'elapsed': current_elapsed}
             
        # 翻页决策与快速跳页
        if current_bottom_rank:
            current_rank_center = current_bottom_rank + (ITEMS_PER_PAGE // 2)
            print(f"{log_prefix} 使用底部排名定位: {current_bottom_rank} (Center ~{current_rank_center})")
        else:
            current_rank_center = page_pairs[len(page_pairs)//2]['rank'] if page_pairs else estimated_target_rank
            print(f"{log_prefix} 使用列表识别定位: Center ~{current_rank_center}")
        
        # 计算需要的翻页数
        rank_dist = estimated_target_rank - current_rank_center
        est_pages_by_rank = int(abs(rank_dist) / items_per_page)
        
        if min_s > target_score:
            # 需要向后翻 (Next)
            dist = min_s - target_score
            
            # 混合决策: 默认用Rank估算
            est_pages = est_pages_by_rank
            
            if avg_diff > 0:
                 est_pages_by_score = int(dist / avg_diff)
                 # 如果有底部精确排名，且排名估算的页数和积分估算差异很大，
                 # 且积分距离较远，此时优先信任排名（因为它是绝对坐标）
                 # 除非积分估算更保守（防止跳过头）
                 if current_bottom_rank and abs(est_pages_by_rank - est_pages_by_score) > 5:
                      print(f"{log_prefix} 排名估算({est_pages_by_rank})与积分估算({est_pages_by_score})差异大，优先信任排名估算(Base: {current_bottom_rank})")
                      est_pages = est_pages_by_rank
                 else:
                      est_pages = est_pages_by_score
            
            # 双重检查：如果距离很近，不要跳太多
            if dist < 100000: # 10w 分以内通常几页就到了
                est_pages = min(est_pages, 3)
            
            if est_pages > 2:
                # 动态调整跳页策略
                skip = int(est_pages * 0.8)
                skip = max(skip, 1)
                skip = min(skip, 30)
                
                print(f"{log_prefix} 距离目标 {dist:,} (Rank Dist ~{rank_dist})，预估需 {est_pages} 页，执行快速跳过 {skip} 页 ->")
                skip_pages(client, 'next', skip)
                just_skipped = True
                continue
            
            print(f"{log_prefix} 本页积分过高，向后翻页 (Next Page) ->")
            abd_command.next_page(client)
            time.sleep(2.0)
            
        elif max_s < target_score:
            # 需要向前翻 (Prev)
            dist = target_score - max_s
            
            est_pages = est_pages_by_rank
            if avg_diff > 0:
                 est_pages_by_score = int(dist / avg_diff)
                 if current_bottom_rank and abs(est_pages_by_rank - est_pages_by_score) > 5:
                      print(f"{log_prefix} 排名估算({est_pages_by_rank})与积分估算({est_pages_by_score})差异大，优先信任排名估算(Base: {current_bottom_rank})")
                      est_pages = est_pages_by_rank
                 else:
                      est_pages = est_pages_by_score
                 
            if dist < 100000:
                est_pages = min(est_pages, 3)

            if est_pages > 2:
                skip = int(est_pages * 0.8)
                skip = max(skip, 1)
                skip = min(skip, 30)
                print(f"{log_prefix} 距离目标 {dist:,} (Rank Dist ~{rank_dist})，预估需 {est_pages} 页，执行快速跳过 {skip} 页 <-")
                skip_pages(client, 'prev', skip)
                just_skipped = True
                continue
                    
            print(f"{log_prefix} 本页积分过低，向前翻页 (Prev Page) <-")
            abd_command.previous_page(client)
            time.sleep(2.0)
            
    print(f"[Target: {target_score:,}] 达到最大翻页次数，未找到精确匹配。")
    return {'target': target_score, 'error': 'Not found within limit', 'elapsed': time.time() - start_time}

def main():
    parser = argparse.ArgumentParser(description='偶像梦幻祭2 排名抓取')
    parser.add_argument('--device', help='ADB Device ID')
    parser.add_argument('--out', default='rank_data', help='输出目录')
    args = parser.parse_args()
    
    ensure_dir(args.out)
    
    client = AdbClient(device_id=args.device)
    client.connect()
    
    print("初始化 OCR...")
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    
    # 1. 获取锚点数据
    anchor_file = os.path.join(args.out, 'anchors.json')
    if os.path.exists(anchor_file):
        print("发现已有锚点数据，跳过采集。")
        with open(anchor_file, 'r', encoding='utf-8') as f:
            anchors = json.load(f)
            # JSON key 是 str，需要转回 int
            anchors = {int(k): v for k, v in anchors.items()}
    else:
        anchors = get_anchors(client, reader, os.path.join(args.out, 'anchors'))
        
        # 保存锚点数据
        with open(anchor_file, 'w', encoding='utf-8') as f:
            json.dump(anchors, f, indent=2)
        
    # 2. 查找目标积分
    results = []
    for target in TARGET_SCORES:
        res = find_target_score(client, reader, target, anchors, os.path.join(args.out, f'search_{target}'))
        results.append(res)
        
    # 3. 输出最终 JSON
    final_data = {
        'anchors': anchors,
        'targets': results
    }
    
    with open(os.path.join(args.out, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
        
    print("\n全部任务完成！结果已保存至 result.json")

if __name__ == '__main__':
    main()
