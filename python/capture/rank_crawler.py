import time
import json
import os
import argparse
import yaml  # 引入 PyYAML
import requests # 引入 requests
from typing import List, Dict, Tuple

import easyocr
from python.capture.adb_client import AdbClient
from python.capture import abd_command
from python.capture.rank_estimator import RankEstimator
from python.parse.activity_parser import parse_rank_page, parse_bottom_start_rank

# 关键排名锚点 (默认值，会被 config.json 覆盖)
DEFAULT_ANCHOR_RANKS = [1, 100, 1000, 5000, 10000, 50000]

# 每页固定显示数量
ITEMS_PER_PAGE = 20

# 目标积分 (默认值，会被 config.json 覆盖)
DEFAULT_TARGET_SCORES = [
    22000000,
    15000000,
    11000000,
    7500000,
    3500000
]

# 导出关键排名 (默认值)
DEFAULT_EXPORT_KEY_RANKS = [100, 1000, 5000, 10000]

def load_config(config_path: str = 'config.yaml') -> Dict:
    """加载配置文件 (YAML)"""
    config = {
        'anchor_ranks': DEFAULT_ANCHOR_RANKS,
        'target_scores': DEFAULT_TARGET_SCORES,
        'export_key_ranks': DEFAULT_EXPORT_KEY_RANKS
    }
    
    if os.path.exists(config_path):
        print(f"加载配置文件: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) # 使用 yaml 加载
                if user_config:
                    config.update(user_config)
        except Exception as e:
            print(f"配置文件加载失败: {e}，使用默认配置。")
    else:
        print(f"未找到配置文件 {config_path}，使用默认配置。")
        
    return config

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_anchors(client: AdbClient, reader: easyocr.Reader, out_dir: str, anchor_ranks: List[int]) -> Dict[int, int]:
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
    for rank in anchor_ranks:
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

def find_target_score(client: AdbClient, reader: easyocr.Reader, target_score: int, anchors: Dict[int, int], estimator: RankEstimator, out_dir: str) -> Dict:
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
    last_jump_direction = None # 记录上一次跳页方向 ('next' or 'prev')
    
    # 预估目标排名
    estimated_target_rank = estimate_rank_from_anchors(target_score, anchors)
    print(f"根据锚点预估目标排名约为: {estimated_target_rank}")
    
    start_time = time.time()
    
    # 记录上一次页面的最大/最小积分，用于回溯判断
    last_page_range = None 
    force_scroll = False # 标记是否需要强制下滑搜索
    
    for i in range(max_pages):
        # 识别底部起始排名 (用于精确翻页计算)
        current_bottom_rank = None
        
        # --- 页内滑动搜索逻辑 Start ---
        page_pairs = []
        last_scroll_min_s = None
        
        # 页内滑动策略：
        # 1. 如果被标记为 force_scroll (回溯回来的)，则必须下滑搜索。
        # 2. 如果是普通页面，先不滑，直接翻页去探测下一页的头部。
        #    除非：预测显示目标就在这页极短范围内(比如 < 10名)，那可以直接滑一下试试。
        
        should_scroll_now = force_scroll
        
        # 预判：如果距离非常近 (e.g. < 20名)，可以直接滑，省去一次翻页回退的开销
        if not should_scroll_now:
            # 简易预判：如果当前页已经能看到 target 在当前 min_s 附近
            # 这里需要重新计算一下本页当前的 min_s (第一屏的)
            pass 

        # 最多滑动 8 次 (只有当 should_scroll_now 为 True 时才执行)
        # 如果还没开始滑，默认先滑 0 次 (只看第一屏)，除非 force_scroll
        max_scrolls = 8
        
        # 注意：这里我们把 range 改大一点，在循环内部控制退出
        # 循环逻辑：
        # scroll_idx=0: 识别第一屏
        #   -> 计算 min_s
        #   -> 判断是否需要继续下滑 (should_scroll_now)
        #      -> 如果 min_s 离 target 极近 -> 强制开启下滑
        #      -> 如果 force_scroll -> 开启下滑
        #      -> 否则 -> break (去翻页)
        # scroll_idx > 0: 执行下滑 -> 识别 -> 循环
        
        for scroll_idx in range(max_scrolls + 1): 
            # 在滑之前判断是否需要继续
            if scroll_idx > 0 and not should_scroll_now: 
                break 
                
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
                    # 实时更新锚点数据，利用已探索的区域优化后续的排名预估
                    anchors[p['rank']] = p['score']
                    estimator.add_data(p['rank'], p['score'])
            
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

            # 按 rank 排序，确保顺序正确 (排行榜 rank 越小越靠前，score 越高)
            page_pairs.sort(key=lambda x: x['rank'])
            
            max_s = page_pairs[0]['score']   # 第一条 (Top)
            min_s = page_pairs[-1]['score']  # 最后一条 (Bottom)
            
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
                # 动态判定是否需要开启下滑 (仅在第一屏判定)
                if scroll_idx == 0 and not should_scroll_now:
                    # 强力兜底：如果距离目标极近 (例如 < 2000 分)，无论如何都要下滑
                    if (min_s - target_score) < 2000:
                        print(f"{log_prefix} [兜底] 距离目标极近 ({min_s - target_score} 分)，强制开启下滑搜索")
                        should_scroll_now = True

                if should_scroll_now:
                    print(f"{log_prefix} [强制搜索] 当前最小积分仍高于目标，继续页内下滑...")
                    abd_command.scroll_list_down(client)
                    time.sleep(1.5)
                else:
                    # 默认不下滑，跳出循环去翻页探测
                    break
            else:
                break
        
        # 重置 force_scroll
        # 关键修正：如果是 force_scroll 模式(回溯回来的)，且滑完了还没命中(min_s <= target)，
        # 说明目标正好夹在缝隙里，或者当前 min_s 就是离目标最近的(略大于目标)。
        # 此时应强制结算，不再翻页，防止死循环。
        if force_scroll:
            print(f"{log_prefix} [结算] 回溯搜索结束，取当前页最接近值 (Min: {min_s:,})")
            best_p = min(page_pairs, key=lambda p: abs(p['score'] - target_score))
            return {'target': target_score, 'rank': best_p['rank'], 'actual_score': best_p['score'], 'elapsed': current_elapsed}

        force_scroll = False 
        
        # --- 页内滑动搜索逻辑 End ---
        
        current_elapsed = time.time() - start_time
        log_prefix = f"[Target: {target_score:,} | {current_elapsed:.1f}s]"
        
        if not page_pairs:
             print(f"{log_prefix} 本页无有效数据，尝试翻页...")
             abd_command.next_page(client)
             time.sleep(2.0)
             continue

        # 按 rank 排序
        page_pairs.sort(key=lambda x: x['rank'])
        
        max_s = page_pairs[0]['score']   # 第一条 (Top)
        min_s = page_pairs[-1]['score']  # 最后一条 (Bottom)
        all_scores = [p['score'] for p in page_pairs] # 依然保留用于计算平均分
        
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
        
        # --- 回溯检测 (Backtracking) ---
        # 如果上一次我们在上一页 (PrevPage)，且当时 MinScore > Target (说明目标在后面)
        # 而现在这一页 (CurrPage) MaxScore < Target (说明目标在前面)
        # 意味着目标被夹在 [PrevPage.Min, CurrPage.Max] 中间，也就是 PrevPage 的底部未显示区域。
        # 此时应该回退到 PrevPage 并强制下滑。
        
        if last_jump_direction == 'next' and last_page_range:
             prev_min = last_page_range[0]
             prev_max = last_page_range[1] # 上一页的顶部积分
             
             # 判定夹逼条件：
             # 1. 上一页的顶部 > Target (说明目标可能在上一页)
             # 2. 本页的顶部 < Target (说明目标在当前页的前面)
             # 结论：目标一定在上一页的范围内 (中间或底部)
             if prev_max > target_score > max_s:
                 # 检查是否上一页的底部其实就是最优解
                 # 场景：Page 10 Min=15,000,009, Target=15,000,000. 
                 # Page 11 Max=14,877,153. 
                 # 目标夹在中间，但显然 15,000,009 离目标极近 (差9分)，而 14,877,153 差远了。
                 # 这时候直接判定上一页的底部为最佳匹配，不再回退。
                 if abs(prev_min - target_score) < 2000: # 阈值可调，2000分以内认为足够接近
                      print(f"{log_prefix} [命中] 上一页底部 ({prev_min:,}) 离目标极近，判定为最佳匹配。")
                      # 这里我们构造一个结果返回，虽然没有 rank 信息(除非我们缓存了)，但可以返回近似信息
                      # 或者为了获取 rank，我们还是得退回去，但这次回去是为了“确认并截图”，而不是“搜索”。
                      # 为了简单，我们还是退回去，但是带上一个标记 "confirm_and_exit"
                      pass

                 pos_desc = "底部" if prev_min > target_score else "中间"
                 print(f"{log_prefix} [回溯] 目标 ({target_score:,}) 在上一页{pos_desc} (PrevMax: {prev_max:,}, CurrMax: {max_s:,})")
                 print(f"{log_prefix} [回溯] 执行回退操作 (Prev Page)...")
                 abd_command.previous_page(client)
                 time.sleep(2.0)
                 
                 last_page_range = None 
                 last_jump_direction = 'prev'
                 force_scroll = True # 标记：回到上一页后必须强制下滑搜索
                 continue

        # 更新本页范围供下一次使用
        last_page_range = (min_s, max_s)

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
        
        # 每一轮循环都重新预估目标排名，利用新收集的锚点
        estimated_target_rank = estimate_rank_from_anchors(target_score, anchors)

        # 计算需要的翻页数
        rank_dist = estimated_target_rank - current_rank_center
        est_pages_by_rank = int(abs(rank_dist) / items_per_page)
        
        # --- 决策方向 ---
        direction = None
        if min_s > target_score:
            direction = 'next'
            dist_score = min_s - target_score
        elif max_s < target_score:
            direction = 'prev'
            dist_score = target_score - max_s
        else:
            # 理论上不会走到这里，因为前面已经判断了精确命中
            pass

        if direction:
            # 震荡检测
            is_oscillation = (last_jump_direction and last_jump_direction != direction and dist_score < 200000)
            if is_oscillation:
                 print(f"{log_prefix} 检测到震荡 (Dir: {last_jump_direction} -> {direction}, Dist: {dist_score:,})")

            # --- 方案 A: 优先尝试非线性模型 ---
            use_model = False
            skip = 0
            
            model_pred_rank = estimator.predict_rank(target_score)
            if model_pred_rank:
                # 打印详细的模型预测日志
                print(f"{log_prefix} [模型预测] 拟合度 R2={estimator.r_squared:.4f}, 样本数={len(estimator.ranks)}, 预测排名={model_pred_rank:,}")
                
                model_step, alpha, reason = estimator.get_adaptive_step(
                    current_rank_center, model_pred_rank, ITEMS_PER_PAGE, 
                    last_direction_flip=is_oscillation
                )
                
                # 只有当模型置信度较高，或者距离很近且模型还算准的时候才用
                # 如果 alpha 被严重打折(震荡或R2低)，model_step 会很小，这也是安全的
                if alpha > 0.3: 
                    # 检查方向是否一致
                    # model_step 正数代表 rank 增加 (Next)，负数代表 rank 减小 (Prev)
                    # direction 'next' 对应 rank 增加
                    if (direction == 'next' and model_step > 0) or (direction == 'prev' and model_step < 0):
                         skip = abs(model_step)
                         print(f"{log_prefix} [Model] PredRank: {model_pred_rank}, Step: {model_step} (alpha={alpha:.2f}, {reason})")
                         use_model = True
                    else:
                         print(f"{log_prefix} [Model] 方向矛盾 (Dir: {direction}, ModelStep: {model_step}), 忽略模型")

            # --- 方案 B: 传统线性估算 (Fallback) ---
            if not use_model:
                est_pages = est_pages_by_rank
                if avg_diff > 0:
                     est_pages_by_score = int(dist_score / avg_diff)
                     if current_bottom_rank and abs(est_pages_by_rank - est_pages_by_score) > 5:
                          est_pages = est_pages_by_rank
                     else:
                          est_pages = est_pages_by_score
                
                # 距离保护
                if dist_score < 100000:
                    est_pages = min(est_pages, 3)
                
                if is_oscillation:
                    est_pages = 1
                
                # 应用保守系数
                if est_pages > 2:
                    skip = int(est_pages * 0.8)
                    skip = max(skip, 1)
            
            # 统一执行跳页
            last_jump_direction = direction
            
            # 限制单次最大跳页
            # 用户要求：最多跳5页，然后截图识别（更新模型）
            skip = min(skip, 5)
            
            if skip > 0:
                # 如果模型计算出 skip=0 (例如距离很近但为了防震荡)，则 skip 为 0，下面会走到普通翻页
                if skip > 2 or (use_model and skip >= 1): 
                    # 模型说跳1页也是跳，但普通翻页通常不用 skip_pages 函数而是直接 next_page
                    # 这里为了统一，如果 skip 大于 2 或者是模型确信的跳跃，就执行 skip
                    print(f"{log_prefix} 执行跳页: {skip} ({direction})")
                    skip_pages(client, direction, skip)
                    just_skipped = True
                    continue
            
            # 普通翻页
            if direction == 'next':
                print(f"{log_prefix} 普通翻页 (Next) ->")
                abd_command.next_page(client)
            else:
                print(f"{log_prefix} 普通翻页 (Prev) <-")
                abd_command.previous_page(client)
            
            time.sleep(2.0)
            
    print(f"[Target: {target_score:,}] 达到最大翻页次数，未找到精确匹配。")
    return {'target': target_score, 'error': 'Not found within limit', 'elapsed': time.time() - start_time}

def check_and_enter_ranking(client: AdbClient, reader: easyocr.Reader, out_dir: str) -> bool:
    """
    检查当前是否在活动主页，并点击'查看排行'进入排行榜。
    """
    print(">>> 检查当前页面状态...")
    ensure_dir(out_dir)
    screenshot_path = os.path.join(out_dir, "check_state.png")
    client.screencap_to_file(screenshot_path)
    
    # 识别页面文字
    results = reader.readtext(screenshot_path)
    
    # 目标关键词
    target_text = "查看排行"
    
    found = False
    click_x, click_y = 0, 0
    
    for bbox, text, conf in results:
        if target_text in text:
            print(f"识别到活动入口: '{text}' (置信度: {conf:.2f})")
            # 计算中心点坐标
            (tl, tr, br, bl) = bbox
            center_x = int((tl[0] + br[0]) / 2)
            center_y = int((tl[1] + br[1]) / 2)
            
            click_x, click_y = center_x, center_y
            found = True
            break
    
    if found:
        print(f"执行点击进入排行榜: ({click_x}, {click_y})")
        abd_command.tap(client, click_x, click_y)
        time.sleep(5.0) # 等待转场动画
        return True
    else:
        print(f"错误: 未识别到'{target_text}'按钮，请确保当前在活动主页。")
        return False

def run_crawler_for_tab(client: AdbClient, reader: easyocr.Reader, 
                        tab_func, out_dir: str, anchor_filename: str, config: Dict,
                        do_search: bool = True) -> Tuple[Dict, List]:
    """
    运行单个 Tab (积分榜或最高记录榜) 的爬取流程
    """
    ensure_dir(out_dir)
    
    # 切换到对应的 Tab
    print(f"\n>>> 切换 Tab 并初始化...")
    tab_func(client)
    time.sleep(3.0) # 等待页面切换
    
    # 初始化排名估算器 (每个榜单独立)
    estimator = RankEstimator()
    
    # 1. 获取锚点数据
    anchor_file = os.path.join(out_dir, anchor_filename)
    if os.path.exists(anchor_file):
        print(f"发现已有锚点数据 ({anchor_filename})，跳过采集。")
        with open(anchor_file, 'r', encoding='utf-8') as f:
            anchors = json.load(f)
            anchors = {int(k): v for k, v in anchors.items()}
    else:
        anchors = get_anchors(client, reader, os.path.join(out_dir, 'anchors'), config['anchor_ranks'])
        with open(anchor_file, 'w', encoding='utf-8') as f:
            json.dump(anchors, f, indent=2)
    
    # 将锚点数据加入估算器
    for rank, score in anchors.items():
        estimator.add_data(rank, score)
        
    # 2. 查找目标积分 (可选)
    results = []
    if do_search:
        for target in config['target_scores']:
            res = find_target_score(client, reader, target, anchors, estimator, os.path.join(out_dir, f'search_{target}'))
            results.append(res)
    else:
        print(">>> 此榜单仅采集锚点，跳过目标积分搜索。")
        
    # 3. 提取关键排名的积分
    key_ranks = config['export_key_ranks']
    targets_rank_score = {}
    for r in key_ranks:
        if r in anchors:
            targets_rank_score[str(r)] = anchors[r]
            
    return targets_rank_score, results

def run_task(device_id: str = None, out_dir: str = 'rank_data') -> str:
    """
    执行完整的爬取任务，并返回报告文本
    """
    total_start_time = time.time()
    ensure_dir(out_dir)
    
    client = AdbClient(device_id=device_id)
    client.connect()
    
    # 加载配置
    config = load_config()
    
    print("初始化 OCR...")
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
    
    # --- Phase 0: 检查状态并进入 ---
    if not check_and_enter_ranking(client, reader, out_dir):
        return "错误：无法进入排行榜页面，任务终止。"

    # --- Phase 1: 积分排行 ---
    print("\n========== 开始抓取: 积分排行 ==========")
    point_targets_rank_score, point_results = run_crawler_for_tab(
        client, reader, 
        abd_command.open_point_rank, 
        os.path.join(out_dir, 'point_rank'), 
        'anchors_point.json',
        config
    )
    
    # --- Phase 2: 最高记录排行 ---
    print("\n========== 开始抓取: 最高记录排行 ==========")
    best_targets_rank_score, best_results = run_crawler_for_tab(
        client, reader, 
        abd_command.open_best_record_rank, 
        os.path.join(out_dir, 'best_record_rank'), 
        'anchors_best.json',
        config,
        do_search=False
    )
            
    # 最终合并结果
    final_data = {
        'scoreRankTargets': point_results,
        'scoreRankKeyPoints': point_targets_rank_score, 
        'highestRankKeyPoints': best_targets_rank_score
    }
    
    with open(os.path.join(out_dir, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
        
    # 生成 TXT 报告
    report_text = generate_report_text(final_data, config)
    report_path = os.path.join(out_dir, 'result.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    total_elapsed = time.time() - total_start_time
    print(f"\n全部任务完成！总耗时: {total_elapsed:.1f}s")
    
    return report_text

def main():
    parser = argparse.ArgumentParser(description='偶像梦幻祭2 排名抓取')
    parser.add_argument('--device', help='ADB Device ID')
    parser.add_argument('--out', default='rank_data', help='输出目录')
    args = parser.parse_args()
    
    # 1. 执行任务
    report_text = run_task(args.device, args.out)

def generate_report_text(data: Dict, config: Dict = None) -> str:
    lines = []
    
    # 默认配置 (兼容旧逻辑)
    if config is None:
        config = {}
        
    export_key_ranks = config.get('export_key_ranks', [100, 1000, 5000, 10000])
    
    # 1. 积分线统计 (从 scoreRankKeyPoints 获取)
    lines.append("积分线统计")
    key_points = data.get('scoreRankKeyPoints', {})
    
    # 按照 export_key_ranks 排序
    for rank in export_key_ranks:
        rank_str = str(rank)
        score = key_points.get(rank_str, 0)
        lines.append(f"{rank_str}位-{score:,}pt")
    lines.append("")
    
    # 2. ★5线统计 (从 scoreRankTargets 获取)
    lines.append("★5线统计")
    targets = data.get('scoreRankTargets', [])
    
    # 获取文案映射
    score_map = config.get('score_labels', {
        3500000: "一卡",
        7500000: "二卡",
        11000000: "三卡",
        15000000: "四卡",
        22000000: "满破"
    })
    
    # 对结果按分数排序 (从小到大)
    sorted_targets = sorted(targets, key=lambda x: x['target'])
    
    for item in sorted_targets:
        t_score = item['target']
        rank = item.get('rank', '???')
        # 格式化 rank (如果是数字，加千分位)
        if isinstance(rank, int):
            rank_str = f"{rank:,}"
        else:
            rank_str = str(rank)
            
        label = score_map.get(t_score, f"{t_score:,}pt")
        # 格式: 一卡 3,500,000pt-4,966位
        lines.append(f"{label} {t_score:,}pt-{rank_str}位")
        
    lines.append("")
    
    # 3. 奖杯分线统计 (从 highestRankKeyPoints 获取)
    lines.append("奖杯分线统计")
    high_points = data.get('highestRankKeyPoints', {})
    
    trophy_map = config.get('rank_labels', {
        100: "(虹杯)",
        1000: "(金杯)",
        5000: "(银杯)",
        10000: "(银杯)"
    })
    
    for rank in export_key_ranks:
        rank_str = str(rank)
        score = high_points.get(rank_str, 0)
        
        # 尝试匹配 trophy prefix (优先尝试 int key，再尝试 str key)
        prefix = trophy_map.get(rank, "")
        if not prefix:
             prefix = trophy_map.get(str(rank), "")
             
        lines.append(f"{prefix}{rank_str}位-{score:,}")
        
    lines.append("*20001位以下为铜杯")
    
    return "\n".join(lines)

def get_access_token(app_id: str, client_secret: str) -> str:
    """
    获取 QQ 机器人 AccessToken
    文档: https://bot.q.qq.com/wiki/develop/api-v2/dev-prepare/interface-framework/api-use.html
    """
    url = "https://bots.qq.com/app/getAppAccessToken"
    payload = {
        "appId": app_id,
        "clientSecret": client_secret
    }
    
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            token = data.get("access_token")
            if token:
                print(f"[QQBot] 获取 AccessToken 成功，有效期 {data.get('expires_in')}秒")
                return token
            else:
                print(f"[QQBot] 获取 AccessToken 失败: {data}")
        else:
            print(f"[QQBot] 获取 AccessToken HTTP 错误: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[QQBot] 获取 AccessToken 异常: {e}")
        
    return ""

if __name__ == '__main__':
    main()
