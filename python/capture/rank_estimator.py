import math
from typing import List, Tuple, Optional

class RankEstimator:
    """
    基于幂律分布 (Power Law) 的排名预估器，结合自适应步长搜索。
    模型: Score = A * Rank^B  =>  ln(Score) = ln(A) + B * ln(Rank)
    """
    def __init__(self):
        self.ranks: List[int] = []
        self.scores: List[int] = []
        self.a: float = 0.0  # ln(A)
        self.b: float = 0.0  # Slope B
        self.r_squared: float = 0.0
        self.is_fitted: bool = False
        self.min_data_points = 3  # 至少需要几个点才开始拟合

    def add_data(self, rank: int, score: int):
        """添加观测点并重新拟合"""
        if rank <= 0 or score <= 0:
            return
        
        # 避免重复添加相同 rank 的数据
        if rank in self.ranks:
            # 如果 rank 相同但 score 不同，更新 score
            idx = self.ranks.index(rank)
            self.scores[idx] = score
        else:
            self.ranks.append(rank)
            self.scores.append(score)
            
        self._fit()

    def _fit(self):
        """使用最小二乘法拟合 ln(Score) = a + b * ln(Rank)"""
        if len(self.ranks) < self.min_data_points:
            self.is_fitted = False
            return

        try:
            # 转换为对数坐标
            log_ranks = [math.log(r) for r in self.ranks]
            log_scores = [math.log(s) for s in self.scores]
            
            n = len(self.ranks)
            sum_x = sum(log_ranks)
            sum_y = sum(log_scores)
            sum_xy = sum(x * y for x, y in zip(log_ranks, log_scores))
            sum_xx = sum(x * x for x in log_ranks)

            # 最小二乘法求解
            denominator = (n * sum_xx - sum_x * sum_x)
            if abs(denominator) < 1e-9:
                self.is_fitted = False
                return
            
            self.b = (n * sum_xy - sum_x * sum_y) / denominator
            self.a = (sum_y - self.b * sum_x) / n

            # 计算 R^2 (拟合优度)
            mean_y = sum_y / n
            ss_tot = sum((y - mean_y) ** 2 for y in log_scores)
            
            if abs(ss_tot) < 1e-9:
                self.r_squared = 0.0 # 样本方差为0，无法计算R^2
            else:
                ss_res = 0.0
                for x, y in zip(log_ranks, log_scores):
                    y_pred = self.a + self.b * x
                    ss_res += (y - y_pred) ** 2
                self.r_squared = 1 - (ss_res / ss_tot)
            
            # 简单的合理性检查：积分通常随排名增加而减少，所以 b 应该小于 0
            if self.b >= 0: 
                 # 如果斜率大于0，说明数据反常（排名越后积分越高？），不可信
                 self.is_fitted = False
            else:
                self.is_fitted = True

        except Exception as e:
            print(f"[Estimator] Fitting error: {e}")
            self.is_fitted = False

    def predict_rank(self, target_score: int) -> Optional[int]:
        """
        根据目标积分预测排名。
        采用局部拟合策略：只使用与 target_score 最接近的 K 个点进行拟合。
        """
        if len(self.ranks) < self.min_data_points:
            return None

        # 1. 找出最近的 K 个点 (局部窗口)
        # 我们希望找到积分在 target_score 附近的点
        # 比如取最近的 6 个点
        K = 6
        
        # 计算每个点与目标的积分差距比例
        diffs = []
        for i in range(len(self.ranks)):
            score_diff = abs(self.scores[i] - target_score)
            diffs.append((score_diff, i))
        
        # 按差距排序，取前 K 个
        diffs.sort(key=lambda x: x[0])
        top_k_indices = [idx for _, idx in diffs[:K]]
        
        # 提取局部数据
        local_ranks = [self.ranks[i] for i in top_k_indices]
        local_scores = [self.scores[i] for i in top_k_indices]
        
        # 2. 对局部数据进行拟合
        # 局部范围内，曲线可能近似线性，也可能符合幂律。
        # 考虑到幂律在局部也适用（对数空间线性），我们继续用幂律，但只针对这几个点。
        
        if len(local_ranks) < 2:
            return None
            
        try:
            log_ranks = [math.log(r) for r in local_ranks]
            log_scores = [math.log(s) for s in local_scores]
            
            n = len(local_ranks)
            sum_x = sum(log_ranks)
            sum_y = sum(log_scores)
            sum_xy = sum(x * y for x, y in zip(log_ranks, log_scores))
            sum_xx = sum(x * x for x in log_ranks)

            denominator = (n * sum_xx - sum_x * sum_x)
            if abs(denominator) < 1e-9:
                return None
            
            b = (n * sum_xy - sum_x * sum_y) / denominator
            a = (sum_y - b * sum_x) / n
            
            # 临时保存局部拟合的 R2，供日志使用
            # (注意：self.r_squared 存的是全局的，这里算一个局部的)
            # 为了简单，我们直接覆盖 self.r_squared 供调用者查看当前预测的可信度
            mean_y = sum_y / n
            ss_tot = sum((y - mean_y) ** 2 for y in log_scores)
            if abs(ss_tot) > 1e-9:
                ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(log_ranks, log_scores))
                self.r_squared = 1 - (ss_res / ss_tot)
            else:
                self.r_squared = 0.0

            # 3. 预测
            log_target_score = math.log(target_score)
            log_target_rank = (log_target_score - a) / b
            predicted_rank = math.exp(log_target_rank)
            
            return int(predicted_rank)

        except Exception as e:
            print(f"[Estimator] Local prediction error: {e}")
            return None

    def get_adaptive_step(self, current_rank: int, target_rank: int, per_page: int = 20, 
                          last_direction_flip: bool = False) -> Tuple[int, float, str]:
        """
        计算自适应跳页步长
        Returns: (final_jump_pages, confidence_alpha, reason)
        """
        if target_rank is None:
            return 0, 0.0, "No prediction"

        raw_diff_pages = (target_rank - current_rank) / per_page
        
        # 基础置信度
        alpha = 1.0
        reasons = []

        # 1. 样本量检查
        if len(self.ranks) < 4:
            alpha *= 0.7
            reasons.append("low_samples")

        # 2. 拟合优度检查
        if self.r_squared < 0.90:
            alpha *= 0.6
            reasons.append(f"low_r2({self.r_squared:.2f})")
        elif self.r_squared < 0.98:
            alpha *= 0.8
        else:
            reasons.append("high_r2")

        # 3. 震荡检测 (最重要)
        if last_direction_flip:
            alpha *= 0.5
            reasons.append("oscillation_detected")

        # 4. 距离衰减保护 (防止在最后几页反复横跳)
        # 如果距离目标很近 (<5页)，信任模型直接跳，或者如果模型不可信就逐页
        if abs(raw_diff_pages) < 5:
            if self.r_squared > 0.95:
                alpha = 1.0 # 距离近且模型准，直接对其
                reasons.append("close_range_precise")
            else:
                # 距离近但模型不准，alpha保持不变(已打折)，防止过冲
                pass
        
        final_jump = int(raw_diff_pages * alpha)
        
        # 最小步长保障：如果算出来是0但有距离，至少动一下，除非震荡且距离极短
        if final_jump == 0 and abs(raw_diff_pages) > 0.8:
             # 如果发生了震荡且距离很近，可能陷入死循环，此时 final_jump=0 是对的，交由外部逐页处理
             # 但如果没有震荡，应该至少前进1页
             if not last_direction_flip:
                 final_jump = 1 if raw_diff_pages > 0 else -1
                 reasons.append("min_step")

        return final_jump, alpha, ",".join(reasons)
