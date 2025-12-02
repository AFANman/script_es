import unittest
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from python.capture.rank_estimator import RankEstimator

class TestRankEstimator(unittest.TestCase):
    def test_fitting_and_prediction(self):
        estimator = RankEstimator()
        
        # 模拟一些符合幂律分布的数据 Score = 100000 * Rank^-0.5
        # Rank 1 -> 100000
        # Rank 100 -> 10000
        # Rank 10000 -> 1000
        
        estimator.add_data(1, 100000)
        estimator.add_data(100, 10000)
        estimator.add_data(10000, 1000)
        
        self.assertTrue(estimator.is_fitted)
        self.assertAlmostEqual(estimator.b, -0.5, delta=0.01)
        
        # 测试预测
        # Target Score 5000 -> Rank should be around 400
        # 5000 = 100000 * R^-0.5 => R^-0.5 = 0.05 => R = 0.05^-2 = 400
        pred = estimator.predict_rank(5000)
        self.assertIsNotNone(pred)
        self.assertTrue(380 < pred < 420, f"Prediction {pred} is not close to 400")

    def test_adaptive_step(self):
        estimator = RankEstimator()
        estimator.add_data(1, 100000)
        estimator.add_data(100, 10000)
        estimator.add_data(10000, 1000)
        
        # Current Rank 300, Target Rank 400 (Target Score 5000)
        # Per Page 20 -> Diff 5 pages
        
        step, alpha, reason = estimator.get_adaptive_step(300, 400, per_page=20)
        # 样本量少(3个)，alpha会被打折 (0.7)
        # 拟合度应该很高 (R2=1.0)
        # 距离 5 页，不算很近 (<5才算近)，所以 alpha 0.7
        # Step = 5 * 0.7 = 3
        
        print(f"Step: {step}, Alpha: {alpha}, Reason: {reason}")
        self.assertGreater(step, 0)
        self.assertTrue("low_samples" in reason)
        
    def test_oscillation_penalty(self):
        estimator = RankEstimator()
        for i in range(1, 10):
            estimator.add_data(i*100, int(100000 * (i*100)**-0.5))
            
        # 模拟震荡
        step, alpha, reason = estimator.get_adaptive_step(300, 400, per_page=20, last_direction_flip=True)
        
        self.assertTrue("oscillation_detected" in reason)
        self.assertLessEqual(alpha, 0.5)

if __name__ == '__main__':
    unittest.main()