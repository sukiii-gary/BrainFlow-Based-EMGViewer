"""分段(Phase 2 实现):baseline_stats / detect_active / detect_reps / segment_task。

复刻 MATLAB:参考通道包络 + 双阈值滞回 + 逐 rep 峰值切分,4 类
(baseline / mvic / cyclic / fatigue_bout),cyclic 输出 phase='rep'(无三相)。
"""
