# ensemble_buffer.py
import numpy as np

class EnsembleBuffer:
    """
    保存最近H次模型输出的未来H步预测。
    buffer[i] = 这个时间点生成的[H, D]整数动作数组
    """

    def __init__(self, H, D):
        self.H = H
        self.D = D
        self.buffer = []  # list of [H,D] np.array(int)

    def push(self, pred_matrix_int):
        """
        pred_matrix_int: np.array shape [H,D] in 0..1000
        """
        self.buffer.append(pred_matrix_int.copy())
        if len(self.buffer) > self.H:
            self.buffer.pop(0)

    def current_action_int(self):
        """
        根据论文描述：
        - 用当前时刻t的第1个动作
        - 上一时刻t-1的第2个动作
        - 上上一时刻t-2的第3个动作
        ...
        拼出多个候选，然后逐维平均。
        返回 size [D] 的 int 向量（0..1000）
        """
        candidates = []
        # 最近的 buffer[-1] 是“当前 t 的 H×D”
        # buffer[-2] 是“t-1 的 H×D”
        # ...
        for k_back, pred in enumerate(reversed(self.buffer)):
            idx = k_back  # 0-based
            if idx >= self.H:
                break
            # 取这一份预测里第 idx 行（也就是它认为 idx+1 步后的动作）
            cand = pred[idx]  # shape [D]
            candidates.append(cand.astype(np.float32))

        if not candidates:
            return None

        avg = np.mean(candidates, axis=0)  # [D]
        avg_rounded = np.clip(np.round(avg), 0, 1000).astype(np.int32)
        return avg_rounded
