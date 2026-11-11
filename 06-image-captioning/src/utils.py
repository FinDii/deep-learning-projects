from torch import Tensor, optim


class AverageMeter:
    """
    跟踪指标的最近值、平均值、总和和计数
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer: optim.Optimizer, shrink_factor: float):
    """
    按指定因子缩小学习率

    :param optimizer: 学习率需要缩小的优化器
    :param shrink_factor: 乘以学习率的因子，在区间(0, 1)内
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] *= shrink_factor


def accuracy(scores: Tensor, targets: Tensor, k: int) -> float:
    """
    计算top-k准确率

    :param scores: 模型的输出分数
    :param targets: 真实标签
    :param k: top-k准确率中的k
    :return: top-k准确率
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D张量
    return correct_total.item() * (100.0 / batch_size)
