# coding:utf8
import time
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    """
    封装了基本的 TensorBoard 操作。
    """

    def __init__(self, log_dir):
        # 初始化 TensorBoard SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir)
        self.index = {}  # 用于追踪图表的点
        self.log_text = ''  # 用于记录日志信息

    def reinit(self, log_dir, **kwargs):
        """
        重新初始化 TensorBoard writer，并设置新的日志目录。
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        return self

    def plot(self, name, y, step=None):
        """
        将标量值记录到 TensorBoard。
        例如：plot('loss', 1.00)
        """
        if step is None:
            step = self.index.get(name, 0)
        self.writer.add_scalar(name, y, step)
        self.index[name] = step + 1

    def img(self, name, img_, step=None):
        """
        将图像记录到 TensorBoard。
        img_ 应该是一个张量（例如，torch.Tensor）。
        """
        if step is None:
            step = self.index.get(name, 0)
        self.writer.add_images(name, img_, step)

    def log(self, info, step=None, win='log_text'):
        """
        记录信息为文本（可选）。
        """
        if step is None:
            step = self.index.get(win, 0)
        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.writer.add_text(win, self.log_text, step)
        self.index[win] = step + 1

    def __getattr__(self, name):
        """
        允许访问其他 TensorBoard writer 的函数。
        """
        return getattr(self.writer, name)
