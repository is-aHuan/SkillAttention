import builtins
import torch.distributed as dist
import logging

original_print = builtins.print

def rank_zero_print(*args, **kwargs):
    is_ddp = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_ddp else 0
    if rank == 0:
        original_print(*args, **kwargs)

# 创建一个日志过滤器，只允许rank 0进程的日志通过
class RankZeroFilter(logging.Filter):
    def filter(self, record):
        try:
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank() == 0
            return True  # 如果不在DDP环境中，则允许所有日志通过
        except:
            # 如果获取rank失败（例如在初始化之前），则允许日志通过
            return True

# 替换全局print函数
builtins.print = rank_zero_print
# 添加过滤器到根日志记录器
rank_zero_filter = RankZeroFilter()
logging.getLogger().addFilter(rank_zero_filter)

# 特别处理metaworld日志
metaworld_logger = logging.getLogger('metaworld')
if metaworld_logger:
    metaworld_logger.addFilter(rank_zero_filter)