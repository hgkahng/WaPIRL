# -*- coding: utf-8 -*-

import os
import sys
import logging
from colorama import Fore


def make_epoch_description(history: dict, current: int, total: int, best: int, exclude: list = []):
    """Create description string for logging progress."""
    pfmt = f">{len(str(total))}d"
    desc = f" Epoch: [{current:{pfmt}}/{total:{pfmt}}] ({best:{pfmt}}) |"
    for metric_name, metric_dict  in history.items():
        if not isinstance(metric_dict, dict):
            raise TypeError("`history` must be a nested dictionary.")
        if metric_name in exclude:
            continue
        for k, v in metric_dict.items():
            desc += f" {k}_{metric_name}: {v:.4f} |"
    return desc


def get_tqdm_config(total: int, leave: bool = True, color: str = 'white'):
    fore_colors = {
        'red': Fore.LIGHTRED_EX,
        'green': Fore.LIGHTGREEN_EX,
        'yellow': Fore.LIGHTYELLOW_EX,
        'blue': Fore.LIGHTBLUE_EX,
        'magenta': Fore.LIGHTMAGENTA_EX,
        'cyan': Fore.LIGHTCYAN_EX,
        'white': Fore.LIGHTWHITE_EX,
    }
    return {
        'file': sys.stdout,
        'total': total,
        'desc': " ",
        'dynamic_ncols': True,
        'bar_format': \
            "{l_bar}%s{bar}%s| [{elapsed}<{remaining}, {rate_fmt}{postfix}]" % (fore_colors[color], Fore.RESET),
        'leave': leave
    }


def get_logger(stream: bool = False, logfile: str = None, level=logging.INFO):
    """
    Arguments:
        stream: bool, default False.
        logfile: str, path.
    """
    _format = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    logFormatter = logging.Formatter(_format)

    rootLogger = logging.getLogger()

    if logfile:
        touch(logfile)
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

    if stream:
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(logFormatter)
        rootLogger.addHandler(streamHandler)

    rootLogger.setLevel(level)

    return rootLogger


def touch(filepath: str, mode: str = 'w'):
    assert mode in ['a', 'w']
    directory, _ = os.path.split(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    open(filepath, mode).close()
