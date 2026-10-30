# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import logging
import sys
from colorama import Fore, Style, init

init()


class ColoredFormatter(logging.Formatter):
    """Format teleop messages with level-specific colors"""

    format_str = "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

    FORMATS = {
        logging.DEBUG: Fore.CYAN + format_str + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + format_str + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + format_str + Style.RESET_ALL,
        logging.ERROR: Fore.RED + format_str + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Style.BRIGHT + format_str + Style.RESET_ALL,
    }

    def format(self, record):
        formatter = logging.Formatter(self.FORMATS.get(record.levelno, self.format_str))
        return formatter.format(record)


class Logger:
    """Return the shared teleop logger with one colored console handler."""

    def __new__(cls):
        instance = logging.getLogger("geniesim_teleop")
        if not instance.handlers:
            instance.setLevel(logging.INFO)
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(ColoredFormatter())
            instance.addHandler(handler)
        return instance


logger = Logger()
