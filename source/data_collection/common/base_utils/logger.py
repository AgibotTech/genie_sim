# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import logging
import sys
import time
from datetime import datetime, timezone

from colorama import Fore, Style, init

init()

# Record process start time
_PROCESS_START_TIME = time.time()


class ColoredFormatter(logging.Formatter):
    """Custom formatter with error handling and colored output"""

    def formatTime(self, record, datefmt=None):
        """Format time as ISO 8601 with UTC timezone"""
        ct = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return ct.strftime("%Y-%m-%dT%H:%M:%SZ")

    def format(self, record):
        """Format log record with process elapsed time"""
        try:
            # Calculate process elapsed time in milliseconds
            elapsed_ms = int((time.time() - _PROCESS_START_TIME) * 1000)

            # Format elapsed time with comma as thousand separator
            elapsed_str = f"{elapsed_ms:,}"

            # Get formatted time
            record.asctime = self.formatTime(record)

            # Build format string with elapsed time
            format_str = f"{record.asctime} [{elapsed_str}ms] %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"

            # Apply color based on log level
            COLOR_FORMATS = {
                logging.DEBUG: Fore.CYAN + format_str + Style.RESET_ALL,
                logging.INFO: Fore.GREEN + format_str + Style.RESET_ALL,
                logging.WARNING: Fore.YELLOW + format_str + Style.RESET_ALL,
                logging.ERROR: Fore.RED + format_str + Style.RESET_ALL,
                logging.CRITICAL: Fore.RED + Style.BRIGHT + format_str + Style.RESET_ALL,
            }

            formatter = logging.Formatter(COLOR_FORMATS[record.levelno])
            return formatter.format(record)
        except Exception as e:
            # Fallback format if primary formatting fails
            try:
                return (
                    f"{Fore.RED}Log format error: {e} | "
                    f"Original message: {repr(record.msg)} | "
                    f"Args: {repr(record.args)}{Style.RESET_ALL}"
                )
            except Exception:
                return f"{Fore.RED}Critical log formatting failure{Style.RESET_ALL}"


class Logger:
    """Unified Logger class, singleton pattern, provides colored output and error handling"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger()
        return cls._instance

    def _init_logger(self):
        self.logger = logging.getLogger("DataCollectionLogger")
        self.logger.setLevel(logging.INFO)  # logging.DEBUG to enable debug log

        # Clear existing handlers to avoid duplicate addition
        self.logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)

        # Prevent log propagation to root logger
        self.logger.propagate = False

        # Wrap all logging methods with error handling
        self._wrap_logging_methods()

    def _wrap_logging_methods(self):
        """Wrap all logger methods with error handling"""
        methods = ["debug", "info", "warning", "error", "critical", "exception"]

        for method_name in methods:
            original = getattr(self.logger, method_name)
            setattr(self, method_name, self._create_safe_method(original))

    def _create_safe_method(self, method):
        """Create a wrapped method that handles formatting exceptions"""

        def safe_method(msg, *args, **kwargs):
            try:
                # Use stacklevel to skip this wrapper and show the actual caller's file and line
                # stacklevel=3: 1 for safe_method, 1 for wrapper call, 1 for actual caller
                kwargs["stacklevel"] = kwargs.get("stacklevel", 3)
                method(msg, *args, **kwargs)
            except Exception as e:
                try:
                    # For error logging, also use stacklevel
                    self.logger.error(
                        f"Logging error: {e} [Message: {repr(msg)}, Args: {repr(args)}]",
                        stacklevel=3,
                    )
                except Exception:
                    self.logger.error("Critical failure in error handling", stacklevel=3)

        return safe_method

    def set_level(self, level):
        """Set log level"""
        self.logger.setLevel(level)


# Create singleton instance and export as 'logger'
logger = Logger()

# Also export Logger class for backward compatibility if needed
__all__ = ["logger", "Logger"]

if __name__ == "__main__":
    # Test error handling
    logger.info("This should work %s", "OK")
    logger.debug("This will fail %s")  # Missing argument
    logger.info("Bad format %d", "string")  # Type mismatch
    logger.info("Works: %s", {"complex": "object"})
    logger.info("This will fail in formatter", extra={"unexpected": "data"})
