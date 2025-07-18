import logging
import sys
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(module)s:%(filename)s:%(funcName)s:%(lineno)d:%(levelname)s:  %(message)s",
    stream=sys.stdout,  # Ensure output goes to stdout
    force=True,  # Override any existing configuration
)


# Create a function to get loggers with consistent naming
def get_logger(name: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the configured format.

    Args:
        name: Logger name. If None, uses the calling module's __name__
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
               If None, uses the global default.

    Returns:
        Configured logger instance
    """
    if name is None:
        # Get the calling module's name
        import inspect

        frame = inspect.currentframe().f_back  # type: ignore
        assert frame is not None, "Frame should not be None"
        name = frame.f_globals.get("__name__", "unknown")

    logger = logging.getLogger(name)

    if level is not None:
        # Convert string level to logging constant
        level_mapping = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        log_level = level_mapping.get(level.upper(), logging.INFO)
        logger.setLevel(log_level)

    return logger


# Create a default logger for convenience
logger = logging.getLogger(__name__)


# Optional: Add some convenience functions
def info(msg: str):
    """Log an info message"""
    logger.info(msg)


def warning(msg: str):
    """Log a warning message"""
    logger.warning(msg)


def error(msg: str):
    """Log an error message"""
    logger.error(msg)


def debug(msg: str):
    """Log a debug message"""
    logger.debug(msg)


# Example usage when run directly
if __name__ == "__main__":
    logger.info("Logger configuration loaded")
    logger.warning("This is a test warning")
    logger.error("This is a test error")
