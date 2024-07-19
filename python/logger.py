import logging
import colorlog
import threading
import multiprocessing

# Create a custom logger
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a logging format with a timestamp, process ID, and thread ID
formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Process %(process)d] - %(message)s')
file_handler.setFormatter(formatter)

# Create a color formatter for console output with process ID and thread ID
color_formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s - [Process %(process)d] - %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
)
console_handler.setFormatter(color_formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Create a lock for thread safety
log_lock = threading.Lock()

# Define a thread-safe logging function
def log(level, message):
    with log_lock:
        if level == 'debug':
            logger.debug(message)
        elif level == 'info':
            logger.info(message)
        elif level == 'warning':
            logger.warning(message)
        elif level == 'error':
            logger.error(message)
        elif level == 'critical':
            logger.critical(message)
        else:
            logger.info(message)

if __name__ == "__main__":
    import time

    # Creating threads
    threads = []
    for i in range(3):
        t = threading.Thread(target=log_messages, args=(f"thread {i}",))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    # Creating processes
    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=log_messages, args=(f"process {i}",))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
