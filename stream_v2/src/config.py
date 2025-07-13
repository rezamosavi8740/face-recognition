from dynaconf import Dynaconf
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re



CONFIG = Dynaconf(
    settings_files=["settings.yaml"],
    envvar_prefix=False,  # allows raw env keys
    load_dotenv=True      # <== this loads .env!
)

def format_with_stars(sentence, width=85, sign="*"):
    sentence = f" {sentence} "  # Add some spacing around the sentence
    padding = (width - len(sentence)) // 2  # Calculate padding for left and right
    return f"{sign * padding}{sentence}{sign * padding}".ljust(width, sign)  # Ensure total width



def is_safe_filename(name):
    # Must not be empty or just whitespace
    if not name or not name.strip():
        return False
    # Allow only safe characters: letters, digits, -, _, .
    return bool(re.match(r'^[\w\-\.]+$', name.strip()))

def setup_logger(log_file, name="BINA", max_bytes=10*1024*1024, backup_count=1):
    name = str(name)
    if not is_safe_filename(name):
        name = "test_stream"
    log_file = log_file.format(stream_id=name)
    log_path = Path(log_file)

    if log_path.exists():
        try:
            log_path.unlink()
        except Exception as e:
            print(f"Failed to delete existing log file {log_file}: {e}")
    
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.handlers = []  # Avoid duplicate handlers

    log_level_str = CONFIG.public.log_level
    log_level = getattr(logging, log_level_str.upper(), logging.DEBUG)
    logger.setLevel(log_level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Conditional terminal output
    # if name == "BINA":
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

