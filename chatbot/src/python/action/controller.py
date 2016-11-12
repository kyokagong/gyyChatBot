
from chatbot.src.python.utils.data_util import LOG_FILE, LOG_LINE_NUM

def get_dgk_log():
    # log_file = "logs/dgk.log"
    with open(LOG_FILE, 'r') as log_file:
        return log_file.readlines()[:LOG_LINE_NUM]