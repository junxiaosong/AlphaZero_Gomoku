from datetime import datetime
import os

def log(message, logpath):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S ")
    message = timestamp + message
    print(message)
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    with open(logpath,'a+') as fs:
        fs.write(message+'\n')
