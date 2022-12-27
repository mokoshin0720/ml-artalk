from slack import WebClient
import os
import logging
import datetime
import pathlib

def notify_message(message: str):
    token = os.getenv('SLACK_API_TOKEN')
    channel = os.getenv('CHANNEL')
    mention_to_me = os.getenv("SLACK_USER_ID")

    client = WebClient(token=token)
    client.chat_postMessage(channel=channel, text=mention_to_me+message)

def notify_success(filename: str):
    token = os.getenv('SLACK_API_TOKEN')
    channel = os.getenv('CHANNEL')
    mention_to_me = os.getenv("SLACK_USER_ID")

    f = open(filename, 'r', encoding='UTF-8')
    log = f.read()

    client = WebClient(token=token)
    
    client.chat_postMessage(channel=channel, text=mention_to_me+ '\n' +log)

    f.close()

def notify_fail(message: str):
    token = os.getenv('SLACK_API_TOKEN')
    channel = os.getenv('CHANNEL')
    mention_to_me = os.getenv("SLACK_USER_ID")

    client = WebClient(token=token)
    client.chat_postMessage(channel=channel, text=mention_to_me+message)

def init_logger():
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    
    filename = '{year}-{month}-{day} {hour}:{minute}:{second}'.format(
        year=now.year,
        month=now.month,
        day=now.day,
        hour=now.hour,
        minute=now.minute,
        second=now.second,
    )

    filename = 'logs/' + filename + '.log'

    f = pathlib.Path(filename)
    f.touch()

    logging.basicConfig(
        filename=filename,
        filemode='w',
        level=logging.DEBUG,
    )

    return filename

if __name__ == '__main__':
    filename = init_logger()
    logging.info("test1-2")
    logging.info("test2-3")
    logging.info("test3-4")

    notify_success(filename)