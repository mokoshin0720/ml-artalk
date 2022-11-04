from slack import WebClient
import os

if __name__ == '__main__':
    token = os.getenv('SLACK_API_TOKEN')
    channel = os.getenv('CHANNEL')
    mention_to_me = os.getenv("SLACK_USER_ID")

    client = WebClient(token=token)
    response = client.chat_postMessage(channel='#shinya-ml-notify', text=mention_to_me+"DONE!")