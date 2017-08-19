import os 
from slacker import Slacker

slack = Slacker(os.environ['SLACKER_TOKEN'])
response = slack.users.list()
slack.chat.post_message('#bookmonitor', 'Hello fellow slackers!')
