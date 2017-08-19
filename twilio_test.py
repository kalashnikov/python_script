import os
from twilio.rest import Client

account_sid = os.environ['TWILIO_SID'] 
account_token = os.environ['TWILIO_TOKEN']

client = Client(account_sid, account_token)

msg = client.messages.create(
        to="+886933311337",
        from_="+14159413785",
        body="Lin YuYin: I Love You !!! from Kala Kuo"
        )

print(msg.sid)
