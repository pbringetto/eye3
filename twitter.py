import tweepy
from io import BytesIO
from PIL import Image
import cfg_load
import os
dir = os.path.dirname(os.path.realpath(__file__))
path = os.path.join(dir, 'alpha.yaml')
alpha = cfg_load.load(path)

class Twitter:
    def __init__(self):
        self.twitter_auth_keys = alpha["twitter_auth_keys"]
        self.auth = tweepy.OAuthHandler(
                self.twitter_auth_keys['consumer_key'],
                self.twitter_auth_keys['consumer_secret']
            )
        self.auth.set_access_token(
                self.twitter_auth_keys['access_token'],
                self.twitter_auth_keys['access_token_secret']
            )
        self.api = tweepy.API(self.auth)

    def tweet(self, data, files):
       
        media_ids = []
        for file in files:
            img = Image.open(file)
            b = BytesIO()
            img.save(b, "PNG")
            b.seek(0)
            ret = self.api.media_upload(filename="dummy_string", file=b)
            media_ids.append(ret.media_id_string)

        #status = self.api.update_status(media_ids=[ret.media_id_string], status=data)
        status = self.api.update_status(media_ids=media_ids, status=data)
        print(data)