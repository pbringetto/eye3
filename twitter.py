import tweepy
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

    def tweet(self, data):
        status = self.api.update_status(status=data)
        print(data)