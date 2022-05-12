import tweepy
import cfg_load
alpha = cfg_load.load('alpha.yaml')

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

    def tweet(self):
        tweet = "hello"
        #status = self.api.update_status(status=tweet)