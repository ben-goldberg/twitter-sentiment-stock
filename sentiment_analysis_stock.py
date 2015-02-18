#
# Ben Goldberg
# 
# Twitter Sentiment Analysis
#

import csv
from random import shuffle
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.base import TransformerMixin
from sklearn.pipeline import FeatureUnion
import numpy
from twython import Twython
from time import sleep
from yahoo_finance import Share
from datetime import date, timedelta
import random

# Global Vars
data_path = 'sanders-twitter-0.2/full-corpus-cleaned.csv'
percent_learning = .8
company_dict = {'microsoft':'MSFT', 'twitter':'TWTR', 'apple':'AAPL'}
sentiment_threshold = .05
start_money = 1000000
shares_to_change = 100
trades_remaining_today = 2
action_threshold = .01
happy_emoticons = [":-)",":)",":D",":o)",":]",":3",":c)",":>","=]","8)",":-))"]
happy_emoticons += [":-D","8-D","8D","x-D","xD","X-D","XD","=-D","=D","=-3","=3"]
negative_emoticons = [">:[",":-(",":(",":-c",":c",":-<",":<",":-[",":[",":{"]
negative_emoticons += [">:\ ",">:/",":-/",":-.",":/",":\ ","=/","=\ ",":L","=L",":S",">.<"]

# Global variables that contains the user credentials to access Twitter API 
access_token = "token"
access_token_secret = "token secret"
consumer_key = "key"
consumer_secret = "secret"

positive_words = ["absolutely","adorable","accepted","acclaimed","accomplish","accomplishment","achievement","action","active","admire","adventure","affirmative","affluent","agree","agreeable","amazing","angelic","appealing","approve","aptitude","attractive","awesome","beaming","beautiful","believe","beneficial","bliss","bountiful","bounty","brave","bravo","brilliant","bubbly","calm","celebrated","certain","champ","champion","charming","cheery","choice","classic","classical","clean","commend","composed","congratulation","constant","cool","courageous","creative","cute","dazzling","delight","delightful","distinguished","divine","earnest","easy","ecstatic","effective","effervescent","efficient","effortless","electrifying","elegant","enchanting","encouraging","endorsed","energetic","energized","engaging","enthusiastic","essential","esteemed","ethical","excellent","exciting","exquisite","fabulous","fair","familiar","famous","fantastic","favorable","fetching","fine","fitting","flourishing","fortunate","free","fresh","friendly","fun","funny","generous","genius","genuine","giving","glamorous","glowing","good","gorgeous","graceful","great","green","grin","growing","handsome","happy","harmonious","healing","healthy","hearty","heavenly","honest","honorable","honored","hug","idea","ideal","imaginative","imagine","impressive","independent","innovate","innovative","instant","instantaneous","instinctive","intuitive","intellectual","intelligent","inventive","jovial","joy","jubilant","keen","kind","knowing","knowledgeable","laugh","legendary","light","learned","lively","lovely","lucid","lucky","luminous","marvelous","masterful","meaningful","merit","meritorious","miraculous","motivating","moving","natural","nice","novel","now","nurturing","nutritious","okay","one","one-hundred percent","open","optimistic","paradise","perfect","phenomenal","pleasurable","plentiful","pleasant","poised","polished","popular","positive","powerful","prepared","pretty","principled","productive","progress","prominent","protected","proud","quality","quick","quiet","ready","reassuring","refined","refreshing","rejoice","reliable","remarkable","resounding","respected","restored","reward","rewarding","right","robust","safe","satisfactory","secure","seemly","simple","skilled","skillful","smile","soulful","sparkling","special","spirited","spiritual","stirring","stupendous","stunning","success","successful","sunny","super","superb","supporting","surprising","terrific","thorough","thrilling","thriving","tops","tranquil","transforming","transformative","trusting","truthful","unreal","unwavering","up","upbeat","upright","upstanding","valued","vibrant","victorious","victory","vigorous","virtuous","vital","vivacious","wealthy","welcome","well","whole","wholesome","willing","wonderful","wondrous","worthy","wow","yes","yummy","zeal","zealous"]
negative_words = ["abysmal","adverse","alarming","angry","annoy","anxious","apathy","appalling","atrocious","awful","bad","banal","barbed","belligerent","bemoan","beneath","boring","broken","callous","can't","clumsy","coarse","cold","cold-hearted","collapse","confused","contradictory","contrary","corrosive","corrupt","crazy","creepy","criminal","cruel","cry","cutting","dead","decaying","damage","damaging","dastardly","deplorable","depressed","deprived","deformed","deny","despicable","detrimental","dirty","disease","disgusting","disheveled","dishonest","dishonorable","dismal","distress","don't","dreadful","dreary","enraged","eroding","evil","fail","faulty","fear","feeble","fight","filthy","foul","frighten","frightful","gawky","ghastly","grave","greed","grim","grimace","gross","grotesque","gruesome","guilty","haggard","hard","hard-hearted","harmful","hate","hideous","homely","horrendous","horrible","hostile","hurt","hurtful","icky","ignore","ignorant","ill","immature","imperfect","impossible","inane","inelegant","infernal","injure","injurious","insane","insidious","insipid","jealous","junky","lose","lousy","lumpy","malicious","mean","menacing","messy","misshapen","missing","misunderstood","moan","moldy","monstrous","naive","nasty","naughty","negate","negative","never","no","nobody","nondescript","nonsense","not","noxious","objectionable","odious","offensive","old","oppressive","pain","perturb","pessimistic","petty","plain","poisonous","poor","prejudice","questionable","quirky","quit","reject","renege","repellant","reptilian","repulsive","repugnant","revenge","revolting","rocky","rotten","rude","ruthless","sad","savage","scare","scary","scream","severe","shoddy","shocking","sick","sickening","sinister","slimy","smelly","sobbing","sorry","spiteful","sticky","stinky","stormy","stressful","stuck","stupid","substandard","suspect","suspicious","tense","terrible","terrifying","threatening","ugly","undermine","unfair","unfavorable","unhappy","unhealthy","unjust","unlucky","unpleasant","upset","unsatisfactory","unsightly","untoward","unwanted","unwelcome","unwholesome","unwieldy","unwise","upset","vice","vicious","vile","villainous","vindictive","wary","weary","wicked","woeful","worthless","wound","yell","yucky","zero"]

class StockQuery:
    """
    Tracks a list of actively trading stocks
    """

    def __init__(self, company_symbol_list):
        self.symbol_to_quote_dict = {}
        for symbol in company_symbol_list:
            self.symbol_to_quote_dict[symbol] = Share(symbol)

    def get_price(self, company_symbol):
        return self.symbol_to_quote_dict[company_symbol].get_price()

    def get_historical(self, company_symbol, early_date, late_date):
        return self.symbol_to_quote_dict[company_symbol].get_historical(early_date, late_date)

    def refresh_all(self):
        for symbol in self.symbol_to_quote_dict:
            self.symbol_to_quote_dict[symbol].refresh()

class BaselineBot:
    """
    Naively buys and sells shares of actively trading stocks
    """
    def __init__(self, company_dict, money, trades_remaining_today):

        self.company_dict = company_dict
        self.money = money
        self.trades_remaining_today = trades_remaining_today
        self.owned = {}
        self.stock_query = StockQuery(company_dict.values())

        for symbol in self.company_dict.values():
            self.owned[symbol] = 0

    def buy(self, company_symbol, num_shares_to_buy):
        """
        Input: stock symbol for company, number of shares to buy
        Output: bool representing success or failure
        """

        print "----------"
        print "BUYING: ", company_symbol
        print "----------"

        # Only buy if enough trades left today
        if self.trades_remaining_today < 1:
            return False

        # Get current price
        self.stock_query.refresh_all()
        current_price = float(self.stock_query.get_price(company_symbol))

        # If enough money, buy stock
        money_needed = current_price * num_shares_to_buy
        if self.money >= money_needed:
            self.trades_remaining_today -= 1
            self.money -= money_needed
            self.owned[company_symbol] += num_shares_to_buy
            return True
        else:
            return False

    def sell(self, company_symbol, num_shares_to_sell):
        """
        Input: stock symbol for company, number of shares to sell
        Output: bool representing success or failure
        """
        print "----------"
        print "SELLING: ", company_symbol
        print "----------"
        if self.owned[company_symbol] >= num_shares_to_sell:
            # Get current price        
            self.stock_query.refresh_all()
            current_price = float(self.stock_query.get_price(company_symbol))

            # Sell stock
            self.money += current_price * num_shares_to_sell
            self.owned[company_symbol] -= num_shares_to_sell
            self.trades_remaining_today -= 1
            return True

        else:
            return False

    def buy_or_sell_action(self, company_symbol, num_shares):
        """
        Input: Symbol to decide on, and shares to buy or sell
        Output: None
        Side Effects: If price of company has increased, execute buy action, 
                      else execute sell
        """

        # Get current price
        self.stock_query.refresh_all()
        current_price = float(self.stock_query.get_price(company_symbol))

        # Get yesterday's price
        today = date.today()
        yesterday = today - timedelta(1)
        formatted_today = "%d-%d-%d" % (today.year, today.month, today.day)
        formatted_yesterday = "%d-%d-%d" % (yesterday.year, yesterday.month, yesterday.day)
        day_range = self.stock_query.get_historical(company_symbol, formatted_yesterday, formatted_today)
        yesterday_price = 0

        # If not 2 days of data, must be 1 day of data as a dict (for now, multiday
        # range not implemented yet)
        if len(day_range) != 2:
            yesterday_price = day_range['Close']
        else:
            # Must be list of 2 days of data, get yesterdays price
            yesterday_price = day_range[-1]['Close']

        # If price has risen since yesterday, buy stock
        if current_price > yesterday_price:
            return self.buy(company_symbol, num_shares)

        # If price has fallen since yesterday, sell stock
        elif current_price < yesterday_price:
            return self.sell(company_symbol, num_shares)

    def current_value(self):
        """
        Input: None
        Output: Current value as int of cash and all stocks held
        """
        self.stock_query.refresh_all()
        value = 0
        for symbol in self.company_dict.values():
            stock_value = float(self.stock_query.get_price(symbol))
            holding_value = stock_value * self.owned[symbol]
            value += holding_value

        value += self.money

        return value


class TwitterBot:
    """
    Based on sentiment analysis on recent tweets about a company, decides to
    buy, sell, or hold shares in the company 
    """
    def __init__(self, classifier, company_dict, api, money):

        self.classifier = classifier
        self.company_dict = company_dict
        self.api = api
        self.new_classified_dict = {}
        self.old_classified_dict = {}
        self.money = money
        self.owned = {}
        self.stock_query = StockQuery(company_dict.values())

        for company in self.company_dict.keys():
            self.new_classified_dict[company] = [0 for i in range(20)]
            self.old_classified_dict[company] = [0 for i in range(20)]

        for symbol in self.company_dict.values():
            self.owned[symbol] = 0

    def get_and_add_tweet(self, company):
        """
        Input: A tweet as a string
        Output: None
        Side Effects: Classifies tweet, updates running average for that company
        """
        # Get tweet about company from Twitter Streaming API
        full_tweet = self.api.search(q=company)
        tweet = full_tweet['statuses'][0]['text']

        # Classify that tweet        
        classification = self.classifier.predict([tweet])

        print classification, tweet

        # Map string classification to an integer
        sentiment = 0
        if classification == u'positive':
            sentiment = 1
        elif classification == u'negative':
            sentiment = -1

        # Update dictionaries
        del self.old_classified_dict[company][-1]
        self.old_classified_dict[company].insert(0, self.new_classified_dict[company][-1])
        del self.new_classified_dict[company][-1]
        self.new_classified_dict[company].insert(0, sentiment)

        return

    def threshold_crossed(self, company, threshold):
        """
        Input: A company name as a string
        Output: 0 if threshold crossed, 1 if detect increase in setiment,
                -1 if decrease in sentiment
        """ 
        # Get average sentiment for a company on new and old tweets
        new_sentiment = float(sum(self.new_classified_dict[company])) / float(len(self.new_classified_dict))
        old_sentiment = float(sum(self.old_classified_dict[company])) / float(len(self.old_classified_dict))

        # If the sentiment change crosses theshold, return specified code
        if abs(new_sentiment - old_sentiment) >= threshold:
            if new_sentiment > old_sentiment:
                return 1
            else:
                return -1

        else:
            return 0

    def buy(self, company_symbol, num_shares_to_buy):
        """
        Input: stock symbol for company, number of shares to buy
        Output: bool representing success or failure
        """

        print "----------"
        print "BUYING: ", company_symbol
        print "----------"

        # Get current price
        self.stock_query.refresh_all()
        current_price = float(self.stock_query.get_price(company_symbol))
        money_needed = current_price * num_shares_to_buy

        # If enough money, buy stock
        if self.money >= money_needed:
            self.money -= money_needed
            self.owned[company_symbol] += num_shares_to_buy
            return True
        else:
            return False

    def sell(self, company_symbol, num_shares_to_sell):
        """
        Input: stock symbol for company, number of shares to sell
        Output: bool representing success or failure
        """
        print "----------"
        print "SELLING: ", company_symbol
        print "----------"
        if self.owned[company_symbol] >= num_shares_to_sell:
            # Get current price        
            self.stock_query.refresh_all()
            current_price = float(self.stock_query.get_price(company_symbol))

            # Sell stock
            self.money += current_price * num_shares_to_sell
            self.owned[company_symbol] -= num_shares_to_sell

        else:
            return False

    def reset_sentiment(self, company):
        """
        Input: A company as a string
        Output: None
        Side Effects: resets sentiment list for that compnay to all 0s
        """
        self.new_classified_dict[company] = [0 for i in range(20)]
        self.old_classified_dict[company] = [0 for i in range(20)]

    def current_value(self):
        """
        Input: None
        Output: Current value as int of cash and all stocks held
        """
        self.stock_query.refresh_all()
        value = 0
        for symbol in self.company_dict.values():
            stock_value = float(self.stock_query.get_price(symbol))
            holding_value = stock_value * self.owned[symbol]
            value += holding_value

        value += self.money

        return value


class AllCapsTransformer(TransformerMixin):
    """
    Adds a features giving the number of ALL CAPS letters in a text
    """
    def transform(self, X, y=None, **fit_params):
        out_list = []

        for i in range(len(X)):
            next_dict = {}
            next_dict[X[i]] = sum(1 for c in X[i] if c.isupper())
            out_list.append(next_dict)

        return out_list

    def fit(self, X, y=None, **fit_params):
        return self

def get_number_phrases(tweet, emot_list):
    num_emot = 0
    for emot in emot_list:
        num_emot += tweet.count(emot)
    return num_emot

class PositiveEmoticonTransformer(TransformerMixin):
    """
    Adds a feature counting number of positive emoticons present in a text
    """
    def transform(self, X, y=None, **fit_params):
        out_list = []

        for i in range(len(X)):
            next_dict = {}
            next_dict[X[i]] = get_number_phrases(X[i], happy_emoticons)
            out_list.append(next_dict)

        return out_list

    def fit(self, X, y=None, **fit_params):
        return self

class NegativeEmoticonTransformer(TransformerMixin):
    """
    Adds a feature counting number of negative emoticons present in a text
    """
    def transform(self, X, y=None, **fit_params):
        out_list = []

        for i in range(len(X)):
            next_dict = {}
            next_dict[X[i]] = get_number_phrases(X[i], negative_emoticons)
            out_list.append(next_dict)

        return out_list

    def fit(self, X, y=None, **fit_params):
        return self

class PositiveWordsTransformer(TransformerMixin):
    """
    Adds a feature counting number of positive words present in a text
    """
    def transform(self, X, y=None, **fit_params):
        out_list = []

        for i in range(len(X)):
            next_dict = {}
            next_dict[X[i]] = get_number_phrases(X[i], positive_words)
            out_list.append(next_dict)

        return out_list

    def fit(self, X, y=None, **fit_params):
        return self

class NegativeWordsTransformer(TransformerMixin):
    """
    Adds a feature counting number of negative words present in a text
    """
    def transform(self, X, y=None, **fit_params):
        out_list = []

        for i in range(len(X)):
            next_dict = {}
            next_dict[X[i]] = get_number_phrases(X[i], negative_words)
            out_list.append(next_dict)

        return out_list

    def fit(self, X, y=None, **fit_params):
        return self

def train_and_evaluate(clf, x_train, x_test, y_train, y_test):
    """
    Trains and evaluates a given classifier
    Taken from http://nbviewer.ipython.org/github/gmonce/scikit-learn-book/
                blob/master/Chapter%202%20-%20Supervised%20Learning%20-%20Text%
               20Classification%20with%20Naive%20Bayes.ipynb
    """
    clf.fit(x_train, y_train)
    
    print "Accuracy on training set:"
    print clf.score(x_train, y_train)
    print "Accuracy on testing set:"
    print clf.score(x_test, y_test)
    
    y_pred = clf.predict(x_test)
    
    print "Classification Report:"
    print metrics.classification_report(y_test, y_pred)
    print "Confusion Matrix:"
    print metrics.confusion_matrix(y_test, y_pred)

    return clf

def make_unicode(input):
    """
    Takes strings and makes them unicode strings
    """
    if type(input) != unicode:
        input =  input.decode('utf-8', 'ignore')
        return input
    else:
        return input

def make_and_train_NB(x_train, x_test, y_train, y_test):
    """
    Trains a Naive Bayes classifier, using a bag-of-words approach
    """
    # Create Naive Bayes classifier
    clf = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB()),
    ])

    # Train classifier, print evaluative stats
    # Multiple NB classifiers were enumerated, the simple CountVectorizer
    # had the best performance, so the others were eliminated
    clf = train_and_evaluate(clf, x_train, x_test, y_train, y_test)

    return clf

def make_and_train_SVM(x_train, x_test, y_train, y_test):
    """
    Makes a Stochastic Gradient Descent classifier, learning on features for
    bag-of-words with tf-idf, number of positve/negative emoticons, and number
    of positive/negative words 
    """
    # Create SVM classifier
    clf = Pipeline([
        ('features', FeatureUnion([
            ('vect', CountVectorizer(ngram_range=(1,2))),
            ('tf_idf', TfidfVectorizer()),
            ('caps', Pipeline([
                ('all_caps', AllCapsTransformer()),
                ('dict', DictVectorizer())
            ])),
            ('pos_emot', Pipeline([
                ('positive_emot', PositiveEmoticonTransformer()),
                ('dict', DictVectorizer())
            ])),
            ('neg_emot', Pipeline([
                ('negative_emot', NegativeEmoticonTransformer()),
                ('dict', DictVectorizer())
            ])),
            ('pos_words', Pipeline([
                ('positive_words', PositiveWordsTransformer()),
                ('dict', DictVectorizer())
            ])),
            ('neg_words', Pipeline([
                ('negative_words', NegativeWordsTransformer()),
                ('dict', DictVectorizer())
            ]))
        ])),
        ('clf', SGDClassifier())
    ])

    clf = train_and_evaluate(clf, x_train, x_test, y_train, y_test)

    return clf

def main():

    # Get tweets from given .csv file
    in_data = list(csv.reader(open(data_path,"U")))

    # Convert all data to UTF-8
    for i in range(len(in_data)):
        in_data[i] = [make_unicode(a) for a in in_data[i]]


    # Partition tweets into training and testing sets
    shuffle(in_data)
    learning_cutoff = int(len(in_data) * percent_learning)
    learning_data = in_data[:learning_cutoff]
    testing_data = in_data[learning_cutoff:]

    # Partition data into tweets and tags
    x_train = [line[4] for line in learning_data]
    x_test = [line[4] for line in testing_data]
    y_train = numpy.array([line[1] for line in learning_data])
    y_test = numpy.array([line[1] for line in testing_data])

    # Get trained Naive Bayes classifier
    clf_NB = make_and_train_NB(x_train, x_test, y_train, y_test)

    # Get trained SVM classifer
    clf_SVM = make_and_train_SVM(x_train, x_test, y_train, y_test)

    # Setup twitter streaming client
    api = Twython(consumer_key, consumer_secret, access_token, access_token_secret)

    # Make TwitterBots
    NB_bot = TwitterBot(clf_NB, company_dict, api, start_money)
    SVM_bot = TwitterBot(clf_SVM, company_dict, api, start_money)

    # Make Baseline Bots
    random_bot = BaselineBot(company_dict, start_money, 1000000)
    check_yesterday_bot = BaselineBot(company_dict, start_money, trades_remaining_today)

    # Stay within rate limits
    max_tweets_per_hr  = 300
    pause_per_tweet =  3600 / max_tweets_per_hr
    pause_for_rate_limit = pause_per_tweet * len(company_dict.keys())

    while True:
        # Process NB_bot
        for company in NB_bot.company_dict.keys():
            NB_bot.get_and_add_tweet(company)
            detect_change = NB_bot.threshold_crossed(company, sentiment_threshold)
            if detect_change != 0:
                print company, "threshold crossed!! sentiment: ", detect_change
                if detect_change == 1:
                    NB_bot.buy(NB_bot.company_dict[company], shares_to_change)
                elif detect_change == -1:
                    NB_bot.sell(NB_bot.company_dict[company], shares_to_change)
                NB_bot.reset_sentiment(company)

        # Process SVM_bot
        for company in SVM_bot.company_dict.keys():
            SVM_bot.get_and_add_tweet(company)
            detect_change = SVM_bot.threshold_crossed(company, sentiment_threshold)
            if detect_change != 0:
                print company, "threshold crossed!! sentiment: ", detect_change
                if detect_change == 1:
                    SVM_bot.buy(NB_bot.company_dict[company], shares_to_change)
                elif detect_change == -1:
                    SVM_bot.sell(NB_bot.company_dict[company], shares_to_change)
                SVM_bot.reset_sentiment(company)

        # Process random_bot
        if random.random() < action_threshold:
            company_symbol = random.choice(random_bot.company_dict.values())
            if random.random() >= .5:
                random_bot.buy(company_symbol, shares_to_change)
            else:
                random_bot.sell(company_symbol, shares_to_change)

        # Process check_yesterday_bot
        if random.random() < action_threshold:
            company_symbol = random.choice(check_yesterday_bot.company_dict.values())
            check_yesterday_bot.buy_or_sell_action(company_symbol, shares_to_change)

        for bot in [NB_bot, SVM_bot, random_bot, check_yesterday_bot]:
            print "--------------------"
            print "NEXT BOT"
            print bot.current_value()
            print "--------------------"

        sleep(pause_for_rate_limit)

    return


if __name__ == '__main__':
    main()