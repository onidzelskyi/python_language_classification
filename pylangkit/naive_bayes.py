import math
import pickle

from nltk import wordpunct_tokenize

from sklearn.model_selection import train_test_split


class NaiveBayes(object):
    """Naïve Bayes class
    Implementation of the simplest model used for classification task."""

    def __init__(self):
        """Define and initialize required fields."""
        self.data = None
        self.frac = None
        self.train = None
        self.cross_validation = None
        self.classes = {}
        self.corpus = {}
        self.vocab = dict(corpus=list())
        self.tokens = {}

    def fit(self, data: list, frac: float =.8) -> None:
        """Main process of training model for classification.
        @:arg data - input data set in form of list of tuples (text: str, label: str), where text is a sample string of
        given class label.
        @:arg frac - fraction of splitting input data set into train and test."""

        # Initialize internal fields
        self.data = data
        self.frac = frac

        # Split data into train and test sets
        self.split_data()

        # Calc #classes and their distribution
        for entry in self.train:
            category = entry[1]
            self.classes[category] = self.classes.get(category, 0) + 1

        # Calc class' priors
        for x in self.classes:
            self.classes[x] /= len(self.train)

        # Load train text corpus
        self.load_corpus()

        # tokenize text
        self.tokenize()

        # Vocabulary
        for key, value in self.tokens.items():
            self.vocab[key] = self.make_dictionary(value)

        # Main loop of training model
        for word in self.vocab['corpus']:
            freq = {}
            for cat in self.classes:
                freq[cat] = float(self.vocab[cat].get(word, 0) + 1) / float(len(self.tokens[cat]) + len(self.vocab['corpus']))
            self.vocab['corpus'][word] = freq

        # Calc model' accuracy to evaluate purposes
        self.calc_accuracy()

    def classify(self, data: list) -> list:
        """Classify input data using by pre-trained model."""
        res = []
        for text in data:
            words = wordpunct_tokenize(text)
            p = []
            for i, word in enumerate(words):
                if word not in self.vocab["corpus"]:
                    freq = {}
                    for cat in self.classes:
                        freq[cat] = float(1) / float(len(self.tokens[cat]) + len(self.tokens["corpus"]))
                    p.append(freq)
                else:
                    p.append(self.vocab["corpus"][word])
            predicted = {}
            for i in p:
                for cat in self.classes:
                    predicted[cat] = predicted.get(cat, 1.) + math.log(i[cat])

            for key in predicted.keys(): predicted[key] += math.log(self.classes[key])
            estim = sorted(predicted, key=predicted.get, reverse=True)

            # if no text present classify is empty
            if len(estim):
                res.append((text, estim[0],))

        return res

    def split_data(self):
        """Split data set to train/cross validation sets (80/20)."""
        self.train, self.cross_validation = train_test_split(self.data, test_size=0.2)

    def calc_accuracy(self):
        """Calc model' accuracy."""
        data = [entry[0] for entry in self.cross_validation]
        c = [entry[1] for entry in self.cross_validation]
        res = self.classify(data)
        hit = 0
        for i, val in enumerate(c):
            if (val.strip() == res[i][1].strip()): hit += 1

        accuracy = (float(hit) / float(len(self.cross_validation))) if len(self.cross_validation) else .0
        print("Accuracy: {}".format(accuracy))

    def load_corpus(self):
        """Load text corpus."""
        for text, cat in self.train:
            if "corpus" not in self.corpus: self.corpus["corpus"] = []
            self.corpus["corpus"].append(text)
            if cat not in self.corpus: self.corpus[cat] = []
            self.corpus[cat].append(text)

        for key in self.corpus: self.corpus[key] = " ".join(self.corpus[key])

    def tokenize(self):
        """Tokenize text on tokens"""
        self.tokens = dict((key, wordpunct_tokenize(value)) for (key, value) in self.corpus.items())

    def dump_model(self, file: str) -> None:
        """Write a pickled representation of trained model to the open file object file."""
        try:
            with open(file, 'wb') as fp:
                pickle.dump(self.classes, fp)
                pickle.dump(self.tokens, fp)
                pickle.dump(self.vocab, fp)

        except (IOError, OSError, FileNotFoundError) as err:
            raise IOError('Cannot save model to {}: {}'.format(file, err))

    def load_model(self, file: str) -> None:
        """Read trained model from the pickle data stored in a file."""
        try:
            with open(file, 'rb') as fp:
                self.classes = pickle.load(fp)
                self.tokens = pickle.load(fp)
                self.vocab = pickle.load(fp)

        except (IOError, EOFError) as err:
            raise IOError('Cannot load model from {}: {}'.format(file, err))

    @staticmethod
    def make_dictionary(text: list) -> dict:
        """Make dictionary of words with their' frequencies.
        @:arg text - list of words.
        @:return dict of words with their' frequencies."""
        d = {}
        for word in text:
            d[word] = d.get(word, 0) + 1

        return d
