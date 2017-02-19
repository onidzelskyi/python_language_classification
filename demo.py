"""Demo of language classification task."""
import pandas as pd

import pylangkit.naive_bayes as nb


N = 5


def main():
    """Demonstrate sample of using the Naive Bayes model for text language classification."""
    # Load train and test sets
    df_train = pd.read_csv('resources/lang_data_train.csv')
    df_test = pd.read_csv('resources/lang_data_test.csv')

    # Create and load trained model
    model = nb.NaiveBayes()
    # model.load_model('resources/trained_model.pickle')

    # Or, instead, train new model from dataset
    model.fit(df_train.as_matrix())

    # Save model to the file for re-emergence
    # model.dump_model('resources/trained_model.pickle')

    # Classify samples' by languages using by trained model
    predicted = model.classify(df_test.text.as_matrix())

    # Sow first N classes
    for i in range(N):
        print('Text: {obj[0][0]}; Predicted language: {obj[0][1]}; Actual language: {obj[1]}'.format(obj=(predicted[i], df_test.ix[i].language)))

    # Show mismatched classes
    mismatched = list(filter(lambda x: x[0][1] != x[1], zip(predicted, df_test.language)))

    print('Fraction of mismatched samples is: {}\n{}'.format(len(mismatched) / len(df_test.language), mismatched))


if __name__ == '__main__':
    main()
