Assuptions
==========

In order to implemented Naive Bayes language classification algorithm worked properly, assumed that:

- Data in dataset is quite well shuffled;
- Train and test sets included well distributed amount of all categories, that should be classified.
- Text well delimited (needn't to make text preprocessing like remove digits and any other special symbols);
- Categories in dataset is well distributed (not skewed).

Report
======

1. Brief overview of the libraries used in your implementation

    To reduce amount of code and make it more concise we use list of libraries, most appropriated for that purpose.
    This list include but not limited to:

    - pandas - for data manipulation: read/write from/to csv-file, data preprocessing (removing empty rows).
    - scikit_learn - for data manipulation: reorganizing/shuffling data, data splitting in train/test sets.
    - nltk - for text processing: tokenizing.

2. Complete overview of any data analyses, preparation, and/or feature extraction that you performed

    - Remove from input data samples with empty text column. Initially, dataset includes 2839 samples in form text / language properties.
    Among this set 78 samples have empty text property (feature). We remove them from input dataset.
    - Check out that data not skewed. Input data contains three language categories with next sample' distributions:

        - English     2055
        - Afrikaans   639
        - Nederlands  67

        As shown above, the initial data set is skewed quite enough, e.g. initial data imbalance is

       > English : Afrikaans : Nederlands roughly is 30 : 10 : 1

    It should be keep in mind when selecting an algorithm for language classification on that set of data.

    - Split input data set into train (80%) and test (20%) sets.
    This procedure must give a datasets that both include quite distributed set of samples for each categories of languages.
    e.g., for example, for Nederlands language in input data set we have only 67 samples and after dividing in into two separate
     parts both sets should contain distribution data for this category approximately 80/20.
     In this task we use **train_test_split** method from **scikit_learn** library to perform this task.

3. Complete overview of your model’s architecture

    **Multinomial Naive Bayes** (MNB) model is a supervised learning method and it is the simplest model applied in classification tasks.
    It is a probabilistic learning method based on term frequency in the document. In MNB classifier each document presented as set of words (tokens).
    Order of tokens in document is irrelevant.
    This method have pretty simple implementation and gives quite good results.
    The probability that the given document belongs to the given class is computed as a sum (in case of using log scale) of document' terms probability.

    The good overview of **Naive Bayes** model can be found [here](http://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html "Naive Bayes").

4. Complete overview of the training process, including detailed discussions of any specific techniques and/or algorithms used in your implementation

    The basic data structures for building classification model is:

     - Total #documents -  total number of documents in the corpus. An integer value.
     - classes - number of classes we want to classify. In language classification classifier we have three classes (languages): English, Afrikaans, and Nederlands. An integer value.
     - Text corpus - store for whole text corpus. String.
     - Text corpus per class - text corpus per class. In language classification classifier we should have one text corpus for each language. String.
     - Class tokens - list of text tokens per class. In language classification classifier we should have one list of tokens for each language. List of strings.
     - Class vocabulary - vocabulary of words with its frequencies per class and for whole text corpus. Dict

    Algorithm for training a model

    - On the first step, calc total #docs. Simply count number of samples in input data set.
    - Calc priors. For language classification classifier calc fraction of each language samples in train data. P(English)+P(Afrikaans) + P(Nederlands)= 1. Simply divide #docs each class by the total #docs.
    - Create text corpus, and text corpuses for each language class. Text corpus should contain all text from train data; text corpus per language class - the text from each language samples.
    - Tokenize text. For text corpus and classes tokenize text into the list of tokens (words).
    - Make dictionary for each text corpus: list of unique words for text corpus and language classes corpuses with their frequencies in text corpus and each language class corpus.
    - Finally, calc scores for each word in dictionary corpus. In result, each word in dictionary corpus must have frequency per language class (three frequencies in total).

    Algorithm for text classifying

    - On the first step, tokenize text to be classified. Split the input text into tokens.
    - Calc frequencies. Using by method from train model calc frequencies for list of tokens.
    - Calc predictions. On basis of frequencies, calculated in the previous step, calc similarity of the text to the given class.
    - Select prediction. Select max value of predicted value and assign the classified text to the selected class.

5. Overview of the testing process

    To make sure the algorithm implemented and worked properly, it was tested on test data set with predefined input and output data.
    The whole process is covered by unit tests.

    - Test of classifying - compare outcome with predefined labels.
    - Test of successful model saving - save and load model and perform classifying using by loaded model.
    - Test of unsuccessful model saving - check if failing saving model is handled appropriately.
    - Test of successful model loading - check if model loaded properly and classifying using by loaded model is working as expected.
    - Test of unsuccessful model loading - check if failing load model handled appropriately.

6. Overview and discussion of the results, including your model’s performance on each language, and how it can potentially be improved

    Full report with process of data analysis, training process, and evaluation of the model, included in Input_Data_Analysis_and_Model_Training.ipynb file.
    The most general metric for algorithm' evaluation is accuracy. Unfortunately, for skewed datasets this metric isn't useful.
    For implemented language classification algorithm' accuracy = 0.98. This is pretty good result, but, unfortunately,
    that metric hasn't enough representative, so another metrics come on scene.

    Precision, recall and F1-score.

    According to the report we can see that for two classes (English and Afrikaans) we have good metrics,
    but for Nederlands recall is quite small (0.38). It is caused by high imbalance.

    Improvements:

    - Unbiased imbalanced classes, i.e. Nederlands

## Bonus Questions

1. Discuss two machine learning approaches (other than the one you used in your language classification implementation) that would also be able to perform the task. Explain how these methods could be applied instead of your chosen method.

    For classification task can be used a sort of ML algorithms. We discuss couple of them: **Decision Trees** and **Support Vector Machines**.

    - Decision Tree. This method is more complicated for implementation and model is hard to evaluate, but, in contrast,
    it's can handle class' imbalance.
    - Support Vector Machines. This method with using Gaussian kernel. Complex implementation.

2. Explain the difference between supervised and unsupervised learning.

    - **supervised learning** is a meant to build predictive model by feeding learning data. Have a Cold start problem,
    e.g. cannot giva a meaningful output without feeding labeled data for training.
    Examples of supervised learning methods: Logistic Regression, Decision Trees, Random Forest, Neural Networks, etc.

    - **unsupervised learning** is a approach when the predicitve model needn't any learning data to work properly and give meaningful output.
     Examples of unsupervised learning methods: clustering, K-means, G-means, etc.

3. Explain the difference between classification and regression.

    Main difference between **classification** and **regression** is that the **classification** give **qualitative** results,
    that scoped by set of categories or classes, e.g. for given input output lays in scoped set of values, like color (white, blue, etc.),
    mood (good, sad, etc.), etc.

    The example of classification is the language classification task, where output lays in one from (English, Afrikaans, Nederlands).

    **regression**, in contrast, give **quantitative** results, e.g. measure, for example, the temperature, predicted rate of the purchased item, etc.

    The example of regression is the task of prediction rate of purchased item in recommender system, where output is a number.

4. In supervised classification tasks, we are often faced with datasets in which one of the classes is much more prevalent than any of the other classes. This condition is called class imbalance. Explain the consequences of class imbalance in the context of machine learning.

    Data imbalance is a situation, when classes in input dataset distributed quite unequal,
    e.g. for data set with two classes A and B, the ration of #samples for class A to the #samples for class B is 80 : 20.
    This problem is called **accuracy paradox** and can lead to improper algorithm' evaluation metrics, such as high accuracy.

5. Explain how any negative consequences of the class imbalance problem (explained in question 4) can be mitigated.

    There are many approaches to solve this problem. Some of them are:

    - Reduce imbalance by collecting more data or synthetically generate samples for minor classes (oversampling) or reducing #samples for major classes (undersampling).
    - Choose other evaluation metrics instead of accuracy, e.g. confusion matrix, precision,recall, and F-score.
    - Apply classification algorithms that resistant to class' imbalance problem, e.g. decision tree or random forest.

6. Provide a short overview of the key differences between deep learning and any non-deep learning method.

    In terms of NN, DNN differs from generic NN by presence #hidden layers > 1.
    More generally, non-deep learning depends on the input features substantially, e.g. if we select for learning wrong set of features,
    it comes to wrong algorithm' output.
    In contrast, deep learning can automatically adjust input features, and extract from them more meaningful information, and at the end
    gives more attractive results. The drawbacks of deep learning is the big amount of input data and computational complexity.