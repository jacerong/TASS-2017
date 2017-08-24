0. Preliminary
==============

The installation this document guides was carried out on a ``Debian 8 "jessie"`` system, reason why the command line instructions correspond to the syntax of such an operating system.

This project uses the following third-party libraries / tools, and their installation is suggested before proceeding with this document:

- **FreeLing 4.0** [http://nlp.lsi.upc.edu/freeling/node/1].
- **foma** [https://fomafst.github.io/].
- **normalesp** [https://github.com/jacerong/normalesp].

How to install such third-party libraries / tools is outside the scope of this document. Therefore, please make sure each installation be successful.

0.1 System Requirements
-----------------------

Installation
    2G of available RAM.

Usage
    2G of available RAM.

0.2 Python Dependencies
-----------------------

This project requires ``Python 2.7``.

The modules as listed in the ``/requirements.txt`` file.

0.3 Compiling Polarity Lexicons
-------------------------------
Several lexicons are used to detect and count positive and negative polarity words in tweets. These polarity lexicons, as well as their compilation into finite-state transducers, are addressed in the ``/jacerong/datasets/lexicons/src/README.md`` document.

1. Preparing the Training Data
==============================
Throughout this section how to load a dataset annotated with different polarity labels, as well as how to prepare the data to train a sentiment analysis system, is discussed. To this end, the InterTASS dataset released for the TASS-2017 Task 1 will be used [#]_. However, the reader is encouraged to use other datasets and modify the code as necessary.

1.1 Loading an Annotated Dataset
--------------------------------
The ``read_and_save_data`` function in the ``/jacerong/utils/load_tweets.py`` file allows to process a xml file that contains an annotated dataset. As a result, tweets are saved as documents in MongoDB. How to load the training and development sets of the InterTASS corpus is described below.

Let ``/jacerong/datasets/data/TASS2017_T1_training.xml`` and ``/jacerong/datasets/data/TASS2017_T1_development.xml`` be the files that contain the training and development sets, respectively.

::

    $ cd /jacerong/utils/
    $ python
    >> train_set_fname = "/jacerong/datasets/data/TASS2017_T1_training.xml"
    >> dev_set_fname = "/jacerong/datasets/data/TASS2017_T1_development.xml"
    >> from load_tweets import read_and_save_data
    >> read_and_save_data(fname=train_set_fname, corpus="intertass", dataset="train", mongo_db="tass_2017")
    >> read_and_save_data(fname=dev_set_fname, corpus="intertass", dataset="development", mongo_db="tass_2017")

Thus, MongoDB stores the tweets of these sets as documents in the ``intertass`` collection, which belongs to the ``tass_2017`` database.

Finally, the structure of the XML data the ``read_and_save_data`` function expects is as follows::

    <tweets>
        <tweet>
            <tweetid>000000000000000001</tweetid>
            <user>user_1</user>
            <content>@user jajajaja la tuya y la d mucha gente seguro!! Pero yo no puedo sin mi melena me muero</content>
            <date>2016-08-23 22:25:29</date>
            <lang>es</lang>
            <sentiment>
                <polarity><value>N</value></polarity>
            </sentiment>
        </tweet>
        <tweet>
            <tweetid>000000000000000002</tweetid>
            <user>user_2</user>
            <content>Quiero mogollón a @user pero sobretodo por lo rápido que contesta a los wasaps</content>
            <date>2016-08-23 23:01:33</date>
            <lang>es</lang>
            <sentiment>
                <polarity><value>P</value></polarity>
            </sentiment>
        </tweet>
    </tweets>

The above example shows a set of 2 tweets. The first tweet is labeled with a negative (N) global polarity, while the second one is labeled with a positive (P) global polarity.

1.2 Preparing the Training Data
-------------------------------
Before addressing the objective of this section, let's see the structure of documents saved in MongoDB::

    $ mongo
    > use tass_2017
    > db.intertass.findOne();
    {
        "lang" : "es",
        "polarity" : "N",
        "dataset" : [
            "train"
        ],
        "content" : "@user jajajaja la tuya y la d mucha gente seguro!! Pero yo no puedo sin mi melena me muero",
        "tweet_id" : "000000000000000001",
        "user" : "user_1",
        "tweet_date" : ISODate("2016-08-23T22:25:29Z")
    }

Where the ``dataset`` field is an array that holds the sets to which a document belongs. In this way, all the tweets of a given set can be filtered.

Having said the above, let's proceed to prepare the data that will be used to train a sentiment analysis system. Such a process consists in applying a rule-based text normalizer and a spell checking program to a set of tweets retrieved from MongoDB. Then, several training sets, in the form of flat text files, are generated by using different instances of the negation detection module. Likewise, several randomly chosen polarity lexicons are used to generate different basic feature vectors for each instance of the negation detection module.

The ``generate_training_data`` function in the ``/jacerong/utils/training_data.py`` file allows to generate different training sets, as well as different basic feature vectors. To this end, the training and development sets of the InterTASS corpus saved in MongoDB are used.

**1. Initializing the required services**. To do that, change directory to ``/jacerong/``, open a new terminal, and type the following instructions::

    $ python
    >>> from sentiment_analysis import _switch_sentiment_services
    >>> _switch_sentiment_services('on')

It is strongly recommended **NOT** to close this terminal or type other Python instructions.

In the same way, please make sure the required services of the ``normalesp`` program are running.

To stop the services, type the following instruction::

    >>> _switch_sentiment_services('off')

**2. Preparing and generating the training data**. To do that, open a new terminal, and type the following instructions::

    $ cd /jacerong/utils/
    $ python
    >> from training_data import generate_training_data
    >> generate_training_data(database='tass_2017', collection='intertass', query={"$or": [{"dataset": "train"}, {"dataset": "development"}]})

The generated data will be put in the ``/jacerong/datasets/data/`` path.

*This process may take several hours to complete*.

2. Training a Sentiment Analysis System
=======================================
In this section the process of training first-level classifiers, as well as how to optimally combine their predictions to obtain better final predictions, is described.

**1. Training first-level classifiers**. A machine learning classifier, or first-level classifier, receives a feature vector and predicts a class label or probability estimates, i.e. the probability of a tweet to be of a certain class. Whichever the prediction be, it is denominated level-one prediction.

Previously several training sets were generated by using different instances of the negation detection module. For each of these instances, different basic feature vectors were also generated using several randomly chosen polarity lexicons. The goal is hence to find the best parameter settings for first-level classifiers; this is, the parameter settings that achieve the maximum cross-validation accuracy values. The search for these parameter settings also includes the vectorizer, which transforms a text into a feature vector (n-gram features), and the algorithm utilized to develop a supervised learning approach.

The ``build_vectorization_based_classifiers`` function in the ``/jacerong/experimentation/model_selection.py`` file performs the search described above.

::

    $ cd /jacerong/experimentation/
    $ python
    >> from model_selection import build_vectorization_based_classifiers
    >> build_vectorization_based_classifiers('intertass')

As a result, this Python function creates the ``/jacerong/experimentation/intertass-model-selection-results.tsv`` file whose structure is described below:

- *negation_id*: parameter setting used to instantiate the negation detection module. This identifier corresponds to one of the keys of the ``NEGATION_SETTINGS`` dictionary in the ``/jacerong/sentiment_analysis.py`` file.
- *lexicon_id*.
- *analyzer*: how the n-gram feature vector is made. ``word`` or ``char`` means the feature vector is made of word or character n-grams, respectively; ``both``, instead, means the feature vector is made by concatenating word and character n-grams.
- *word_ngram_range*: range of n-values for different word n-grams to be extracted. ``(-1,-1)`` when ``analyzer`` is ``char``.
- *char_ngram_range*: range of n-values for different character n-grams to be extracted. ``(-1,-1)`` when ``analyzer`` is ``word``.
- *lowercase*: if ``True``, all characters are converted to lowercase before tokenizing.
- *max_df*: tokens whose document frequency is higher than this threshold are ignored.
- *min_df*: tokens whose document frequency is lower than this threshold are ignored.
- *binary*: if ``True``, the term frequency (tf) is binary.
- *algo*: whether the algorithm utilized is ``LogisticRegression`` or ``LinearSVC``.
- *C*: penalty parameter for the algorithm.
- *cv_score*: mean cross-validated score for the parameter setting.

*This process may take several days to complete*.

**2. Filtering the best first-level classifiers and preparing level-one data**. The ``prepare_level_one_data`` function in the ``/jacerong/experimentation/model_selection.py`` file filters the best ``n`` first-level classifiers according to their predictive performance on cross validation. Then, it persists these first-level classifiers and the vectorizers they use in the ``/jacerong/model_persistence/intertass/classifiers/`` and ``/jacerong/model_persistence/intertass/vectorizers/`` paths, respectively, and saves their out-of-fold predictions as the level-one data in the ``/jacerong/experimentation/level-one-data/intertass/`` path. These out-of-fold predictions will be used to train second-level classifiers, i.e. the ones that take level-one predictions and then optimally combine them to obtain better final predictions.

::

    $ cd /jacerong/experimentation/
    $ python
    >> from model_selection import prepare_level_one_data
    >> prepare_level_one_data('intertass', 100)

Thus, the best 100 first-level classifiers are filtered.

*This process may take several minutes to complete*.

**3. Finding the less-correlated combinations of first-level classifiers**. Empirical findings indicate that the less-correlated combinations of first-level classifiers achieves top results (Cerón-Guzmán, 2016). Therefore, the Pearson correlation for all the out-of-fold predictions of the best first-level classifiers will be calculated using the ``find_low_correlated_combinations`` function in the ``/jacerong/experimentation/model_selection.py`` file.

::

    $ cd /jacerong/experimentation/
    $ python
    >> from model_selection import find_low_correlated_combinations
    >> find_low_correlated_combinations('intertass', n_classifiers=50)

As a result, the less-correlated combinations of 2 and up to ``n_classifiers`` first-level classifiers are determined.

*This process may take several minutes and even hours to complete*.

**4. Searching for the best second-level classifiers**. To take level-one predictions and then optimally combine them two ensemble methods were implemented, namely: stacking and averaging. Regarding this, the ``search_for_the_best_second_level_classifiers`` function in the ``/jacerong/experimentation/model_selection.py`` file searches for the best ensembles based on stacking and based on averaging. For such a search, the function also selects the classifiers with the highest out-of-fold prediction accuracy values to constitute an ensemble; take into account that the other selection method is based on the less-correlated combinations. As a final point, the ensembles based on stacking are persisted in the ``/jacerong/model_persistence/intertass/stackers/`` path.

::

    $ cd /jacerong/experimentation/
    $ python
    >> from model_selection import search_for_the_best_second_level_classifiers
    >> search_for_the_best_second_level_classifiers('intertass')

As a result, this Python function creates the ``/jacerong/experimentation/intertass-model-selection-ensemble-results.tsv`` file whose structure is described below:

- *n_classifiers*: number of classifiers that constitute the ensemble.
- *ensemble_method*: {'unweighted_average', 'stacking'}.
- *selection_method*: {'low_crltn', 'best_ranked'}.
- *algo*: ``both`` when ``selection_method`` takes the ``stacking`` value; otherwise, ``logit``.
- *clf_ids*: comma-separated list of the classifiers that constitute the ensemble. One id in the list corresponds to the id-th row in the ``/jacerong/experimentation/intertass-model-selection-results.tsv`` file (zero-based numbering)
- *stacking_algo*: {'logit', 'SVM_rbf', 'rf'} when the ``ensemble_method`` field takes the ``stacking`` value; otherwise, ``(None)``. ``logit``, ``SVM_rbf``, ``rf`` stand for Logistic Regression, Support Vector Machine with ``rbf`` kernel, and Random Forest, respectively. In other words, this is the algorithm utilized by the second-level classifier.
- *hyperparameters*: hyperparameters for the algorithm utilized by the second-level classifier.
- *CV_score*: mean cross-validated score for the ensemble.

*This process may take several minutes to complete*.

**NOTE**: because the scikit-learn implementation of the ``Linear Support Vector Classification`` algorithm does not support the ``predict_proba`` method, only first-level classifiers that utilize the ``Logistic Regression`` algorithm are eligibles to constitute ensembles based on averaging.

.. [#] The InterTASS dataset can be downloaded from the workshop official page as indicated `there <http://www.sepln.org/workshops/tass/2017/#datasets>`_.
