# -*- coding: iso-8859-15 -*-

import os, sys

import numpy as np, pymongo

from . import convert_into_str as _to_str
from . import convert_into_unicode as _to_unicode
from . import write_in_file as _write_in_file


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = '/'.join(CURRENT_PATH.split('/')[:-1])
DATA_PATH = BASE_PATH + '/datasets/data'

sys.path.append(BASE_PATH)

from sentiment_analysis import NEGATION_SETTINGS, SentimentAnalysis


def generate_training_data(database, collection, query):
    """Genera un conjunto de datos de entrenamiento.

    Esta generación se realiza con el propósito de encontrar los mejores
    parámetros para la clasificación de sentimientos. En específico,
    se intenta encontrar la mejor combinación de la negación y el
    léxicon de polaridad.

    paráms:
        database: str
            Base de datos de MongoDB que se utilizará.
        collection:
            Colección donde se encuentran los tweets etiquetados para
            entrenamiento.
        query:
            Filtro que se utilizará para recuperar los tweets de entrenamiento.
            Nótese que los tweets deben tener el campo "polarity".

    Example:

        >>> generate_training_data(database='tass_2017',
                                   collection='intertass',
                                   query={"$or": [{"dataset": "train"},
                                                  {"dataset": "development"}]})
    """
    four_label_homologation = {u'N+': 0, u'N': 0,
                               u'NEU': 1,
                               u'P': 2, u'P+': 2,
                               u'NONE': 3}

    client = pymongo.MongoClient()
    coll = client[database][collection]

    tweets = coll.find(filter=query,
                       projection=['tweet_id', 'content', 'polarity'],
                       sort=[('polarity', pymongo.ASCENDING),])

    tweets_ = [[_to_str(tweet['tweet_id']), _to_unicode(tweet['content']),
                _to_unicode(tweet['polarity']).upper()]
               for tweet in tweets if (tweet['content'] and len(tweet['content']) > 0)]

    client.close()

    tweets = None
    tweets = tweets_
    tweets_ = None

    output_path = DATA_PATH + '/train/' + collection
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for negation_id in NEGATION_SETTINGS.iterkeys():

        lexicons = np.random.choice(np.arange(1, 7),
                                    size=3, replace=False).tolist() +\
                   np.random.choice(np.arange(7, 16),
                                    size=3, replace=False).tolist() +\
                   np.random.choice(np.arange(16, 26),
                                    size=4, replace=False).tolist() +\
                   np.random.choice(np.arange(26, 31),
                                    size=2, replace=False).tolist()

        lexicons = np.random.choice(lexicons, size=6, replace=False).tolist()

        if np.random.choice(range(2), p=[.9, .1]) == 1:
            lexicons.append(31)

        negation_path = output_path + '/%s' % negation_id
        if not os.path.isdir(negation_path):
            os.mkdir(negation_path)

        for lexicon_id in lexicons:

            output_fname = negation_path +\
                           '/metafeatures-lexicon-%s.tsv' % lexicon_id
            if os.path.isfile(output_fname):
                continue

            clf = SentimentAnalysis(negation_id=negation_id,
                                    lexicon='lexicon-%i' % lexicon_id)

            documents = []
            four_label_polarities = []
            metafeatures_list = []

            for j, (tweet_id, content, polarity) in enumerate(tweets):
                try:
                    text, metafeatures = clf.preprocess_tweet(content)
                except:
                    _write_in_file(fname=negation_path + '/errors-1.log',
                                   content=tweet_id + '\n', mode='a')
                    continue

                metafeatures = metafeatures.reshape(1, metafeatures.shape[0])

                if j == 0:
                    metafeatures_list = metafeatures
                else:
                    if metafeatures_list.shape[1] == metafeatures.shape[1]:
                        metafeatures_list = np.vstack((metafeatures_list,
                                                       metafeatures))
                    else:
                        _write_in_file(fname=negation_path + '/errors-2.log',
                                       content=tweet_id + '\n', mode='a')
                        continue

                documents.append(_to_str(text))
                four_label_polarities.append(four_label_homologation[polarity])

            if not os.path.isfile(negation_path + '/tweets.txt'):
                np.savetxt(negation_path + '/tweets.txt',
                           np.array(documents, dtype=str), fmt='%s')

            if not os.path.isfile(negation_path + '/target-labels.dat'):
                np.savetxt(negation_path + '/target-labels.dat',
                           np.array(four_label_polarities, dtype=int), fmt='%i')

            np.savetxt(output_fname, metafeatures_list, fmt='%i', delimiter='\t')

            clf = None
