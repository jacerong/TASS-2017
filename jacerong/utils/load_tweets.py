# -*- coding: iso-8859-15 -*-

"""Guarda en una base de datos de MongoDB, una colección de tweets determinada."""


import datetime, re, xml.etree.ElementTree as ET

import pymongo


# Parámetros del repositorio que se utilizará en MongoDB
MONGO_SETTINGS = {'db': 'tass_2017'}


def _to_unicode(token):
    return token.decode('utf-8') if not isinstance(token, unicode) else token

def _to_str(token):
    return token.encode('utf-8') if not isinstance(token, str) else token

def _tweet_date_to_datetime_object(tweet_date):
    """Convierte la fecha de publicación del tweet en un objeto Python datetime.
    """
    m = re.compile(r"""(?P<year>20[0-9]{2})\-
                       (?P<month>[0-9]{2})\-
                       (?P<day>[0-9]{2})
                       .{1}
                       (?P<hour>[0-9]{2}):
                       (?P<minute>[0-9]{2}):
                       (?P<second>[0-9]{2})$""", re.X)

    r = m.match(_to_str(tweet_date))
    if not r:
        raise ValueError('El formato de la fecha del tweet no es válido.')
    else:
        return datetime.datetime(
            int(r.group('year')), int(r.group('month')), int(r.group('day')),
            int(r.group('hour')), int(r.group('minute')), int(r.group('second')))


def read_and_save_data(fname, corpus, dataset):
    """Lee un archivo que contiene una colección de tweets.

    parámetros:
        fname: str
            Nombre del archivo que contiene la colección de tweets.
        corpus: str
        dataset: str
            Toma alguno de estos valores: train | development | test

            De esta manera se diferencia cuál colección del corpus especificado se
            utilizará.

    salida:
        (ninguna)

    Por favor note que cada tweet se guarda (como un documento) en una misma base
    de datos, mientras que la colección es dada por el corpus al que este
    pertenece.
    """
    corpus = _to_str(corpus).lower()
    dataset = _to_str(dataset).lower()

    if dataset not in ['train', 'development', 'test']:
        raise ValueError('El conjunto de datos especificado no es válido.')

    client = pymongo.MongoClient()
    coll = client[MONGO_SETTINGS['db']][corpus]

    is_coll_empty = True if coll.count() == 0 else False

    # leer el archivo
    tweets = ET.parse(fname).getroot()

    for tweet in tweets:
        tweet_id = _to_unicode(tweet.find('tweetid').text)
        user = _to_unicode(tweet.find('user').text)
        content = _to_unicode(tweet.find('content').text)
        tweet_date = _tweet_date_to_datetime_object(tweet.find('date').text)
        lang = _to_unicode(tweet.find('lang').text)

        document = coll.find_one({'tweet_id': tweet_id})
        if (document is not None and dataset not in document['dataset']):
            coll.update_one(
                {'tweet_id': tweet_id},
                {"$set":
                    {'dataset': document['dataset'] + [dataset]}
                }, upsert=False)
            continue
        elif document is not None:
            continue

        document = {
            'tweet_id': tweet_id,
            'user': user,
            'content': content,
            'tweet_date': tweet_date,
            'lang': lang,
            'dataset': [dataset]}

        sentiment = tweet.find('sentiment')
        if (sentiment is not None and
                len(sentiment) == 1 and
                isinstance(sentiment[0].find('value').text, str) and
                len(sentiment[0].find('value').text) > 0):
            document['polarity'] = sentiment[0].find('value').text

        coll.insert_one(document)

        if is_coll_empty:
            coll.create_index([('tweet_id', pymongo.ASCENDING)],
                              unique=True)
            coll.create_index([('dataset', pymongo.ASCENDING)])

        is_coll_empty = False

    client.close()
