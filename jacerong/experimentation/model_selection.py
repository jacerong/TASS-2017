# -*- coding: iso-8859-15 -*-

import os, re, sys

import numpy as np, scipy.sparse as sp, scipy.stats as stats

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

from sklearn.externals import joblib

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = '/'.join(CURRENT_PATH.split('/')[:-1])
DATA_PATH = BASE_PATH + '/datasets/data'


def _write_in_file(fname, content, mode='w', makedirs_recursive=True):
    dir_ = '/'.join(fname.split('/')[:-1])
    if not os.path.isdir(dir_) and makedirs_recursive:
        os.makedirs(dir_)
    with open(fname, mode) as f:
        f.write(content)

def report_model_selection_results(negation_id, lexicon_id, analyzer,
                                   word_ngram_range, char_ngram_range,
                                   lowercase, max_df, min_df, binary,
                                   algo, C, cv_score,
                                   corpus):
    line = '{negation_id}\t{lexicon_id}\t{analyzer}\t'.\
           format(negation_id=negation_id, lexicon_id=lexicon_id,
                  analyzer=analyzer)

    line += '({min_w},{max_w})\t({min_c},{max_c})\t'.\
            format(min_w=word_ngram_range[0], max_w=word_ngram_range[1],
                   min_c=char_ngram_range[0], max_c=char_ngram_range[1])

    line += '%s\t' % ('True' if lowercase else 'False')
    line += '%.2f\t' % max_df
    line += '%i\t' % min_df
    line += '%s\t' % ('True' if binary else 'False')

    line += '%s\t' % algo
    line += '%.10f\t' % C
    line += '%.4f\n' % cv_score

    fname = CURRENT_PATH + '/%s-model-selection-results.tsv' % corpus
    with open(fname, 'a') as f:
        f.write(line)

def vectorize_tweet_collection(fname, analyzer, ngram_range, lowercase,
                               max_df, min_df, binary, split_underscore=True,
                               return_vectorizer=False):
    """Vectoriza una colección de tweets utilizando el esquema Tf-Idf.

    Retorna la matriz documentos-términos calculada utilizando el esquema Tf-Idf.
    La matriz retornada es dispersa, de tipo csr (scipy.sparse.csr_matrix).

    paráms:
        fname: str
            Nombre de archivo que contiene la colección de tweets.
        split_underscore: bool
            Divide una palabra que tiene el prefijo NEG_. Es decir, separa la
            palabra removiendo el guion bajo.
            NOTA: este parámetro es válido si analyzer == 'char'
    """
    vectorizer = TfidfVectorizer(analyzer=analyzer,
                                 ngram_range=ngram_range,
                                 lowercase=lowercase,
                                 max_df=max_df,
                                 min_df=min_df, binary=binary)

    tweets = []
    with open(fname) as f:
        for tweet in f:
            t = tweet.rstrip('\n').decode('utf-8')
            if analyzer == 'char' and split_underscore:
                t = t.replace(u'_', u' ').strip()
            tweets.append(t)

    if not return_vectorizer:
        return vectorizer.fit_transform(tweets)
    else:
        return vectorizer.fit_transform(tweets), vectorizer

def perform_grid_search(estimator, features, target_labels,
                        param_grid='default', n_jobs=4):
    # las siguientes probabilidades se calcularon de los resultados
    # consignados en 'intertass-model-selection-results.tsv'
    C_values = np.random.choice(np.power(2., np.arange(-5, 10, dtype=float)),
                                size=6,
                                replace=False,
                                p=[0.02, 0.016, 0.104, 0.146, 0.081, 0.119, 0.214,
                                   0.147, 0.059, 0.027, 0.019, 0.012, 0.014,
                                   0.011, 0.011])
    C_values = np.sort(C_values)

    if isinstance(param_grid, str) and param_grid == 'default':
        param_grid = {'C': C_values}

    clf = GridSearchCV(estimator=estimator,
                       param_grid=param_grid,
                       scoring='accuracy',
                       n_jobs=n_jobs,
                       cv=5,
                       refit=False)

    clf.fit(features, target_labels)

    return clf.best_params_, clf.best_score_

def build_vectorization_based_classifiers(corpus):
    """Método principal para construir clasificadores basados en vectorización.

    paráms:
        corpus: str
    """
    corpus = corpus.lower()

    ##################
    # ngram settings #
    ##################

    word_ngram_range = [(1, i) for i in xrange(1, 5)]
    char_ngram_range = [(i, j)
                        for i in xrange(2, 6) for j in xrange(2, 6) if i < j]

    ngram_params = ParameterGrid({'analyzer': ['word', 'char', 'both'],
                                  'word_ngram_idx': range(len(word_ngram_range)),
                                  'char_ngram_idx': range(len(char_ngram_range))})

    ngram_settings = []

    for params in ngram_params:
        if params['analyzer'] == 'word' and params['char_ngram_idx'] == 0:
            ngram_settings.append('analyzer:word-word_idx:%i-char_idx:%i' %
                                  (params['word_ngram_idx'], -1))
        elif params['analyzer'] == 'char' and params['word_ngram_idx'] == 0:
            ngram_settings.append('analyzer:char-word_idx:%i-char_idx:%i' %
                                  (-1, params['char_ngram_idx']))
        elif params['analyzer'] == 'both':
            ngram_settings.append('analyzer:both-word_idx:%i-char_idx:%i' %
                                  (params['word_ngram_idx'], params['char_ngram_idx']))

    ngram_params = None

    ###################
    # global settings #
    ###################

    model_selection = ParameterGrid({'ngram_settings': ngram_settings,
                                     'lowercase': [True, False],
                                     'max_df': [.85, .9],
                                     'min_df': [1, 2, 4],
                                     'binary': [True, False]})

    corpus_path = DATA_PATH + '/train/' + corpus

    for negation_id in os.listdir(corpus_path):

        negation_path = corpus_path + '/' + negation_id
        if not os.path.isdir(negation_path):
            continue

        fname = negation_path + '/tweets.txt'

        target_labels = np.loadtxt(negation_path + '/target-labels.dat',
                                   dtype=int)

        lexicons = []
        for metaftures_fname in os.listdir(negation_path):
            if re.match(r'metafeatures-lexicon-(?:[0-9]+)\.tsv$', metaftures_fname):
                lexicons.append(
                    '-'.join(metaftures_fname.rstrip('.tsv').split('-')[1:3]))

        for lexicon_id in lexicons:

            metaftures_fname = negation_path + '/metafeatures-%s.tsv' % lexicon_id
            metafeatures = np.loadtxt(metaftures_fname, dtype=float, delimiter='\t')
            metafeatures = sp.csr_matrix(metafeatures)

            random_idx = np.random.choice(len(model_selection),
                                          size=41, replace=False)

            for idx in random_idx:
                params = model_selection[idx]

                m = re.match('analyzer:([a-z]+)-word_idx:(-?[0-9]+)-char_idx:(-?[0-9]+)',
                             params['ngram_settings'])

                analyzer = m.group(1)
                w_idx = int(m.group(2))
                c_idx = int(m.group(3))

                ngram_range = None
                ngrams_features = None

                analyzers = ['word', 'char'] if analyzer == 'both' else [analyzer,]
                for analyzer in analyzers:

                    if analyzer == 'word':
                        ngram_range = word_ngram_range[w_idx]
                    else:
                        ngram_range = char_ngram_range[c_idx]

                    features_ = vectorize_tweet_collection(fname=fname,
                                                           analyzer=analyzer,
                                                           ngram_range=ngram_range,
                                                           lowercase=params['lowercase'],
                                                           max_df=params['max_df'],
                                                           min_df=params['min_df'],
                                                           binary=params['binary'])

                    if ngrams_features is None:
                        ngrams_features = features_
                    else:
                        ngrams_features = sp.hstack([ngrams_features, features_],
                                                    format='csr')

                features = sp.hstack([metafeatures, ngrams_features], format='csr')

                algorithms = ['LinearSVC', 'LogisticRegression']
                algo = np.random.choice(algorithms, p=[.37, .63])

                estimator = LinearSVC() if algo == 'LinearSVC' else LogisticRegression()

                best_params, best_score = perform_grid_search(
                    estimator=estimator,
                    features=features,
                    target_labels=target_labels)

                report_model_selection_results(
                    negation_id=negation_id,
                    lexicon_id=lexicon_id,
                    analyzer=m.group(1),
                    word_ngram_range=word_ngram_range[w_idx] if w_idx != -1 else (-1, -1),
                    char_ngram_range=char_ngram_range[c_idx] if c_idx != -1 else (-1, -1),
                    lowercase=params['lowercase'],
                    max_df=params['max_df'],
                    min_df=params['min_df'],
                    binary=params['binary'],
                    algo=algo,
                    C=best_params['C'],
                    cv_score=best_score,
                    corpus=corpus)

def prepare_level_one_data(corpus, n_classifiers=100):
    """Prepara los datos de nivel 'uno' que utilizarán los 'ensembles'.

    Los datos de nivel 'cero' corresponden a los datos originales provistos para
    entrenar modelos de clasificación supervisada. Entonces, las predicciones que
    se realizan durante la respectiva validación cruzada, se utilizan para entrenar
    los 'ensembles'; es a esto a que llamamos datos de nivel 'uno'.

    Referencias:
        [1] http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/stacked-ensembles.html
        [2] https://www.kaggle.com/general/18793 ("Strategy A")

    paráms:
        corpus: str
        n_classifiers: int
            Utilizar las predicciones de los mejores 'n' clasificadores para
            preparar los datos de nivel uno.

    Esta función, además de preparar los datos de nivel uno, realiza la persisten-
    cia tanto de los clasificadores como de los 'vectorizadores'.
    """
    corpus = corpus.lower()

    corpus_path = DATA_PATH + '/train/' + corpus

    # cargar los resultados de selección de modelos
    model_selection_results = np.loadtxt(
        CURRENT_PATH + '/%s-model-selection-results.tsv' % corpus,
        dtype=str, delimiter='\t')

    # los resultados entonces se ordenan descendentemente,
    # obteniéndose los respectivos índices
    indexes = np.argsort(np.array(model_selection_results[:,-1], dtype=float))[::-1]
    indexes = indexes[:n_classifiers]

    persistence_path = BASE_PATH + '/model_persistence/%s' % corpus
    if not os.path.isdir(persistence_path):
        os.makedirs(persistence_path)

    level_one_data_path = CURRENT_PATH + '/level-one-data/%s' % corpus
    if not os.path.isdir(level_one_data_path):
        os.makedirs(level_one_data_path)

    for idx in indexes:
        # Leer parámetros
        tmp = model_selection_results[idx,:]

        negation_id = tmp[0]
        lexicon_id = tmp[1]

        analyzer = tmp[2]
        word_ngram_range =\
            tuple([int(i) for i in re.sub('[\(\)]', '', tmp[3]).split(',')])
        char_ngram_range =\
            tuple([int(i) for i in re.sub('[\(\)]', '', tmp[4]).split(',')])

        lowercase = True if tmp[5] == 'True' else False

        max_df = float(tmp[6])
        min_df = int(tmp[7])

        binary = True if tmp[8] == 'True' else False

        algo = tmp[9]
        C = float(tmp[10])

        temp = None

        # Cargar colección de documentos, "ground truth" y "metafeatures"
        negation_path = corpus_path + '/' + negation_id
        if not os.path.isdir(negation_path):
            continue

        fname = negation_path + '/tweets.txt'

        target_labels = np.loadtxt(negation_path + '/target-labels.dat',
                                   dtype=int)

        metaftures_fname = negation_path + '/metafeatures-%s.tsv' % lexicon_id
        metafeatures = np.loadtxt(metaftures_fname, dtype=float, delimiter='\t')
        metafeatures = sp.csr_matrix(metafeatures)

        # Vectorizar colección de documentos
        ngram_range = None
        ngrams_features = None

        analyzers = ['word', 'char'] if analyzer == 'both' else [analyzer,]
        for analyzer in analyzers:

            ngram_range = word_ngram_range if analyzer == 'word' else char_ngram_range

            features_, vectorizer =\
                vectorize_tweet_collection(fname=fname,
                                           analyzer=analyzer,
                                           ngram_range=ngram_range,
                                           lowercase=lowercase,
                                           max_df=max_df,
                                           min_df=min_df,
                                           binary=binary,
                                           return_vectorizer=True)

            if ngrams_features is None:
                ngrams_features = features_
            else:
                ngrams_features = sp.hstack([ngrams_features, features_],
                                            format='csr')

            vectorizer_fname = '%s-%s-%i_%i-%s-%.2f-%i-%s.pkl' %\
                               (negation_id, analyzer,
                                ngram_range[0], ngram_range[1],
                                tmp[5], max_df, min_df, tmp[8])
            vectorizer_fname = persistence_path + '/vectorizers/' + vectorizer_fname

            # realizar persistencia del 'vectorizer'
            if not os.path.isfile(vectorizer_fname):
                joblib.dump(vectorizer, vectorizer_fname)

        features = sp.hstack([metafeatures, ngrams_features], format='csr')

        skf = list(StratifiedKFold(n_splits=5, shuffle=False, random_state=None).\
                   split(np.zeros(features.shape[0], dtype=float), target_labels))

        class_label_prediction = np.zeros(features.shape[0], dtype=int)
        class_proba_prediction = np.zeros((features.shape[0],
                                           np.unique(target_labels).shape[0]),
                                          dtype=float)

        for train_index, test_index in skf:

            X_train = features[train_index]
            y_train = target_labels[train_index]

            clf = LinearSVC(C=C) if algo == 'LinearSVC' else LogisticRegression(C=C)
            clf.fit(X_train, y_train)

            X_test = features[test_index]
            y_test = target_labels[test_index]

            class_label_prediction[test_index] = clf.predict(X_test)

            if algo == 'LogisticRegression':
                class_proba_prediction[test_index] = clf.predict_proba(X_test)

        class_label_fname = level_one_data_path + '/clf_%i-label.tsv' % idx
        class_proba_fname = level_one_data_path + '/clf_%i-proba.tsv' % idx

        np.savetxt(fname=class_label_fname, X=class_label_prediction, fmt='%i',
                   delimiter='\t')

        if algo == 'LogisticRegression':
            np.savetxt(fname=class_proba_fname, X=class_proba_prediction,
                       fmt='%.4f', delimiter='\t')

        # realizar persistencia del clasificador
        clf_fname = persistence_path + '/classifiers/' + 'clf_%i.pkl' % idx
        if not os.path.isfile(clf_fname):
            clf = LinearSVC(C=C) if algo == 'LinearSVC' else LogisticRegression(C=C)
            clf.fit(features, target_labels)
            joblib.dump(clf, clf_fname)

        _write_in_file(
            fname=CURRENT_PATH + '/%s-model-selection-filtered-results.tsv' % corpus,
            content='\t'.join(['%i' % idx,] + model_selection_results[idx,:].tolist()) + '\n',
            mode='a')

def find_low_correlated_combinations(corpus, n_classifiers=50):
    """Encuentra las combinaciones de más baja correlación.

    paráms:
        corpus: str
        n_classifiers: int
            Límite de clasificadores que pueden constituir una combinación.

    Nota: los datos de nivel uno deben haber sido generados; esto es, debió
    haberse ejecutado el método 'prepare_level_one_data'.
    """
    corpus = corpus.lower()

    level_one_data_path = CURRENT_PATH + '/level-one-data/%s' % corpus

    filtered_results = np.loadtxt(
        CURRENT_PATH + '/%s-model-selection-filtered-results.tsv' % corpus,
        dtype=str, delimiter='\t', usecols=(0, 10))

    logit_results =\
        filtered_results[np.where(filtered_results[:,1] == 'LogisticRegression')]

    low_correlated_combinations = {
        1: {'filtered_results': [[i] for i in xrange(filtered_results.shape[0])],
            'logit_results': [[i] for i in xrange(logit_results.shape[0])]}}

    output_fname = CURRENT_PATH +\
        '/%s-model-selection-low-correlated-combinations.tsv' % corpus

    for i in xrange(2, n_classifiers + 1):

        for which_results_to_use in low_correlated_combinations[i-1].iterkeys():

            results = filtered_results
            all_clf_ids = range(filtered_results.shape[0])

            if which_results_to_use == 'logit_results':
                results = logit_results
                all_clf_ids = range(logit_results.shape[0])

            correlation_results = []

            prev_results = low_correlated_combinations[i-1][which_results_to_use]
            for prev_rslt in prev_results:
                for j in all_clf_ids:
                    if j in prev_rslt or (i == 2 and prev_rslt[0] > j):
                        continue

                    # calcular la correlación entre todos
                    # los miembros de la combinación
                    tmp = prev_rslt[:]
                    tmp.append(j)

                    pearson_correlation = []

                    for y in xrange(len(tmp)):

                        labels_y = np.loadtxt(
                            level_one_data_path + '/clf_%s-label.tsv' % results[tmp[y],0],
                            dtype=int)

                        for z in xrange(len(tmp)):
                            if z <= y:
                                continue

                            labels_z = np.loadtxt(
                                level_one_data_path + '/clf_%s-label.tsv' % results[tmp[z],0],
                                dtype=int)

                            pearson_correlation.append(stats.pearsonr(labels_y,
                                                                      labels_z)[0])

                    correlation_results.append([tmp, np.mean(pearson_correlation),
                                                np.std(pearson_correlation)])

            correlation_results = np.array(correlation_results)

            min_crltn = np.amin(correlation_results[:,1])
            min_crltn_indexes = np.where(correlation_results[:,1] == min_crltn)[0]
            min_crltn_idx = np.argmin(correlation_results[min_crltn_indexes, 2])

            lowest_crltn = correlation_results[min_crltn_indexes[min_crltn_idx]]

            if i not in low_correlated_combinations.keys():
                low_correlated_combinations[i] = {}

            low_correlated_combinations[i][which_results_to_use] = [lowest_crltn[0],]

            output_str = '\t'.join(['%i' % i,
                'both' if which_results_to_use == 'filtered_results' else 'logit',
                ','.join([results[clf_id,0] for clf_id in lowest_crltn[0]]),
                '%.4f' % lowest_crltn[1],
                '%.4f' % lowest_crltn[2]])

            if not os.path.isfile(output_fname):
                _write_in_file(output_fname,
                               '#n\talgo\tclf_ids\tavg_correlation\tcorrelation_std\n')

            _write_in_file(output_fname, output_str + '\n', 'a')

def search_for_the_best_second_level_classifiers(corpus):
    """Buscar las mejores configuraciones de los clasificadores de segundo nivel.

    paráms:
        corpus: str

    Por otra parte, se listan los pre-requisitos para entrenar los clasificadores
    de segundo nivel:
        1. Haber generado los datos de nivel uno; función 'prepare_level_one_data'
        2. Haber encontrado las combinaciones de clasificadores con la más baja
           correlación, función 'find_low_correlated_combinations'
    """
    corpus = corpus.lower()
    
    level_one_data_path = CURRENT_PATH + '/level-one-data/%s' % corpus

    persistence_path = BASE_PATH + '/model_persistence/%s/stackers' % corpus
    if not os.path.isdir(persistence_path):
        os.makedirs(persistence_path)

    target_labels = None
    for dir_ in os.listdir(DATA_PATH + '/train/%s' % corpus):
        if os.path.isfile(DATA_PATH + '/train/%s/%s/target-labels.dat' % (corpus, dir_)):
            target_labels =\
                np.loadtxt(DATA_PATH + '/train/%s/%s/target-labels.dat' % (corpus, dir_),
                           dtype=int)
            break

    if target_labels is None:
        raise Exception('No se encontró un "ground truth".')

    n_classes = np.unique(target_labels).shape[0]

    # leer las mejores configuraciones de modelos
    filtered_results = np.loadtxt(
        CURRENT_PATH + '/%s-model-selection-filtered-results.tsv' % corpus,
        dtype=str, delimiter='\t', usecols=(0, 10))

    logit_results =\
        filtered_results[np.where(filtered_results[:,1] == 'LogisticRegression')]

    low_crltd_combinations = np.loadtxt(
        CURRENT_PATH + '/%s-model-selection-low-correlated-combinations.tsv' % corpus,
        dtype=str, delimiter='\t', usecols=(0, 1, 2))

    filtered_combi =\
        low_crltd_combinations[np.where(low_crltd_combinations[:,1] == 'both')]

    logit_combi =\
        low_crltd_combinations[np.where(low_crltd_combinations[:,1] == 'logit')]

    # determinar el número máximo de clasificadores
    # que pueden constituir una combinación
    n_classifiers = int(low_crltd_combinations[-1,0])

    low_crltd_combinations = None

    # reordenar los arreglos, dejando solo los id's
    filtered_results = filtered_results[:n_classifiers,0]
    logit_results = logit_results[:n_classifiers,0]

    filtered_combi = np.array(filtered_combi[-1, 2].split(','), dtype=str)
    logit_combi = np.array(logit_combi[-1, 2].split(','), dtype=str)

    # archivo donde guardar los resultados
    output_fname = CURRENT_PATH + '/%s-model-selection-ensemble-results.tsv' % corpus
    _write_in_file(output_fname,
                   '\t'.join(['#n_classifiers',
                              'ensemble_method',
                              'selection_method',
                              'algo',
                              'clf_ids',
                              'stacking_algo',
                              'hyperparameters',
                              'CV_score']) + '\n',
                   mode='w')

    for i in xrange(2, n_classifiers+1):

        for selection_method in ['low_crltn', 'best_ranked']:

            # unweighted average
            results = logit_results
            if selection_method == 'low_crltn':
                results = logit_combi

            clf_ids = results[:i]

            class_proba = {}
            for j in xrange(i):
                class_proba[j] = np.loadtxt(
                    level_one_data_path + '/clf_%s-proba.tsv' % clf_ids[j],
                    dtype=float, delimiter='\t')

            predicted_class_labels = []

            for j in xrange(target_labels.shape[0]):
                matrix = None
                for k in class_proba.iterkeys():
                    vector = class_proba[k][j,:].reshape(1, n_classes)
                    if matrix is None:
                        matrix = vector
                    else:
                        matrix = np.vstack((matrix, vector))
                predicted_class_labels.append(np.argmax(np.mean(matrix, axis=0)))

            predicted_class_labels = np.array(predicted_class_labels, dtype=int)

            output_str = '\t'.join(['%i' % i,
                                    'unweighted_average',
                                    selection_method,
                                    'logit',
                                    ','.join(clf_ids),
                                    '(None)',
                                    '(None)',
                                    '%.4f' % accuracy_score(target_labels,
                                                            predicted_class_labels)
                                  ])

            _write_in_file(output_fname, output_str + '\n', mode='a')

            # stacking
            results = filtered_results
            if selection_method == 'low_crltn':
                results = filtered_combi

            clf_ids = results[:i]

            matrix = None
            for j in xrange(i):
                vector = np.loadtxt(
                    level_one_data_path + '/clf_%s-label.tsv' % clf_ids[j],
                    dtype=int).reshape(target_labels.shape[0], 1)
                if matrix is None:
                    matrix = vector
                else:
                    matrix = np.hstack((matrix, vector))

            stacking_algos = {
                'logit': {'estimator': LogisticRegression(),
                          'param_grid': {'C': np.logspace(-3, 2, 6)}
                         },
                'SVM_rbf': {'estimator': SVC(),
                            'param_grid': {'kernel': ['rbf',],
                                           'C': np.logspace(-3, 2, 6),
                                           'gamma': np.logspace(-3, 2, 6)}
                           },
                }

            if i >= 7:
                stacking_algos['rf'] = {
                    'estimator': RandomForestClassifier(),
                    'param_grid': {
                        'n_estimators': np.array([10, 20, 40, 100]),
                        'criterion': ['gini', 'entropy'],
                        'max_features': np.arange(2,int(np.round(np.sqrt(i),0))+1)
                    }
                }

            stacking_cv_results = []

            for algo in stacking_algos.iterkeys():
                estimator = stacking_algos[algo]['estimator']
                param_grid = stacking_algos[algo]['param_grid']

                best_params, best_score =\
                    perform_grid_search(estimator=estimator,
                                        features=matrix,
                                        target_labels=target_labels,
                                        param_grid=param_grid,
                                        n_jobs=3)

                params_str = []
                for param in param_grid.iterkeys():
                    value = best_params[param]
                    if isinstance(value, int):
                        value = '%i' % value
                    elif isinstance(value, float):
                        value = '%.10f' % value
                    else:
                        value = str(value)
                    params_str.append('%s:%s' % (param, value))
                else:
                    params_str = ';'.join(params_str)

                stacking_cv_results.append([algo, params_str, best_score])

            stacking_cv_results = np.array(stacking_cv_results)
            algo, best_params, best_score =\
                stacking_cv_results[np.argmax(np.array(stacking_cv_results[:,2],dtype=float))]

            output_str = '\t'.join(['%i' % i,
                                    'stacking',
                                    selection_method,
                                    'both',
                                    ','.join(clf_ids),
                                    algo,
                                    best_params,
                                    best_score])
            _write_in_file(output_fname, output_str + '\n', mode='a')

            clf_fname = persistence_path + '/clf_%i-%s.pkl' % (i,selection_method)
            if not os.path.isfile(clf_fname):

                params = {}
                for param in best_params.split(';'):
                    key, value = param.split(':')
                    params[key] = value

                clf = None

                if algo == 'logit':
                    clf = LogisticRegression(C=float(params['C']))
                elif algo == 'SVM_rbf':
                    clf = SVC(kernel='rbf', C=float(params['C']),
                              gamma=float(params['gamma']))
                else:
                    clf = RandomForestClassifier(
                        n_estimators=int(params['n_estimators']),
                        criterion=params['criterion'],
                        max_features=int(params['max_features']))

                clf.fit(matrix, target_labels)

                joblib.dump(clf, clf_fname)
