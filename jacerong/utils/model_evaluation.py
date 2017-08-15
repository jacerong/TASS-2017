# -*- coding: iso-8859-15 -*-

"""Evaluación de un sistema en cuatro niveles de polaridad."""

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score,\
                            classification_report, precision_score, recall_score
from prettytable import PrettyTable


####################
# Global variables #
####################
STR_TO_INT_HOMOLOGATION = {u'N': 0,
                           u'NEU': 1,
                           u'P': 2,
                           u'NONE': 3}

INT_TO_STR_HOMOLOGATION = {0: 'N', 1: 'NEU', 2: 'P', 3: 'NONE'}


def model_evaluation(target_val_fname, predicted_val_fname):
    """Realiza la evaluación a un sistema / modelo.

    paráms:
        target_val_fname: str
            Nombre del archivo que contiene el "ground truth".
        predicted_val_fname: str
            Nombre del archivo que contiene las predicciones.

    Para ambos archivos, su estructura debe ser:
    tweet_id\tpolarity

    Donde "polarity" toma uno de los siguientes valores: ["N", "NEU", "P", "NONE"]
    """
    target_values = np.loadtxt(target_val_fname, delimiter='\t', dtype=str)
    predicted_values = np.loadtxt(predicted_val_fname, delimiter='\t', dtype=str)

    if (target_values.shape[0] != predicted_values.shape[0] or
            target_values.shape[1] != predicted_values.shape[1]):
        raise Exception('La estructura de los archivos no es válida.')

    y_true, y_pred = [], []

    for i in xrange(target_values.shape[0]):
        tweet_id = target_values[i,0]

        idx = np.where(predicted_values[:,0] == tweet_id)[0][0]

        y_true.append(STR_TO_INT_HOMOLOGATION[target_values[i,1]])
        y_pred.append(STR_TO_INT_HOMOLOGATION[predicted_values[idx,1]])

    # accuracy
    print 'Accuracy: %.4f\n' % accuracy_score(y_true, y_pred)

    # confusion_matrix
    data = confusion_matrix(y_true, y_pred)

    tbl = PrettyTable()
    tbl.field_names = ["Actual", "Predicted", "#"]

    for i in xrange(data.shape[0]):
        actual_class = INT_TO_STR_HOMOLOGATION[i]
        for j in xrange(data.shape[1]):
            predicted_class = INT_TO_STR_HOMOLOGATION[j]

            tbl.add_row([actual_class, predicted_class, '%i' % data[i,j]])

    print 'Confusion matrix'
    print tbl

    # category evaluation
    target_names = [INT_TO_STR_HOMOLOGATION[c]
                    for c in sorted(INT_TO_STR_HOMOLOGATION.keys())]

    print '\nCategory evaluation'
    print classification_report(y_true=y_true, y_pred=y_pred,
                                target_names=target_names)

    # precision, recall, f1
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = 2 * ((precision * recall) / (precision + recall))

    print '\nMacroaveraged Precision: %.4f' % precision
    print '\nMacroaveraged Recall: %.4f' % recall
    print '\nMacroaveraged F1: %.4f' % f1
