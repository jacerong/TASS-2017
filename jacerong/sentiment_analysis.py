# -*- coding: iso-8859-15 -*-

import os, re, sys, warnings, xml.etree.ElementTree as ET

import numpy as np, scipy.sparse as sp, scipy.stats as stats
from sklearn.externals import joblib
from unidecode import unidecode

from utils import convert_into_str as _to_str
from utils import convert_into_unicode as _to_unicode
from utils import remove_accent_marks as _deaccent
from utils import write_in_file as _write_in_file
from utils.services import switch_freeling_server as _switch_freeling_server
from utils.services import analyze_morphologically as _analyze_morphologically
from utils.services import switch_flookup_server as _switch_flookup_server
from utils.services import foma_string_lookup as _foma_string_lookup
from utils.services import perform_spell_checking as _perform_spell_checking


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

###############################
# read the configuration file #
###############################

config = ET.parse(CURRENT_PATH + '/config/general.xml').getroot()

IP_ADDRESS = config[1][1].text

config = None

#############################
# List of polarity lexicons #
#############################

LEXICON_PATH = CURRENT_PATH + '/datasets/lexicons/bin'

LEXICONS = {'lexicon-%i' % i: [LEXICON_PATH + '/lexicon-%i.bin' % i,
                               IP_ADDRESS,
                               '421%02d' % i
                              ] for i in xrange(1, 32)}

#########################
# Othe config variables #
#########################

NEGATION_SETTINGS = {'without_negation': {'negation_enabled': False,
                                          'negation_type': 'default',
                                          'scope': -1,
                                          'tags': ['V', 'A', 'R', 'NC']},
                     'negation_1': {'negation_enabled': True,
                                    'negation_type': 'default',
                                    'scope': -1,
                                    'tags': ['V', 'A', 'R', 'NC']},
                     'negation_2': {'negation_enabled': True,
                                    'negation_type': 'filtered',
                                    'scope': -1,
                                    'tags': ['V', 'A']},
                     'negation_3': {'negation_enabled': True,
                                    'negation_type': 'filtered',
                                    'scope': -1,
                                    'tags': ['V', 'A', 'R']},
                     'negation_4': {'negation_enabled': True,
                                    'negation_type': 'filtered',
                                    'scope': -1,
                                    'tags': ['V', 'A', 'NC']},
                     'negation_5': {'negation_enabled': True,
                                    'negation_type': 'filtered',
                                    'scope': -1,
                                    'tags': ['V', 'A', 'R', 'NC']},
                     'negation_6': {'negation_enabled': True,
                                    'negation_type': 'filtered',
                                    'scope': 1,
                                    'tags': ['V', 'A']},
                     'negation_7': {'negation_enabled': True,
                                    'negation_type': 'filtered',
                                    'scope': 1,
                                    'tags': ['V', 'A', 'R']},
                     'negation_8': {'negation_enabled': True,
                                    'negation_type': 'filtered',
                                    'scope': 1,
                                    'tags': ['V', 'A', 'NC']},
                     'negation_9': {'negation_enabled': True,
                                    'negation_type': 'filtered',
                                    'scope': 1,
                                    'tags': ['V', 'A', 'R', 'NC']},
                     'negation_10': {'negation_enabled': True,
                                     'negation_type': 'filtered',
                                     'scope': 2,
                                     'tags': ['V', 'A']},
                     'negation_11': {'negation_enabled': True,
                                     'negation_type': 'filtered',
                                     'scope': 2,
                                     'tags': ['V', 'A', 'R']},
                     'negation_12': {'negation_enabled': True,
                                     'negation_type': 'filtered',
                                     'scope': 2,
                                     'tags': ['V', 'A', 'NC']},
                     'negation_13': {'negation_enabled': True,
                                     'negation_type': 'filtered',
                                     'scope': 2,
                                     'tags': ['V', 'A', 'R', 'NC']},
                     'negation_14': {'negation_enabled': True,
                                     'negation_type': 'filtered',
                                     'scope': 3,
                                     'tags': ['V', 'A']},
                     'negation_15': {'negation_enabled': True,
                                     'negation_type': 'filtered',
                                     'scope': 3,
                                     'tags': ['V', 'A', 'R']},
                     'negation_16': {'negation_enabled': True,
                                     'negation_type': 'filtered',
                                     'scope': 3,
                                     'tags': ['V', 'A', 'NC']},
                     'negation_17': {'negation_enabled': True,
                                     'negation_type': 'filtered',
                                     'scope': 3,
                                     'tags': ['V', 'A', 'R', 'NC']}}

####################
# Global variables #
####################

PUNCTUATION_REPLACEMENTS = (
    (u'[\u00AB\u00BB\u201C\u201D\u201F\u275D\u275E\u301D\u301E\uFF02]', u'"'),
    (u'[\u2018\u2019\u2039\u203A\u201B\u275B\u275C\u276E\u275F\u0060\u00B4]', u"'"),
    (u'[\u201A\u201E]', u','))

HTML_ENTITIES_REPLACEMENTS = ((u'&nbsp;', ' '), (u'&lt;', u'<'), (u'&gt;', u'>'),
                              (u'&amp;', u'&'))

UNICODE_EMOJI_POS = (u'\U0001F600', u'\U0001F601', u'\U0001F602', u'\U0001F603',
                     u'\U0001F604', u'\U0001F605', u'\U0001F606', u'\U0001F609',
                     u'\U0001F60A', u'\U0001F60B', u'\U0001F60E', u'\U0001F60D',
                     u'\U0001F618', u'\U0001F617', u'\U0001F619', u'\U0001F61A',
                     u'\u263A', u'\U0001F642', u'\U0001F917', u'\U0001F607',
                     u'\U0001F60F', u'\U0001F61B', u'\U0001F61C', u'\U0001F61D',
                     u'\U0001F643', u'\U0001F63A', u'\U0001F638', u'\U0001F639',
                     u'\U0001F63B', u'\U0001F63D', u'\U0001F44D')

UNICODE_EMOJI_NEG = (u'\U0001F623', u'\U0001F625', u'\U0001F62B', u'\u2639',
                     u'\U0001F641', u'\U0001F612', u'\U0001F614', u'\U0001F615',
                     u'\U0001F616', u'\U0001F632', u'\U0001F61E', u'\U0001F61F',
                     u'\U0001F624', u'\U0001F622', u'\U0001F62D', u'\U0001F626',
                     u'\U0001F627', u'\U0001F628', u'\U0001F629', u'\U0001F62C',
                     u'\U0001F620', u'\U0001F631', u'\U0001F635', u'\U0001F621',
                     u'\U0001F620', u'\U0001F4A9', u'\U0001F640', u'\U0001F63F',
                     u'\U0001F63E', u'\U0001F44E')

# las siguientes expresiones regulars capturan el 96% de los emoticones textuales
# fuente: <http://sentiment.christopherpotts.net/tokenizing.html>
EMOTICON_STRING_L2R = r"""
    (
        [<>]?
        [:;=8]                     # eyes
        [\-o\*\']?                 # optional nose
        [\)\]\(\[DpP/\:\}\{@\|\\]  # mouth
    )
    (.|\Z)
    """

EMOTICON_STRING_R2L = r"""
    (\A|.)
    (
        [\)\]\(\[DpP/\:\}\{@\|\\]  # mouth
        [\-o\*\']?                 # optional nose
        [:;=8]                     # eyes
        [<>]?
    )
    """

ALPHANUM_RE = re.compile(
    u'[0-9a-zA-Z\xe1\xe9\xed\xf3\xfa\xf1\xc1\xc9\xcd\xd3\xda\xd1]',
    re.U)

NEGATION_WORDS = (u'denegar', u'jamas', 'jamás'.decode('utf-8'), u'nada',
                  u'nadie', u'negar', u'negativa', u'negativo', u'ni', u'ningun',
                  'ningún'.decode('utf-8'), u'ninguna', u'ninguno', u'no',
                  u'nunca', u'rehusar', u'tampoco')

ONE_LETTER_WORDS = [u'a', u'e', u'o', u'u', u'y']

CONTRACTIONS = ((u'de el', u'del'), (u'a el', u'al'))


def _detect_negation(analysis, type_, scope, tags):
    """Detectar contextos de negación en el tweet.

    paráms:
        analysis: list
            resultado del análisis morfológico de FreeLing
        type_: str
            toma alguno de estos valores: default | filtered
            default: enfoque más común en la literatura: añadir prefijo NEG_
                     a todos los tokens después de una palabra de negación
                     y hasta un signo de puntuación
            filtered: añade prefijo NEG_ sólo a tokens con ciertos PoS tags
                      y hasta un signo de puntuación
        scope: int
            alcance de la negación
            -1: hasta un signo de puntuación
            i > 0: los iésimos tokens después de una palabra de negación y
                hasta un signo de puntuación
                NOTA: aplica solo para type_ = filtered
        tags: list
            cuáles PoS tags a considerar para la negación
            NOTA: aplica solo para type_ = filtered

    salida:
        analyis: list
            el mismo analysis de entrada, pero por cada token,
            representado en un array, se agrega al final un valor bool
            que es True si va negado, False en caso contrario
        negated_contexts: int
            número de contextos negados
    """
    analysis_ = analysis
    negated_contexts = 0
    for i, sentence in enumerate(analysis):
        scope_cont = 0
        is_negation_enabled = False
        for j, (token, lemma, tag) in enumerate(sentence):
            token = _to_unicode(token)
            lemma = _to_unicode(lemma)
            tag = _to_unicode(tag)
            if (re.match(r'[\w]+', token, re.U)
                    and re.search(r'[^0-9_]', token, re.U)
                    and not re.match(r'EMO_(?:POS|NEG)$', token, re.U)):
                if (token.lower() in NEGATION_WORDS
                        or lemma.lower() in NEGATION_WORDS):
                    analysis_[i][j].append(False)
                    if not is_negation_enabled:
                        negated_contexts += 1
                    is_negation_enabled = True
                elif is_negation_enabled:
                    if type_ == 'default':
                        analysis_[i][j].append(True)
                    elif ((tag[0].upper() in tags or tag[:2].upper() in tags)
                            and (scope == -1 or scope_cont < scope)):
                        analysis_[i][j].append(True)
                        scope_cont += 1
                    else:
                        analysis_[i][j].append(False)
                else:
                    analysis_[i][j].append(False)
            else:
                if tag.lower() in (u'fat', u'fc', u'fd', u'fit', u'fp', u'fx'):
                    scope_cont = 0
                    is_negation_enabled = False
                analysis_[i][j].append(False)

    return analysis_, negated_contexts

def _capture_emoticons(text):
    """Capturar emoticones textuales y con representación en unicode."""
    # reemplazar emoji's
    for i in xrange(2):
        emoji_list = UNICODE_EMOJI_POS if i == 0 else UNICODE_EMOJI_NEG
        for emoji in emoji_list:
            text = text.replace(emoji, ' EMO_POS ' if i == 0 else ' EMO_NEG ')

    # capturar emoticones textuales
    for i in xrange(2):
        emoticon_re = re.compile(
            EMOTICON_STRING_L2R if i == 0 else EMOTICON_STRING_R2L,
            re.VERBOSE | re.U)
        for match in emoticon_re.findall(text):
            # las siguientes dos decisiones reducen la tasa de falsos positivos
            if (i == 0 and match[0][-1] in (u'D', u'p', u'P')
                    and len(match[1]) > 0 and ALPHANUM_RE.match(match[1])):
                continue
            elif (i == 1 and match[1][0] in (u'D', u'p', u'P')
                    and len(match[0]) > 0 and ALPHANUM_RE.match(match[0])):
                continue

            # a continuación, se homologan los emoticones
            emoticon = match[0] if i == 0 else match[1]
            if (re.match(r'[<>]?[:;=8][\-o\*\']?[\)DpP\]\}]', emoticon, re.U)
                    or re.match(r'[\[\(\{][\-o\*\']?[:;=8][<>]?', emoticon, re.U)):
                text = text.replace(emoticon, ' EMO_POS ', 1)
            elif (re.match(r'[<>]?[:;=8][\-o\*\']?[\[\(\{\|@]', emoticon, re.U)
                    or re.match(r'[\)D\]\}][\-o\*\']?[:;=8][<>]?', emoticon, re.U)):
                text = text.replace(emoticon, ' EMO_NEG ', 1)

    # homologar otros emoticones
    emoticon_list = ((r'<3', u' EMO_POS '), (r':\$', u' EMO_NEG '),
                     (r'(?:\A|\W)?[xX]D(?:\W|\Z)?', u' EMO_POS '),
                     (r'(?:\A|\W)?D[xX](?:\W|\Z)?', u' EMO_NEG '))
    for emoticon, replacement in emoticon_list:
        text = re.sub(emoticon, replacement, text, flags=re.U)

    return re.sub(r'[ ]+', ' ', text, flags=re.U).strip()

def _normalize_punctuation_marks(text):
    """Normalizar signos de puntuación.

    Inserta espacios cuando es necesario.
    """
    # comillas dobles
    for i, match in enumerate(re.findall(r'(\A|.)"(.|\Z)', text, flags=re.U|re.M)):
        if i % 2 == 0 and len(match[0]) > 0:
            text = text.replace(''.join([match[0] , '"', match[1]]),
                                ''.join([match[0] , ' "', match[1]]),
                                1)
        elif i % 2 != 0 and len(match[1]) > 0:
            text = text.replace(''.join([match[0] , '"', match[1]]),
                                ''.join([match[0] , '" ', match[1]]),
                                1)

    # comillas simples
    text = re.sub(u"""'(.+?)'""", r" '\1' ", text, flags=re.U)

    # aperturas y cierres de puntuación
    for i, regex in enumerate([u'(\w)([(\u00BF\u00A1])', u'([?!)])(\w)']):
        for match in re.findall(regex, text, flags=re.U):
            if not re.match(r'[0-9_]', match[0] if i % 2 == 0 else match[1], re.U):
                text = text.replace(''.join(match),
                                    ''.join([match[0], ' ', match[1]]),
                                    1)

    # puntuación simple
    text = re.sub(r'[.]{2,}', u'\u2026', text, flags=re.U)

    for match in re.findall(u'(\A|.)([.,;:\u2026])(\w)', text, flags=re.U):
        if (len(match[0]) > 0 and re.match(r'[0-9]', match[0], flags=re.U)
                and match[1] in ('.', ',')
                and re.match(r'[0-9]', match[2], flags=re.U)):
            continue
        text = text.replace(''.join(match),
                            ''.join([match[0], match[1], ' ', match[2]]),
                            1)

    text = text.replace(u'\u2026', '...')

    return re.sub(r'[ ]+', ' ', text, flags=re.U).strip()

def _switch_sentiment_services(mode='on'):
    """Inicia/termina los servicios requeridos del sistema."""
    _switch_flookup_server(set_of_transducers=LEXICONS,
                           transducer='all',
                           mode=mode)
    _switch_freeling_server(mode=mode)


class SentimentAnalysis(object):
    def __init__(self, negation_id, lexicon, analyzer=None, word_ngram_range=None,
                 char_ngram_range=None, lowercase=None, binary=None, clf=None):
        """Instancia un sistema de análisis de sentimientos según parámetros."""
        self.negation_id = negation_id
        self.is_negation_enabled = NEGATION_SETTINGS[negation_id]['negation_enabled']
        self.negation_type = NEGATION_SETTINGS[negation_id]['negation_type']
        self.scope_negation = NEGATION_SETTINGS[negation_id]['scope']
        self.negation_PoS_tags = NEGATION_SETTINGS[negation_id]['tags']

        self.lexicon = lexicon

        self.analyzer = analyzer
        self.word_vectorizer = None
        self.char_vectorizer = None

        self.clf = None

        if self.analyzer is not None:
            self.instantiate_models(word_ngram_range, char_ngram_range, lowercase,
                                    binary, clf)

    def instantiate_models(self, word_ngram_range, char_ngram_range, lowercase,
                           binary, clf):
        """Instancia los modelos de vectorización y clasificación."""
        models_path = CURRENT_PATH + '/models'

        self.clf = joblib.load(models_path + '/classifiers/%s.pkl' % clf)

        analyzers = [self.analyzer,]
        if self.analyzer == 'both':
            analyzers = ['word', 'char']

        for analyzer in analyzers:
            ngram_range = word_ngram_range\
                          if analyzer == 'word' else char_ngram_range

            vectorizer_fname = '%s-%s-%i_%i-%s-%s.pkl' %\
                               (self.negation_id, analyzer,
                                ngram_range[0], ngram_range[1],
                                'True' if lowercase else 'False',
                                'True' if binary else 'False')
            vectorizer_fname = models_path + '/vectorizers/' + vectorizer_fname

            if analyzer == 'word':
                self.word_vectorizer = joblib.load(vectorizer_fname)
            else:
                self.char_vectorizer = joblib.load(vectorizer_fname)

    def preprocess_tweet(self, text):
        """Método responsable del preprocesamiento del tweet."""
        text = _to_unicode(text)

        # estas características corresponden a las 24
        # primeras descritas en el método get_sentiment
        features = np.zeros(24, dtype=int)

        # remover URL's y email's
        text = re.sub(r'(?:https?://\S+)|(?:www\.\S+)', ' ', text, flags=re.U)
        for match in re.findall(r'[\w\.-]+@[\w\.-]+', text, flags=re.U):
            if re.search(r'\.\w+$', match, flags=re.U):
                text = text.replace(match, ' ', 1)
        text = re.sub(r'[ ]+', ' ', text, flags=re.U).strip()

        # homologar algunos signos a su representación en castellano
        for regex, replacement in PUNCTUATION_REPLACEMENTS:
            text = re.sub(regex, replacement, text, flags=re.U)

        # homologar algunas entidades HTML
        for html_entity, replacement in HTML_ENTITIES_REPLACEMENTS:
            text = text.replace(html_entity, replacement)

        # remover strings: ht, htt, htttp, https
        text = re.sub(r'(\A|\W)ht(?:t|tp|tps)?(\W|\Z)', r'\1 \2', text, flags=re.U)
        text = re.sub(r'[ ]+', ' ', text, flags=re.U).strip()

        # placeholders de lenguaje de Twitter
        text = re.sub(r'(\A|\W)@\w+', r'\1@usuario', text, flags=re.U)
        text = re.sub(r'(\A|\W)#\w+', r'\1#etiqueta', text, flags=re.U)

        # realizar tokenización para calcular las dos primeras características
        words = []
        for match in re.findall(r'(?:\A|\W)?(\w+)(?:\W|\Z)?', text, flags=re.U):
            if re.match(r'[0-9_]+', match, re.U):
                continue
            words.append(match)

        # si el texto del tweet no tiene ningún caracter,
        # retornar valores por defecto
        if len(text.strip()) == 0:
            return u'', features

        # normalización léxica
        oov_words = _perform_spell_checking(text=text)
        for oov in oov_words:
            oov[2] = _to_unicode(oov[2])
            if oov[2] not in words:
                words.append(oov[2])
            text = re.sub(r'(\A|\W)' + oov[2],
                          r'\1' + oov[3].replace('_', ' '),
                          text,
                          count=1, flags=re.U)

        # calcular las dos primeras características
        for word in words:
            if (word == word.upper() and len(word) > 1):
                features[0] += 1
            if re.search(r'(.)\1{2,}', word, re.U):
                features[1] += 1

        # características relacionadas con signos de puntuación
        features[2] = len(re.findall(r'[?!]{2,}', text, flags=re.U))
        if re.search(r'[?!]$', text, flags=re.U):
            features[3] = 1

        # normalizar repetición de caracteres
        text = re.sub(r'[.]{2,}', u'\u2026', text, flags=re.U)
        text = re.sub(r'(\W)\1+', r'\1', text, flags=re.U)
        text = text.replace(u'\u2026', '...')

        # capturar emoticones
        text = _capture_emoticons(text)

        text = _normalize_punctuation_marks(text)

        # características relacionadas con emoticones
        for match in re.findall(r'(EMO_(?:POS|NEG))', text, flags=re.U):
            i = 4 if match.split('_')[1] == 'POS' else 5
            features[i] += 1
        if re.search(r'EMO_(?:POS|NEG)$', text, re.U):
            features[6] = 1

        # realizar análisis morfológico y etiquetado P-o-S
        analysis, features[7] = _detect_negation(
            analysis=_analyze_morphologically(text),
            type_=self.negation_type,
            scope=self.scope_negation,
            tags=self.negation_PoS_tags)

        text_ = []
        for sentence in analysis:
            for token, lemma, tag, token_has_to_be_negated in sentence:
                if len(token) == 1 and token.lower() in ONE_LETTER_WORDS:
                    token = token.lower()
                    lemma = token
                    tag = u'CC'

                token = _to_unicode(token)
                lemma = _to_unicode(lemma)
                tag = _to_unicode(tag).lower()

                # características basadas en PoS tags
                if tag[0] == 'a':
                    features[8] += 1
                elif tag[0] == 'r':
                    features[9] += 1
                elif tag[0] == 'd':
                    features[10] += 1
                elif tag[:2] == 'nc':
                    features[11] += 1
                elif tag[:2] == 'np':
                    features[12] += 1
                elif tag[0] == 'v':
                    features[13] += 1
                elif tag[0] == 'p':
                    features[14] += 1
                elif tag[0] == 'c':
                    features[15] += 1
                elif tag[0] == 'i':
                    features[16] += 1
                elif tag[0] == 's':
                    features[17] += 1
                elif tag[0] == 'f':
                    features[18] += 1
                    continue
                elif tag[0] == 'z':
                    features[19] += 1
                elif tag[0] == 'w':
                    features[20] += 1

                if tag[0] in [u'w', u'z']:
                    token_ = ''.join([t
                                      for t in token.lower()
                                      if (re.match(r'\w', t, re.U)
                                          and not re.match(r'[0-9]', t, re.U))])

                    if len(token_) > 1:
                        token = token_
                        lemma = token
                    else:
                        continue

                # características basadas en lexicones
                successful_polarity_search = False
                for i, word in enumerate([token, lemma]):
                    tokens = word.lower().split('_')
                    if i == 1 and len(tokens) > 1:
                        continue
                    elif successful_polarity_search:
                        break
                    for token_ in tokens:
                        if (not re.match(r'\w+', token_, re.U)
                                or re.search(r'[0-9]', token_, re.U)):
                            continue
                        candidates = [token_,
                                      _to_unicode(unidecode(token_)),
                                      _deaccent(token_)]
                        polarity = None
                        for candidate in candidates:
                            search = _foma_string_lookup(candidate,
                                                         self.lexicon,
                                                         LEXICONS)
                            if len(search) == 1:
                                successful_polarity_search = True
                                if search[0].lower()[0] == 'p':
                                    polarity = 'POS'
                                else:
                                    polarity = 'NEG'
                                break
                        if (polarity is not None and self.is_negation_enabled
                                and token_has_to_be_negated):
                            if polarity == 'POS':
                                polarity = 'NEG'
                            else:
                                polarity = 'POS'
                        if polarity == 'POS':
                            features[21] += 1
                        elif polarity == 'NEG':
                            features[22] += 1

                tokens = []
                if tag[:2] == 'np':
                    tokens += token.split('_')
                else:
                    if re.match(r'EMO_(?:POS|NEG)', lemma, re.U):
                        tokens += [lemma]
                    else:
                        tokens += lemma.lower().split('_')

                for token_ in tokens:
                    if (token_.startswith(u'@') or token_.startswith(u'#')
                            or (re.match(r'[\w-]+$', token_, re.U)
                                    and not re.search(r'[0-9]', token_, re.U))):
                        if self.is_negation_enabled and token_has_to_be_negated:
                            token_ = u'NEG_' + token_
                        text_.append(token_)

        # característica de polaridad global
        features[23] = features[21] - features[22]
        if features[23] > 0:
            features[23] = 1
        elif features[23] < 0:
            features[23] = -1

        text = u' '.join(text_)
        for isolated_words, contraction in CONTRACTIONS:
            text = text.replace(isolated_words, contraction)
        text = re.sub(r'[ ]+', ' ', text, flags=re.U).strip()

        return text, features

    def predict_sentiment(self, text):
        # características que complementan las ex-
        # traídas en el proceso de vectorización
        #   0, # palabras completamente en MAYÚSCULA
        #   1, # palabras con un caracter repetido más de dos veces
        #   2, # secuencias de signos de exclamación, interrogación
        #      o una combinación de ambos (ej: !!, ??, ?!)
        #   3, ¿el último token contiene un signo de exclamación o interrogación?
        #   4, # emoticones clasificados como positivo
        #   5, # emoticones clasificados como negativo
        #   6, ¿el último token es un emoticón?
        #   7, # contextos negados
        #   8, # adjetivos
        #   9, # adverbios
        #   10, # determinantes
        #   11, # nombres comunes
        #   12, # nombres propios
        #   13, # verbos
        #   14, # pronombres
        #   15, # conjunciones
        #   16, # interjecciones
        #   17, # preposiciones
        #   18, # puntuaciones
        #   19, # numerales
        #   20, # fechas y horas
        #   21, # palabras positivas
        #   22, # palabras negativas
        #   23, polaridad: diferencia entre palabras positivas y negativas
        text, metafeatures = self.preprocess_tweet(text)

        metafeatures = sp.csr_matrix(metafeatures.reshape(1, metafeatures.shape[0]))

        ngram_features = None

        analyzers = ['word', 'char'] if self.analyzer == 'both' else [self.analyzer,]
        for analyzer in analyzers:
            vectorizer = self.word_vectorizer if analyzer == 'word' else self.char_vectorizer

            text_ = text if analyzer == 'word' else text.replace(u'_', u' ').strip()
            features_ = vectorizer.transform([text_,])

            if ngram_features is None:
                ngram_features = features_
            else:
                ngram_features = sp.hstack([ngram_features, features_], format='csr')

        features = sp.hstack([metafeatures, ngram_features], format='csr')

        return self.clf.predict_proba(features)
