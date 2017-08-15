# -*- coding: iso-8859-15 -*-

import os


__all__ = ["convert_into_str", "convert_into_unicode", "remove_accent_marks",
           "write_in_file", "services"]


def convert_into_str(token):
    return token.encode('utf-8') if not isinstance(token, str) else token

def convert_into_unicode(token):
    return token.decode('utf-8') if not isinstance(token, unicode) else token

def remove_accent_marks(word):
    word = convert_into_unicode(word)

    remove_accents = {u'\xe1': u'a',
                      u'\xe9': u'e',
                      u'\xed': u'i',
                      u'\xf3': u'o',
                      u'\xfa': u'u',
                      u'\xfc': u'u'}

    return convert_into_unicode(
        ''.join([remove_accents[s] if s in remove_accents.keys() else s
                 for s in word])
        )

def write_in_file(fname, content, mode='w', makedirs_recursive=True):
    dir_ = '/'.join(fname.split('/')[:-1])
    if not os.path.isdir(dir_) and makedirs_recursive:
        os.makedirs(dir_)
    with open(fname, mode) as f:
        f.write(content)
