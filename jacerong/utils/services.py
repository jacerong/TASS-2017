# -*- coding: iso-8859-15 -*-

import json, multiprocessing, os, re, socket, subprocess,\
       xml.etree.ElementTree as ET

import numpy as np, psutil, requests

from . import convert_into_str as _to_str
from . import convert_into_unicode as _to_unicode
from . import write_in_file as _write_in_file


CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = '/'.join(CURRENT_PATH.split('/')[:-1])
CONFIG_PATH = BASE_PATH + '/config'

###############################
# read the configuration file #
###############################

config = ET.parse(CONFIG_PATH + '/general.xml').getroot()

FREELING_PORT = config[0][0].text

FOMA_PATH = config[1][0].text.rstrip('/')

SYSTEM_USER = config[2][0].text

SPELL_CHECKER_API_URL = config[3][0].text

config = None


def switch_freeling_server(mode='on',
                           initialization_command='default',
                           port=FREELING_PORT):
    '''Inicia/termina el servico de análisis de FreeLing.

    paráms:
        initialization_command: str | list
            especificar la configuración de FreeLing. Por defecto, es la requerida
            para realizar análisis de sentimientos de tweets en español.
        port: str
            cuál puerto se utilizará para ejecutar el servicio de FreeLing.

    NOTA: el proceso se inicia y se termina usando el usuario SYSTEM_USER.
    '''
    pid = None
    for process in psutil.process_iter():
        cmd_line = process.cmdline()
        if (process.username() == SYSTEM_USER and len(cmd_line) > 1
                and re.search('analyzer$', cmd_line[0], re.I)
                and (cmd_line[-1] == port
                     or cmd_line[-2] == port)):
            pid = process.pid
            break
    if pid is not None and mode == 'off':
        psutil.Process(pid=pid).kill()
    elif pid is None and mode == 'on':
        if (isinstance(initialization_command, str)
                and initialization_command == 'default'):
            subprocess.Popen(['analyze', '-f', 'es.cfg' , '--flush',
                '--ftok', CONFIG_PATH + '/es-twit-tok.dat',
                '--usr', '--fmap', CONFIG_PATH + '/es-twit-map.dat',
                '--outlv', 'tagged', '--noloc',
                '--fdict', CONFIG_PATH + '/dicc.src',
                '--server', '--port', FREELING_PORT, '&'])
        elif (isinstance(initialization_command, list)
                and len(initialization_command) > 0):
            subprocess.Popen(initialization_command)
        else:
            raise Exception('No ha especificado un comando de inicialización válido')

def analyze_morphologically(text, port=FREELING_PORT):
    '''Analiza morfologicamente el texto de un tweet utilizando FreeLing.'''
    text = _to_str(text)

    fname = CURRENT_PATH + '/.tmp/FreeLing-%03d%s%05d' %(
        np.random.randint(0, 100),
        '-' if np.random.randint(0,2) == 1 else '',
        np.random.randint(0, 100000))

    _write_in_file(fname + '.txt', text)

    subprocess.call(["analyzer_client", port],
        stdin=open(fname  + '.txt'),
        stdout=open(fname + '.tagged', 'w'))

    sentences = []
    sentence = []
    with open(fname + '.tagged') as foutput:
        for line in foutput:
            line = line.rstrip('\n')
            if len(line) == 0:
                sentences.append(sentence)
                sentence = []
                continue
            try:
                form, lemma, tag = re.split('\s+', line)[:3]
                sentence.append([
                    form.decode('utf-8'), lemma.decode('utf-8'),
                    tag.decode('utf-8')])
            except:
                form = line
                sentence.append([form.decode('utf-8'), '', ''])

    os.remove(fname + '.txt')
    os.remove(fname + '.tagged')

    return sentences

def check_flookup_server_status(transducer):
    """Evalúa si el transductor está ejecutándose como servicio.

    paráms:
        transducer: str
            Nombre del transductor. Puede ser la ruta completa
            o parte de esta.

    Retorna el pid del proceso de flookup que ejecuta como servidor
    el transductor.

    NOTA: los procesos deben haber sido ejecutados por el usuario SYSTEM_USER.
    """
    pid = None
    transducer = _to_str(transducer)
    for process in psutil.process_iter():
        cmd_line = process.cmdline()
        if (process.username() == SYSTEM_USER and len(cmd_line) > 1
                and re.search('flookup$', cmd_line[0], re.I)
                and re.search(transducer + '.bin', _to_str(cmd_line[-2]), re.I)):
            pid = process.pid
            break
    return pid

def switch_flookup_server(set_of_transducers, transducer='all', mode='on'):
    """Iniciar o terminar un servicio de transductor como servidor.

    paráms:
        set_of_transducers: dict
            conjunto de transductores
        transducer: str
            nombre del transductor definido como clave en el diccionario
            set_of_transducers.
            Por defecto se asumen todos los transductores.
        mode: str
            toma dos posibles valores: ON, para iniciar el servidor;
            OFF, para terminar el servidor.

    NOTA: los procesos deben ser ejecutados por el usuario SYSTEM_USER.
    """
    transducer = _to_str(transducer).lower()
    if transducer != 'all' and transducer not in set_of_transducers.keys():
        raise Exception('Transductor %s no reconocido' % transducer)
    elif mode not in ['on', 'off']:
        raise Exception('La acción definida no es válida')

    if transducer == 'all':
        pool = multiprocessing.Pool(processes=3)
        for t in set_of_transducers.keys():
            pool.apply_async(
                switch_flookup_server,
                [set_of_transducers, t, mode])

        pool.close()
        pool.join()

        return

    pid = check_flookup_server_status(transducer)
    transducer = set_of_transducers[transducer]

    if mode == 'on':
        if pid is None:
            subprocess.Popen([FOMA_PATH + '/flookup', '-S',
                '-A', transducer[1], '-P', transducer[2],
                '-i', '-x', transducer[0], '&'])
    else:
        if pid is not None:
            process = psutil.Process(pid=pid)
            process.kill()

def foma_string_lookup(token, transducer, set_of_transducers):
    '''Analiza el token a través del transductor especificado.

    paráms:
        token: str
            cadena de caracteres a ser analizada.
        transducer: str
            transductor que analizará el token. Puede ser una ruta completa
            o alguna de las claves especificadas en set_of_transducers.
        set_of_transducers: dict
            conjunto de transductores

    NOTA: si el transductor no es una ruta física del sistema, sino una de las
    claves del diccionario set_of_transducers, se analizará como servicio de
    flookup. Para esto, debe haberse iniciado con anterioridad el servicio de
    flookup.
    '''
    use_server = False
    if transducer.lower() in set_of_transducers.keys():
        use_server = True
    elif not os.path.isfile(transducer):
        raise Exception('El transductor especificado no existe')

    token = _to_str(token)

    result = []
    if not use_server:
        fname_input = '%s-%03d%s%05d.txt' % (
            CURRENT_PATH + '/.tmp/flookup',
            np.random.randint(0, 100),
            '-' if np.random.randint(0,2) == 1 else '_',
            np.random.randint(0, 100000))
        _write_in_file(fname_input, token, mode='w')

        fname_output = fname_input.replace('.txt', '.out')
        subprocess.call([FOMA_PATH + '/flookup', '-i', '-x', transducer],
            stdin=open(fname_input),
            stdout=open(fname_output, 'w'))

        with open(fname_output) as finput:
            for line in finput:
                line = line.rstrip('\n')
                if len(line.strip()) > 0 and line != '?+':
                    result.append(_to_unicode(line))

        os.remove(fname_input)
        os.remove(fname_output)
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        transducer = set_of_transducers[transducer.lower()]
        sock.sendto(token, (transducer[1], int(transducer[2])))

        data, addr = sock.recvfrom(4096)

        result = [_to_unicode(d)
                  for d in data.split('\n')
                  if len(d.strip()) > 0 and d != '?+']

        sock.close()

    return result

def perform_spell_checking(text, api_url=SPELL_CHECKER_API_URL):
    """Realizar corrección ortográfica a un texto.

    Retorna lista de palabras fuera-de-vocabulario ("Out-of-Vocabulary words").

    Este método consume la API del programa de normalización léxica "normalesp".

    "normalesp": <https://github.com/jacerong/normalesp>
    """
    head = {"Content-type": "application/json"}
    data = json.dumps({'text': _to_str(text)})
    r = requests.post(api_url, data=data, headers=head)
    if r.status_code == requests.codes.ok:
        return json.loads(r.content)
    else:
        warnings.warn("Error consumiendo la API de corrección ortográfica")
        return []
