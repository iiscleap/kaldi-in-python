from os.path import abspath

import re
import numpy as np

from kaldi.services.run import Kaldi


def append_vector(arr, utt_id, ark_file, offset=None):
    arr = np.array(arr)
    output = Kaldi().run_command('echo [ {} ] | copy-vector - -'.format(np.array_str(arr)[1:-1]), decode=False)

    if offset is None:
        with open(ark_file, 'rb') as f:
            offset = len(f.read())

    content = bytes('{} '.format(utt_id).encode('utf-8'))
    offset = offset + len(content)
    content = content + output

    with open(ark_file, 'ab') as f:
        f.write(content)

    return '{} {}:{}\n'.format(utt_id, abspath(ark_file), offset), offset


def read_feat(scp_file, n_features):
    output = Kaldi().run_command('copy-feats scp:{} ark,t:'.format(scp_file))
    output = re.split('\[', output)[1][1:-2]
    return np.fromstring(output, dtype=float, sep=' \n').reshape([-1, n_features]).T


def read_vectors(scp_file, dtype=np.float):
    vectors = Kaldi().run_command('copy-vector scp:{} ark,t:'.format(scp_file))
    vectors = re.split('\n', vectors)[:-1]
    utt_list = []
    vector_list = []
    for vector in vectors:
        vector = re.split('\[', vector)
        utt_id = vector[0][:-1]
        vector = vector[1][1:-2]
        utt_list.append(utt_id)
        vector_list.append(np.fromstring(vector, dtype, sep=' '))
    return (utt_list, vector_list) if len(vector_list) > 1 else (utt_list[0], vector_list[0])


def write_vector(arr, utt_id, ark_file):
    arr = np.array(arr)
    output = Kaldi().run_command('echo [ {} ] | copy-vector - -'.format(np.array_str(arr)[1:-1]), decode=False)
    content = bytes('{} '.format(utt_id).encode('utf-8'))
    offset = len(content)
    content = content + output
    with open(ark_file, 'wb') as f:
            f.write(content)
    return '{} {}:{}\n'.format(utt_id, abspath(ark_file), offset), offset
