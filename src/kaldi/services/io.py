from os.path import abspath

import re
import numpy as np

from kaldi.services.run import Kaldi


def read_feat(scp_file, n_features):
    output, _ = Kaldi().run_command('copy-feats scp:{} ark,t:'.format(scp_file))
    output = re.split('[\[]+', output)[1][1:-3]
    return np.fromstring(output, dtype=float, sep=' \n').reshape([-1, n_features]).T


def read_vector(scp_file, dtype=np.float):
    output, _ = Kaldi().run_command('copy-vector scp:{} ark,t:'.format(scp_file))
    output = re.split('[\[]+', output)[1][1:-3]
    return np.fromstring(output, dtype, sep=' ')


def write_vector(arr, utt_id, ark_file):
    scp_file = '{}.scp'.format(ark_file[:-4])
    arr = np.array(arr)
    Kaldi().run_command('echo [ {} ] | copy-vector - - > {}'.format(np.array_str(arr)[1:-1], ark_file))

    with open(scp_file, 'w') as f:
        f.write('{} {}\n'.format(utt_id, abspath(ark_file)))
