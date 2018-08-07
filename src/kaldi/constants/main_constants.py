from os.path import join as join_path


KALDI_PATH_FILE = './kaldi/scripts/path.sh'

DATA_DIR = 'data'
MFCC_DIR = 'mfcc'
VAD_DIR = 'vad'

DATA_SCP_FILE = join_path(DATA_DIR, 'data.scp')
FEATS_SCP_FILE = join_path(MFCC_DIR, 'feats.scp')
VAD_SCP_FILE = join_path(VAD_DIR, 'vad.scp')
