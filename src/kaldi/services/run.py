from subprocess import Popen, PIPE

from kaldi.constants.main_constants import KALDI_PATH_FILE


class Kaldi:
    def __init__(self, path_file=KALDI_PATH_FILE):
        self.command = 'source {}'.format(path_file)

    def run_command(self, cmd):
        cmd = '{} && ({})'.format(self.command, cmd)
        output, error = Popen(cmd, stdout=PIPE, shell=True).communicate()
        return output.decode("utf-8"), error.decode("utf-8") if error is not None else error
