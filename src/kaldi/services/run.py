from subprocess import Popen, PIPE

from kaldi.constants.main_constants import KALDI_PATH_FILE


class Kaldi:
    def __init__(self, path_file=KALDI_PATH_FILE):
        self.command = 'source {}'.format(path_file)

    def run_command(self, cmd, decode=True):
        cmd = '{} && ({})'.format(self.command, cmd)
        output, _ = Popen(cmd, stdout=PIPE, shell=True).communicate()
        if decode:
            return output.decode("utf-8")
        return output
