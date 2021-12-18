import sys


class Logger(object):
    def __init__(self, filename="logfile.txt"):
        super().__init__()
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """"""

    def close(self):
        self.terminal.close()
        self.log.close()
