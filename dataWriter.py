class dataWriter:
    def __init__(self):
        pass
    def writeProgress(self, path, data):
        fh = open(path, 'w')
        fh.write(str(data))
        fh.close()

def writeProgress(path, data):
    fh = open(path, 'w')
    fh.write(str(data))
    fh.close()