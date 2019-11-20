import errno
import os
import pickle

def saveData(path, filename, dict_):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(os.path.join(path, filename + ".pickle"), 'wb') as f:
        pickle.dump(dict_, f)

def loadData(path, filename):
    with open(os.path.join(path, filename + ".pickle"), 'rb') as f:
        dict_= pickle.load(f)
    return dict_

