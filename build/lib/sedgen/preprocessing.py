import pickle


def save_obj(obj, name):
    """Save Python instances (e.g. dict) as a binary file to disk
    """
    with open("../_DATA/obj/" + name + ".pkl", 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    """Save Python instances in binary file format from disk"""
    with open("../_DATA/obj/" + name + ".pkl", 'rb') as f:
        return pickle.load(f)
