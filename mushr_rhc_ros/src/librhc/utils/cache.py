import os


def get_root_cache_dir(params):
    path = params.get_str("cache", default="~/.cache/mushr_rhc/")
    path = os.path.expanduser(path)

    if not os.path.isdir(path):
        os.mkdir(path)

    return path


def get_cache_dir(params, path):
    root = get_root_cache_dir(params)
    fullpath = os.path.join(root, path)

    if not os.path.isdir(fullpath):
        os.makedirs(fullpath)

    return fullpath


def get_cache_map_dir(params, map):
    return get_cache_dir(params, map.name)
