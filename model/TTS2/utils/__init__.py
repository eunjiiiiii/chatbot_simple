# -*- coding: utf-8 -*-
import re, json, sys, os
import tensorflow as tf
from tqdm import tqdm
from contextlib import closing
from multiprocessing import Pool
from collections import namedtuple
from datetime import datetime, timedelta
from shutil import copyfile as copy_file
from tensorflow.python import pywrap_tensorflow

PARAMS_NAME = "params.json"
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
LOGDIR_ROOT_Wavenet = './logdir-wavenet'


class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []





def load_hparams(hparams, load_path, skip_list=[]):
    # log dir에 있는 hypermarameter 정보를 이용해서, hparams.py의 정보를 update한다.
    path = os.path.join(load_path, PARAMS_NAME)

    new_hparams = load_json(path)
    hparams_keys = vars(hparams).keys()

    for key, value in new_hparams.items():
        if key in skip_list or key not in hparams_keys:
            print("Skip {} because it not exists".format(key))  # json에 있지만, hparams에 없다는 의미
            continue

        if key not in ['xxxxx', ]:  # update 하지 말아야 할 것을 지정할 수 있다.
            original_value = getattr(hparams, key)
            if original_value != value:
                print("UPDATE {}: {} -> {}".format(key, getattr(hparams, key), value))
                setattr(hparams, key, value)


def load_json(path, as_class=False, encoding='euc-kr'):
    with open(path, encoding=encoding) as f:
        content = f.read()
        content = re.sub(",\s*}", "}", content)
        content = re.sub(",\s*]", "]", content)

        if as_class:
            data = json.loads(content, object_hook= \
                lambda data: namedtuple('Data', data.keys())(*data.values()))
        else:
            data = json.loads(content)

    return data

def add_prefix(path, prefix):
    dir_path, filename = os.path.dirname(path), os.path.basename(path)
    return "{}/{}.{}".format(dir_path, prefix, filename)


def add_postfix(path, postfix):
    path_without_ext, ext = path.rsplit('.', 1)
    return "{}.{}.{}".format(path_without_ext, postfix, ext)


def remove_postfix(path):
    items = path.rsplit('.', 2)
    return items[0] + "." + items[2]


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def parallel_run(fn, items, desc="", parallel=True):
    results = []

    if parallel:
        with closing(Pool(10)) as pool:
            for out in tqdm(pool.imap_unordered(fn, items), total=len(items), desc=desc):
                if out is not None:
                    results.append(out)
    else:
        for item in tqdm(items, total=len(items), desc=desc):
            out = fn(item)
            if out is not None:
                results.append(out)

    return results


def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)


def str2bool(v):
    return v.lower() in ('true', '1')

