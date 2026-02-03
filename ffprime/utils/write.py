#!/usr/bin/env python


import os
import json
import subprocess
import numpy as np


def get_popen(command):
    p = subprocess.Popen(
        command,
        universal_newlines=True,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return p.stdout.read().strip()


def write_dict_to_json(data, fn_json):

    if not fn_json.endswith(".json"):
        fn_json += ".json"

    # load JSON file (if existing) & compare its content
    if os.path.isfile(fn_json):
        with open(fn_json, "r") as fn:
            data_existing = json.load(fn)
        for key, value in list(data.items()):
            # if key exists, the corresponding value gets updated, if not created
            value_existing = data_existing.setdefault(key, {})
            # add another key:value loop if key is numerical (like molecule number)
            if key.isnumeric():
                for k2, v2 in iter(value.items()):
                    if k2 in ["atnums", "atcoords"]:
                        continue
                    value_existing.setdefault(k2, {}).update(v2)
            else:
                value_existing.update(value)
        data = data_existing

    # write JSON file
    with open(fn_json, "w") as fn:
        json.dump(data, fn, indent=4)

def write_npz(mol, filename, allow_object=False):
    data = {}

    for attr in dir(mol):

        # skip private, methods, dunders
        if attr.startswith("_"):
            continue

        try:
            val = getattr(mol, attr)
        except Exception:
            continue

        # skip callables (methods, functions)
        if callable(val):
            continue

        try:
            # NumPy array
            if isinstance(val, np.ndarray):
                data[attr] = val

            # list / tuple
            elif isinstance(val, (list, tuple)):
                data[attr] = np.asarray(val)

            # scalar
            elif isinstance(val, (int, float, bool, str)):
                data[attr] = np.asarray(val)

            # dict → flatten
            elif isinstance(val, dict):
                for k, v in val.items():
                    key = f"{attr}.{k}"
                    data[key] = np.asarray(v)

            # optional object fallback
            elif allow_object:
                data[attr] = np.asarray(val, dtype=object)

            else:
                continue

        except Exception:
            # skip anything that cannot be serialized
            continue

    np.savez_compressed(filename, **data)
