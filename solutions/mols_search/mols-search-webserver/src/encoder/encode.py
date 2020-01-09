import os
import numpy as np
from diskcache import Cache
from rdkit import DataStructs
from common.const import default_cache_dir
from rdkit import Chem
import math
from rdkit.Chem import AllChem
from common.config import VECTOR_DIMENSION


def smiles_to_vec(smiles):
    mols = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mols, 2, VECTOR_DIMENSION)
    hex_fp = DataStructs.BitVectToFPSText(fp)
    # print(hex_fp)
    vec = bytes.fromhex(hex_fp)
    return vec


def feature_extract(table_name, filepath):
    names = []
    feats = []
    cache = Cache(default_cache_dir)
    total = len(open(filepath,'rU').readlines())
    cache['total'] = total
    current = 0
    with open(filepath, 'r') as f:
        for line in f:
            current += 1
            cache['current'] = current
            line = line.strip()
            line = line.split()
            line = line[0]
            try:
                vec = smiles_to_vec(line)
                feats.append(vec)
                names.append(line.encode())
            except:
                continue
            print ("extracting feature from smi No. %d , %d molecular in total" %(current, total))
    return feats, names