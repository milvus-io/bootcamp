import sys
from diskcache import Cache
from logs import LOGGER
from encode import smiles_to_vector
from config import DEFAULT_TABLE
from config import UPLOAD_PATH
from rdkit import Chem
from rdkit.Chem import Draw


def save_mols_img(milvus_ids, smiles):
    # Save the molecular images
    for ids, mol in zip(milvus_ids, smiles):
        mol = Chem.MolFromSmiles(mol)
        sub_img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(500, 500))
        sub_img.save(UPLOAD_PATH + "/" + str(ids) + ".png")

def extract_features(filepath):
    # Extract all vectors from mol file
    try:
        cache = Cache('./tmp')
        feats = []
        names = []
        total = len(open(filepath, encoding="utf8", mode='rU').readlines())
        cache['total'] = total
        current = 0
        vec = 0
        with open(filepath, encoding="utf8", mode='r') as f:
            for line in f:
                current += 1
                cache['current'] = current
                line = line.strip()
                line = line.split()
                line = line[0]
                try:
                    vec = smiles_to_vector(line)
                    feats.append(vec)
                    names.append(line.encode())
                    LOGGER.info(f"extracted feature from smi No.{cache['current']} , {cache['total']} molecular in total")
                except Exception as e:
                    LOGGER.error(f"Failed to extract fingerprint {e}")
        return feats, names
    except Exception as e:
        LOGGER.error(f"Error with extracting feature from image {e}")
        sys.exit(1)

def format_data(ids, names):
    # Combine the id of the vector and the molecule structure into a list
    data = []
    for i in range(len(ids)):
        value = (str(ids[i]), names[i])
        data.append(value)
    return data

def do_load(table_name, mol_path, milvus_client, mysql_cli):
    # Import vectors to Milvus and data to Mysql respectively
    if not table_name:
        table_name = DEFAULT_TABLE
    vectors, names = extract_features(mol_path)
    ids = milvus_client.insert(table_name, vectors)
    save_mols_img(ids, names)
    mysql_cli.create_mysql_table(table_name)
    mysql_cli.load_data_to_mysql(table_name, format_data(ids, names))
    return len(ids)
