from rdkit import DataStructs, Chem
from rdkit.Chem import AllChem
from config import VECTOR_DIMENSION


def smiles_to_vector(smiles):
    # Convert from smile to vector
    mols = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mols, 2, VECTOR_DIMENSION)
    hex_fp = DataStructs.BitVectToFPSText(fp)
    vec = bytes.fromhex(hex_fp)
    return vec
