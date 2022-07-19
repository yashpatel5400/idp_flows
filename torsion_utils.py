from rdkit.Chem import AllChem as Chem
from rdkit.Chem import TorsionFingerprints

def get_torsion_tuples(mol):
    """Gets the tuples for the torsion angles of the molecule.

    Parameters
    ----------
    mol : RDKit molecule
        Molecule for which torsion angles are to be extracted

    * tuples_original, tuples_reindexed : list[int]
        Tuples (quadruples) of indices that correspond to torsion angles. The first returns indices
        for the original molecule and the second for a version of the molecule with Hydrogens removed
        (since there are many cases where this stripped molecule is of interest)
    """

    [mol.GetAtomWithIdx(i).SetProp("original_index", str(i)) for i in range(mol.GetNumAtoms())]
    stripped_mol = Chem.rdmolops.RemoveHs(mol)

    nonring, _ = TorsionFingerprints.CalculateTorsionLists(mol)
    nonring_original = [list(atoms[0]) for atoms, ang in nonring]
            
    original_to_stripped = {
        int(stripped_mol.GetAtomWithIdx(reindex).GetProp("original_index")) : reindex 
        for reindex in range(stripped_mol.GetNumAtoms())
    }
    nonring_reindexed = [
        [original_to_stripped[original] for original in atom_group] 
        for atom_group in nonring_original
    ]

    return nonring_original, nonring_reindexed