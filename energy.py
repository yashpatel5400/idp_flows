import os
import torch
import numpy as np
import rdkit.Chem.AllChem as Chem
from rdkit.Geometry import Point3D

def get_conformer_energy(mol: Chem.Mol, conf_id: int = None) -> float:
    """Returns the energy of the conformer with `conf_id` in `mol`.
    """
    if conf_id is None:
        conf_id = mol.GetNumConformers() - 1
    Chem.MMFFSanitizeMolecule(mol)
    mmff_props = Chem.MMFFGetMoleculeProperties(mol)
    ff = Chem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
    energy = ff.CalcEnergy()
    return energy