#!/usr/bin/python

"""
Energy layer to get the loss.
"""

import torch
from torch.autograd import Function
from torch import Tensor
from rdkit.Chem.rdchem import Mol
import rdkit.Chem.AllChem as Chem2


class Energy(Function):
    """Energy loss with forward and backward pass."""

    @staticmethod
    def forward(ctx, input: Tensor, mol: Mol) -> Tensor:
        """Forward pass of energy. Only mol matters here."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Chem2.MMFFSanitizeMolecule(mol)
        mmff_props = Chem2.MMFFGetMoleculeProperties(mol)
        energy = []
        ff_list = []
        for i in range(mol.GetNumConformers()):
            ff = Chem2.MMFFGetMoleculeForceField(
                mol, mmff_props, confId=i)
            ff_list.append(ff)
            energy.append(ff.CalcEnergy())
        ctx.ff_list = ff_list
        energy = torch.tensor(energy, requires_grad=True, device=device)
        return energy

    @staticmethod
    def backward(ctx, grad_output) -> Tensor:
        """Backward pass of energy. Only ff_list from forward matters here."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        grad_list = []
        for ff in ctx.ff_list:
            grad_list.append(torch.tensor(ff.CalcGrad(), device=device).reshape(-1, 3))
        grad_energy = torch.stack(grad_list)
        grad_input = grad_output[:, None, None] * grad_energy
        return grad_input, None
