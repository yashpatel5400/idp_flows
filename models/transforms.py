#!/usr/bin/python

"""
Coordinate transformation from dihedral angles to 3D coordinates.
Code adapted from https://github.com/rdkit/rdkit
"""

from rdkit.Chem.rdchem import Mol
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import normalize
from rdkit import Chem
from logging import Logger


class Dihedral2Coord(Module):
    """Transform dihedral angles of a batch of conformers into 3D coordinates."""

    def __init__(self, mol: Mol, angles: Tensor, logger: Logger, device='cuda'):
        """
        Initialization of D2C layer.

        Args:
            mol (Mol): N molecular conformation with the same backbone and possibly different dihedral angles.
            angles (Tensor): a Tensor of shape (K, 4) where K is the number of dihedral angles for a conformer, 4 is (iAtomId, jAtomId, kAtomId, lAtomId).
        """
        super().__init__()
        self.mol = mol
        self.angles = angles
        self.alist = {}
        self.toBeMovedIdxList()
        self.device = device
        self.logger = logger

    def toBeMovedIdxList(self):
        """
        An implementation of toBeMovedIdxList from rdkit.
        See https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/MolTransforms/MolTransforms.cpp#L426
        """
        nAtoms = self.mol.GetNumAtoms()
        K = self.angles.shape[0]
        for i in range(K):
            iAtomId = self.angles[i, 1].item()
            jAtomId = self.angles[i, 2].item()
            if (iAtomId, jAtomId) not in self.alist:
                self.alist[(iAtomId, jAtomId)] = []
                visitedIdx = [False for _ in range(nAtoms)]
                stack = []
                stack.append(jAtomId)
                visitedIdx[iAtomId] = 1
                visitedIdx[jAtomId] = 1
                tIdx = 0
                wIdx = 0
                doMainLoop = True
                while len(stack) > 0:
                    doMainLoop = False
                    tIdx = stack[-1]
                    tAtom = self.mol.GetAtomWithIdx(tIdx)
                    neighbors = tAtom.GetNeighbors()
                    nbrIdx = 0
                    endNbrs = len(neighbors)
                    while nbrIdx != endNbrs:
                        wIdx = neighbors[nbrIdx].GetIdx()
                        if not visitedIdx[wIdx]:
                            visitedIdx[wIdx] = 1
                            stack.append(wIdx)
                            doMainLoop = True
                            break
                        nbrIdx += 1
                    if doMainLoop:
                        continue
                    visitedIdx[tIdx] = 1
                    stack.pop()
                self.alist[(iAtomId, jAtomId)].clear()
                for j in range(nAtoms):
                    if visitedIdx[j] and j != iAtomId and j != jAtomId:
                        self.alist[(iAtomId, jAtomId)].append(j)

    def transformPoint(self, pt: Tensor, input: Tensor, axis: Tensor):
        """
        An implementation of differentiable SetRotation and TransformPoint from rdkit.
        See https://github.com/rdkit/rdkit/blob/master/Code/Geometry/Transform3D.cpp

        Args:
            pt (Tensor): a Tensor of shape (N, 3) where N is the batch size, 3 is 3D coordinates.
            input (Tensor): a Tensor of shape (N) where N is the batch size, 1 is the rotation angle.
            axis (Tensor): a Tensor of shape (N, 3) where N is the batch size, 3 is 3D coordinates of the axis.
        """
        cosT = input.cos()
        sinT = input.sin()
        t = 1 - cosT
        X = axis[:, 0]
        Y = axis[:, 1]
        Z = axis[:, 2]
        N = input.shape[0]
        data = t[..., None, None] * axis[..., None].bmm(axis[:, None, :])
        mat1 = torch.stack([cosT        ,-sinT * Z  ,sinT * Y, 
                            sinT * Z    ,cosT       ,-sinT * X, 
                            -sinT * Y   ,sinT * X   ,cosT],
                           dim=1).reshape(N, 3, 3)
        data += mat1
        return data.bmm(pt[..., None]).squeeze()

    def setDihedralRad(self, input: Tensor, angle: Tensor, pos: Tensor) -> Tensor:
        """
        An implementation of differentiable setDihedralRad from rdkit.
        Note: This version has eliminated all fault checks temporarily. Add them if needed from the link below.
        See https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/MolTransforms/MolTransforms.cpp#L612

        Args:
            mol (Mol): N molecular conformation with the same backbone and possibly different dihedral angles.
            input (Tensor): a Tensor of shape (N) where N is the batch size, 1 is (dihedral angle value).
            angle (Tensor): a Tensor of shape (4) where 4 is (iAtomId, jAtomId, kAtomId, lAtomId).

        Returns:
            output (Tensor): a Tensor of shape (N, M, 3) where N is the batch size, M is the number of atoms, 3 is the 3D coordinates (x, y, z).
        """
        rIJ = pos[:, angle[1], :] - pos[:, angle[0], :]
        rJK = pos[:, angle[2], :] - pos[:, angle[1], :]
        rKL = pos[:, angle[3], :] - pos[:, angle[2], :]
        nIJK = rIJ.cross(rJK)
        nJKL = rJK.cross(rKL)
        m = nIJK.cross(rJK)
        values = input + \
            torch.atan2(
                m[:, None, :].bmm(nJKL[..., None]).squeeze() / (nJKL.norm(dim=-1) * m.norm(dim=-1)),
                nIJK[:, None, :].bmm(nJKL[..., None]).squeeze() / (nIJK.norm(dim=-1) * nJKL.norm(dim=-1)))
        rotAxisBegin = pos[:, angle[1], :]
        rotAxisEnd = pos[:, angle[2], :]
        rotAxis = normalize(rotAxisEnd - rotAxisBegin)
        for it in self.alist[(angle[1].item(), angle[2].item())]:
            pos[:, it, :] -= rotAxisBegin
            pos[:, it, :] = self.transformPoint(pos[:, it, :].clone(), values, rotAxis)
            pos[:, it, :] += rotAxisBegin
        return pos

    def forward(self, input: Tensor) -> Tensor:
        """
        An implementation of differentiable setDihedralRad from rdkit.
        TODO: This version has eliminated all fault checks temporarily. Add them if needed from the link below.
        See https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/MolTransforms/MolTransforms.cpp#L612

        Args:
            input (Tensor): a Tensor of shape (N, K) where N is the batch size, K is the number of dihedral angles for a conformer, 1 is (dihedral angle value).

        Returns:
            output (Tensor): a Tensor of shape (N, M, 3) where N is the batch size, M is the number of atoms, 3 is the 3D coordinates (x, y, z).
        """
        N, K = input.shape
        pos = []
        confs = self.mol.GetConformers()
        for conf in confs:
            pos.append(torch.tensor(
                conf.GetPositions(),
                dtype=torch.float32,
                device=self.device,
                requires_grad=True))
        pos = torch.stack(pos)
        # torch.set_printoptions(profile="full")
        for i in range(K):
            pos = self.setDihedralRad(input[:, i], self.angles[i, :], pos)
            # print(pos)
            # for r in range(N):
            #     Chem.rdMolTransforms.SetDihedralRad(
            #         confs[r],
            #         self.angles[i, 0].item(),
            #         self.angles[i, 1].item(),
            #         self.angles[i, 2].item(),
            #         self.angles[i, 3].item(),
            #         input[r, i].item())
            # newPos = []
            # for r in range(N):
            #     newPos.append(torch.tensor(confs[r].GetPositions(),
            #                             dtype=torch.float32))
            # newPos = torch.stack(newPos)
            # diff = (pos-newPos).abs()
            # # print(newPos)
            # print(diff.max(), diff.argmax(), (diff < 1e-6).sum())
            # print(diff)
        for i in range(N):
            for j in range(K):
                Chem.rdMolTransforms.SetDihedralRad(
                    confs[i],
                    self.angles[j, 0].item(),
                    self.angles[j, 1].item(),
                    self.angles[j, 2].item(),
                    self.angles[j, 3].item(),
                    input[i, j].item())
        # newPos = []
        # for r in range(N):
        #     newPos.append(torch.tensor(confs[r].GetPositions(),
        #                             dtype=torch.float32))
        # newPos = torch.stack(newPos)
        # diff = (pos-newPos).abs()
        # print(newPos)
        # print(diff.max(), diff.argmax(), (diff < 1e-6).sum())
        return pos
