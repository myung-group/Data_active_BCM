import jax.numpy as jnp
from matscipy.neighbours import neighbour_list 
import io
import os
from ase.io import read 
from ase.atoms import Atoms

from dataclasses import dataclass
from sesoap import SeSoap

#import sys


@dataclass
class AtomLCE: # local chemical environment
    atom_i: int  # (the first atom)
    Z_i: int # (the atomic number)
    atom_js: jnp.ndarray
    Z_js: jnp.ndarray
    rij: jnp.ndarray
    p_sp: jnp.ndarray
    dp_sp: jnp.ndarray


def AtomsLocalChemEnv (species, lmax, nmax, cutoff, max_neigh=100):

    soap_fun = SeSoap (species, lmax, nmax, cutoff)
    
    def compute (atoms):
        """
        atoms: ase.Atoms
        """
        iatoms, jatoms, dij, Dij = neighbour_list(
            quantities="ijdD",
            atoms=atoms,
            cutoff=cutoff,
        )

        iatoms = jnp.array(iatoms)
        jatoms = jnp.array(jatoms)
        Dij = jnp.array(Dij)
        atoms_Z = jnp.array(atoms.numbers)
        natoms = len(atoms)

        atoms_lce = []
        
        for ia in range (natoms):
            mark = (iatoms==ia)
            
            n_neigh = mark.sum()
            if n_neigh < max_neigh+1:
                Z_js = jnp.empty(max_neigh, dtype=jnp.int64)
                rij = jnp.empty( (max_neigh,3) )
                atom_js = jnp.empty (max_neigh, dtype=jnp.int64)
                atom_js = atom_js.at[:n_neigh].set(jatoms[mark])
                atom_js = atom_js.at[n_neigh:].set (0)
                rij = rij.at[:n_neigh].set( Dij[mark] )
                rij = rij.at[n_neigh:].set (jnp.array ([2.0, cutoff, 2.0]))
                Z_js = Z_js.at[:n_neigh].set(atoms_Z[atom_js[:n_neigh]])
                Z_js = Z_js.at[n_neigh:].set (-1)
            else:
                tij = dij[mark]
                q = jnp.argsort (tij)[:max_neigh]
                rij = Dij[mark][q]
                atom_js=jatoms[mark][q]
                Z_js = atoms_Z[atom_js]
            
            p_sp, dp_sp = soap_fun (rij, Z_js)
            
            #iz = int (atoms_Z[ia])
            lce = AtomLCE(ia, atoms_Z[ia], atom_js, Z_js, rij, p_sp, dp_sp)

            atoms_lce.append (lce)
     
        return atoms_lce
    
    return compute 


def SgprIO (species, lmax, nmax, cutoff, max_neigh=100):

    soap_fun = SeSoap (species, lmax, nmax, cutoff)
    atoms_js = jnp.arange (max_neigh)+1

    def read_lce (blk):
        ia = 0
        Z_i = int (blk[0].strip())
        Z_js = []
        rij = []
        for line in blk[1:]:
            s = line.split()
            Z_js.append (int(s[0]))
            rij.append ([float(s[i]) for i in [1,2,3]])
        
        n_neigh = len(Z_js)
        if n_neigh < max_neigh+1:
            for _ in range (n_neigh, max_neigh):
                Z_js.append (-1)
                rij.append ([2.0, cutoff, 2.0])
            rij = jnp.array (rij)
            Z_js = jnp.array (Z_js)
        else:
            rij = jnp.array (rij)
            dij = jnp.sqrt(jnp.einsum('ij,ij->i', rij, rij))
            q = jnp.argsort (dij)[:max_neigh]
            rij = rij[q]
            Z_js = jnp.array(Z_js)[q]

        p_sp, dp_sp = soap_fun (rij, Z_js)
        obj = AtomLCE (ia, Z_i, atoms_js, Z_js, rij, p_sp, dp_sp)
        return obj 
    

    def convert_block (typ, blk):
        if typ == "atoms":
            obj = read (io.StringIO("".join(blk)), format="extxyz")
        elif typ == "local":
            obj = read_lce (blk)
        elif typ== "params":
            obj = {}
            for line in blk:
                a, b = line.split ()
                obj[a] = eval (b)
        else:
            raise RuntimeError(f"type {typ} is unknown")
        return obj 

    def read_sgpr (fname):
        #from collections import Counter 

        if not os.path.isfile (fname):
            print (f"{fname} does not exist! -> []")

        with open (fname, 'r') as f:
            lines = f.readlines()
    
        on = False 
        data = []
        c = {"atoms": 0, "local": 0}

        for line in lines:
            if not on:
                if line.startswith("start:"):
                    on = True 
                    typ = line.split()[-1]
                    blk = []
            else:
                if line.startswith("end:"):
                    assert line.split()[-1] == typ
                    on = False 
                    obj = convert_block(typ, blk)
                    data.append ((typ, obj))
                    c[typ] += 1
                else:
                    blk.append (line)
    
        print (f"included {fname} {c}")

        return data 
    
    def write_sgpr (fname, obj):

        if type (obj) == AtomLCE:
            with open (fname, "a") as f:
                f.write ("\nstart: local\n")
                f.write (f"{obj.Z_i:4d}\n")
                for zj, rij in zip (obj.Z_js, obj.rij):
                    f.write ("{:4d} {:16.8f} {:16.8f} {:16.8f}\n".format(
                        zj, rij[0], rij[1], rij[2]
                    ))
                f.write ("end:local\n")

        elif type(obj) == Atoms:
            with open (fname, "a") as f:
                f.write ("\nstart: atoms\n")
            obj.write (fname, format="extxyz", append=True)
            with open (fname, "a") as f:
                f.write ("end: atoms\n")
        else:
            raise RuntimeError(f"no recipe for {type(obj)}")

    return read_sgpr, write_sgpr 


if __name__ == '__main__':
    
    import sys
    atoms = read ('md_bcm.traj', index=0)
    atoms_lce_fn = AtomsLocalChemEnv(species=[1,8], 
                                    lmax=3, 
                                    nmax=3, 
                                    cutoff=6.0)
    atoms_lce = atoms_lce_fn (atoms)
    expont = 4.0

    
    for IS in atoms_lce[:5]:
        
    
        for LS in atoms_lce[-10:]:
            print ('LS IS', LS.Z_i, IS.Z_i)
            if LS.Z_i == IS.Z_i and LS.Z_i > 0:
                cc = (LS.p_sp*IS.p_sp).sum()
                dg = jnp.einsum('ijkl,ij->kl', LS.dp_sp, IS.p_sp)
                fr = expont*cc**(expont-1) * dg 
                #frc = frc.at[LS.atom_i].add (fr.sum(axis=0))
                print ('fr', fr)
                print ('jatoms', LS.atom_js)
    