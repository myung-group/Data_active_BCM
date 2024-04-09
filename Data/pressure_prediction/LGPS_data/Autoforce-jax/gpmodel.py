
import re
from xmlrpc.client import Boolean
import jax
from jax.scipy.optimize import minimize
from dataclasses import dataclass
from typing import Any, List, Dict

import ase 
import jax.numpy as jnp 
from atoms_lce import AtomLCE
import kernel 

import sys


@dataclass
class GPModel:
    ase_atoms: List[ase.Atoms] # list (ase.Atoms)
    target_energy: Any # jnp.ndarray
    target_forces: Any # jnp.ndarray
    target_virial: Any # jnp.ndarray
    atoms_lce : List[List[AtomLCE]] # list (AtomLCE) 
    inducing_lce: List[AtomLCE] # list (AtomLCE)
    Ke: Any # jnp.ndarray
    Kf: Any # jnp.ndarray
    Kv: Any # jnp.ndarray
    M:  Any # jnp.ndarray
    mu: Any # jnp.ndarray 
    mean_weights: Dict[int, float]
    _is_kernel_virial: Boolean
    
    def add_atoms_data (self, 
                        ase_atoms,
                        atoms_lce,
                        target_energy,
                        target_forces,
                        target_virial):
        
        self.ase_atoms.append (ase_atoms)
        self.atoms_lce.append (atoms_lce)
        
        # (self.N, 1) --> (self.N+N, 1)
        self.target_energy = jnp.vstack ([self.target_energy, target_energy.reshape(-1,1)])
        self.target_forces = jnp.vstack ([self.target_forces, target_forces.reshape(-1,1)])
        self.target_virial = jnp.vstack ([self.target_virial, target_virial.reshape(-1,1)])

        # UPDATE KE, KF, KV, M
        Ke = kernel.get_gpr_energy (atoms_lce, self.inducing_lce)
        Kf, Kv = kernel.get_gpr_forces_virial (atoms_lce, self.inducing_lce)
        self.Ke = jnp.vstack ([self.Ke, Ke])
        self.Kf = jnp.vstack ([self.Kf, Kf])
        self.Kv = jnp.vstack ([self.Kv, Kv])

        self.make_munu (optimize=True)


    def add_inducing_lce (self, inducing_lce, remake=True):
        
        Ke = []
        Kf = []
        Kv = []
        nind = len(inducing_lce)
        for atoms_lce in self.atoms_lce:
            Ke.append (kernel.get_gpr_energy (atoms_lce, inducing_lce))
            _Kf, _Kv = kernel.get_gpr_forces_virial (atoms_lce, inducing_lce)
            Kf.append (_Kf)
            Kv.append (_Kv)

        Ke = jnp.array (Ke).reshape(-1,nind)
        Kf = jnp.array (Kf).reshape(-1,nind)
        Kv = jnp.array (Kv).reshape(-1,nind)

        # natoms_lce is fixed, ninducing_lce increases
        # (self.N, self.M) --> (self.N, self.M + nind)
        self.Ke = jnp.hstack ([self.Ke, Ke])
        self.Kf = jnp.hstack ([self.Kf, Kf])
        self.Kv = jnp.hstack ([self.Kv, Kv])

        M  = kernel.get_gpr (self.inducing_lce, inducing_lce) # (M,nind)
        MT = M.T #(nind,M)
        # (M, M) --> (M, M+nind)
        M = jnp.hstack ([self.M, M])
        M2 = kernel.get_gpr (inducing_lce, inducing_lce)
        # (nind, M) --> (nind, M + nind)
        MT = jnp.hstack([MT,M2])
        # (M,M+nind) --> (M+nind, M+nind)
        self.M = jnp.vstack([M,MT])
        self.inducing_lce += inducing_lce
        
        if remake:
            self.make_munu (optimize=False)


    def make_stats (self):

        def cd(pred, target):
            var1 = target.var()
            var2 = (target - pred).var()
            R2 = 1 - var2 / var1
            return R2

        data_natoms = jnp.array ([len(atoms) for atoms in self.ase_atoms])
        
        ediff = jnp.einsum('ij,j->i', self.Ke, self.mu)
        ediff = ediff/data_natoms 
        predicted_forces = jnp.einsum('ij,j->i', self.Kf, self.mu)
        fdiff = self.target_forces.reshape(-1) - predicted_forces
        force_r2 = cd(predicted_forces, self.target_forces.reshape(-1))

        self._stats = [
            ediff.mean(),
            ediff.std(),
            fdiff.mean(),
            fdiff.std(),
            force_r2,
        ]
        
        # predictive variance for inducing_lce
        # vscale * [k(x,x) - k(x, m)*k(m,m)^{-1}k(m,x)] 
        #      
        indu_Z = jnp.array ([lce.Z_i 
                             for lce in self.inducing_lce])
        species = jnp.unique (indu_Z).tolist()
        ind_counts = {iz: (indu_Z == iz).sum() for iz in species}
        ind_mu = self.mu * (self.M @ self.mu)
        self.vscale_sqrt = {iz : jnp.sqrt(ind_mu[indu_Z==iz].sum()/ind_counts[iz])
                        for iz in species}
        
        
    def make_munu (self, optimize=False):
    
        LL = jax.scipy.linalg.cholesky (self.M, lower=True)

        def make_mu (with_energies=None):

            noise = 0.01
            if with_energies is None:
                A = jnp.vstack ([self.Kf, self.Kv, noise*LL.T])
                Y = jnp.vstack ([self.target_forces, 
                                 self.target_virial,
                                 jnp.zeros ( (LL.shape[0], 1) ) ]).reshape(-1)
            else:
                A = jnp.vstack ([self.Ke, self.Kf, self.Kv, noise*LL.T])
                Y = jnp.vstack ([with_energies.reshape(-1,1), 
                                 self.target_forces,
                                 self.target_virial,
                                 jnp.zeros ( (LL.shape[0], 1) ) ]).reshape(-1)
            A = jax.device_put (A)   
            Q, R = jnp.linalg.qr (A)
            R = jax.device_put (R)   
            R_inv = jnp.linalg.inv(R)
            mu = jnp.einsum('ij,kj,k->i', R_inv, Q, Y)
            
            return mu
        
        def make_mu_old (with_energies=None):

            noise = 0.01
            if with_energies is None:
                A = jnp.vstack ([self.Kf, noise*LL.T])
                Y = jnp.vstack ([self.target_forces, 
                                 jnp.zeros ( (LL.shape[0], 1) ) ]).reshape(-1)
            else:
                A = jnp.vstack ([self.Ke, self.Kf, noise*LL.T])
                Y = jnp.vstack ([with_energies.reshape(-1,1), 
                                 self.target_forces,
                                 jnp.zeros ( (LL.shape[0], 1) ) ]).reshape(-1)
            A = jax.device_put (A)   
            Q, R = jnp.linalg.qr (A)
            R = jax.device_put (R)   
            R_inv = jnp.linalg.inv(R)
            mu = jnp.einsum('ij,kj,k->i', R_inv, Q, Y)
            
            return mu
        
        ## optimize mean 
        mean_counts = {iz : jnp.array([ (atoms.numbers == iz).sum() 
                              for atoms in self.ase_atoms]) 
                        for iz in self.mean_weights.keys()}

        x0 = jnp.array([self.mean_weights[iz] for iz in self.mean_weights.keys()])
        c0 = jnp.array([mean_counts[iz] for iz in self.mean_weights.keys()])

        
        def objective_mean (x0, c0): 
            # i: nspec, j: ndata
            return jnp.einsum('i,ij->j', x0, c0).reshape(-1,1)
        
        def loss_fn (w, count):
            e = objective_mean (w, count)
            #diff = (e-delta_energies)
            diff = (e - self.target_energy)
            return (diff*diff).mean()
        
        if optimize:
            results = minimize (loss_fn, x0=x0, args=(c0,), method='BFGS')
            x0 = results.x
            self.mean_weights = {iz: float(x) for x, iz in zip (x0, self.mean_weights.keys())}


        residual = self.target_energy - objective_mean(x0, c0)
        
        if self._is_kernel_virial:
            self.mu = make_mu (with_energies=residual)
        else:
            self.mu = make_mu_old (with_energies=residual)
            
        self.make_stats()
        
