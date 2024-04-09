import os
import time 
import sys
#import warnings 
import jax 
import jax.numpy as jnp
import pickle 
import numpy as np 

import ase 
from ase.calculators.calculator import Calculator, all_changes 
from ase.calculators.singlepoint import SinglePointCalculator 
import ase.units as units 
from atoms_lce import AtomsLocalChemEnv, SgprIO
import gpmodel 
import kernel

kcal_mol = units.kcal/units.mol 
inf = float ("inf")

class ActiveCalculator (Calculator):
    implemented_properties = ["energy", "forces", "stress", "free_energy"]

    def __init__ (
            self,
            json_data,
            calculator=None,
    ):
        Calculator.__init__(self)
        self._calc = calculator 
        self.step = 0
        self._last_test = self.step 
        self.fname_tape = json_data['tape']
        self.logfile = json_data['logfile']
        self.log ("AutoForce Calculator says Hello!", mode="w")
        kernel_kw = json_data['kernel_kw']
        self.ediff = json_data['ediff']
        self.fdiff = json_data['fdiff']
        self.test = json_data['test']
        self.nbeads = json_data['nbeads']
        self.species = kernel_kw['species']
        self._is_active = json_data['active']
        self._is_kernel_virial = json_data['kernel_virial']
        self.atoms_lce_fn = AtomsLocalChemEnv(species=kernel_kw['species'], 
                                              lmax=kernel_kw['lmax'], 
                                              nmax=kernel_kw['nmax'], 
                                              cutoff=kernel_kw['cutoff'],
                                              max_neigh=kernel_kw['max_neigh'])
        _, self.write_sgpr_fn = SgprIO (species=kernel_kw['species'], 
                                        lmax=kernel_kw['lmax'], 
                                        nmax=kernel_kw['nmax'], 
                                        cutoff=kernel_kw['cutoff'],
                                        max_neigh=kernel_kw['max_neigh'])
        
        self.kernel_kw = kernel_kw
        self.pckl_dir = json_data['pckl']

        self.gp_model = None
        if os.path.isdir (self.pckl_dir):
            fname = self.pckl_dir + '/model'
            if os.path.isfile (fname):
                with open (fname, 'rb') as fp:
                    self.gp_model = pickle.load (fp)
    
        self.l_save_gmodel = False 


    def log (self, mssge, mode="a"):
        import datetime 
        def date (fmt="%m/%d/%Y %H:%M:%S"):
            return datetime.datetime.now().strftime(fmt)

        if self.logfile:
            with open (self.logfile, mode) as f:
                f.write ("{} {}  {} \n".format(date(), self.step, mssge))

    @property
    def active (self):
        return self._calc is not None 
    
    @property
    def active_update (self):
        return self.active and ((self.step+1)%self.nbeads == 1)
    
    @property
    def data_size(self):
        return len(self.gp_model.ase_atoms), \
               len(self.gp_model.inducing_lce)


    def calculate (self, atoms, 
                   properties=["energy"], system_changes=all_changes):
        
        
        Calculator.calculate (self, atoms, properties, system_changes)
        l_debug = False 
        
        if l_debug:
            print ('0)start Active_Calculator')
            timeings = [time.time()] # 0) start

        # get atom_lce : (use instead of self.atoms.update in AutoFoce.calculator)
        self.atoms_lce = self.atoms_lce_fn (self.atoms)
        _ready_data = True

        # build a model 
        if self.gp_model is None:
            self.initiate_model (self.atoms)
            _ready_data = False

        if self.data_size[1] == 0 and self._calc is None:
            raise RuntimeError ("You forgot to assign a External (DFT) calculator")

        if l_debug:
            timeings.append (time.time()) # 1) desc (lce)
            print ('1)get atoms_lce:CPU_time', timeings[-1]-timeings[-2])

        # kernel: K(*,\rho_m)
        self.Ke_sm = kernel.get_gpr (self.atoms_lce, self.gp_model.inducing_lce)
        if l_debug:
            timeings.append (time.time()) # 2) kernel
            print ('2)get kernel:CPU_time', timeings[-1]-timeings[-2])
        
        
        # active learning
        #if self.active_update:
        n = 0
        if _ready_data and self._is_active:
            m, n = self.update ()
            if (m+n) > 0:
                self.l_save_gmodel = True
        
        if l_debug:
            timeings.append (time.time()) # 4) active
            print ('3)done active:CPU_time', timeings[-1]-timeings[-2])
        
        # energy/forces
        self.update_results ()
        
        if n > 0:
            # Note: target_energy (Ndata,1)
            dE = self.results["energy"] - self.gp_model.target_energy[-1,0]
            frc = self.results['forces'].reshape(-1)
            nfrc = frc.shape[0]
            dF = abs(frc - self.gp_model.target_forces[-nfrc:,0])
            target_stress = self.gp_model.target_virial[-6:,0]/self.atoms.get_volume()
            p_str = self.results['stress']
            dS = abs (p_str - target_stress)

            self.log ("errors (pre):  del-E: {:.3g}  max|del-F|: {:.3g}  mean|del-F|: {:.3g} mean|del-S|: {:.3g}".format(
                    dE, dF.max(), dF.mean(), dS.mean() ) )
            self.log ("predicted stress[GPa]: {}  {}  {}".format(p_str[0]/units.GPa, 
                                                                p_str[1]/units.GPa, 
                                                                p_str[2]/units.GPa))
        covloss_max = float (self.get_covloss().max())
        #if covloss_max > self.ediff:
        #    tmp = self.atoms
        #    tmp.calc = None 
        #    ase.io.Trajectory("active_uncertain.traj", "a").write (tmp)
                                                            
        if l_debug:
            timeings.append (time.time()) # 3) results
            print ('4)update results:CPU_time', timeings[-1]-timeings[-2])

        energy = self.results['energy']
        self.log (f"ML {energy} {self.atoms.get_temperature()} {covloss_max}")
        self.step += 1
        
        if (self.step)%5 == 0 and self.l_save_gmodel:
            self.save_gpmodel ()
            self.l_save_gmodel = False



    def update_results (self):
        
        energy = np.einsum('ij,j->', self.Ke_sm, self.gp_model.mu)
        
        energy_mean = 0.0
        for iz in self.gp_model.mean_weights.keys():
            count = (self.atoms.numbers == iz).sum()
            energy_mean += self.gp_model.mean_weights[iz]*count 

        #energy_mean = self.gp_model.target_energy.mean()
        self.results['energy'] = float(energy + energy_mean)
        #print ('precalc_energy', self.results['energy'], energy, energy_mean)
        Kf_sm, Kv_sm = kernel.get_gpr_forces_virial (self.atoms_lce, self.gp_model.inducing_lce)
        self.results['forces'] = np.einsum('ij,j->i', Kf_sm, self.gp_model.mu).reshape(-1,3) 
        self.results['stress'] = np.einsum('ij,j->i', Kv_sm, self.gp_model.mu) / self.atoms.get_volume()
        

    def _exact (self, atoms):
        tmp = atoms.copy()
        tmp.set_calculator (self._calc)
        energy = tmp.get_potential_energy()
        forces = tmp.get_forces()
        stress = tmp.get_stress()
        self.log ("exact energy: {}".format(energy))
        self.log ("exact stress[GPa]: {}  {}  {}".format(stress[0]/units.GPa,
                                                         stress[1]/units.GPa,
                                                         stress[2]/units.GPa))
            
        self._last_test = self.step 
        return energy, forces, stress 


    def snapshot (self, atoms):
        copy = atoms.copy()
        energy, forces, stress = self._exact (atoms)
        copy.set_calculator(SinglePointCalculator(copy, 
                            energy=energy, 
                            forces=forces, 
                            stress=stress))
        return copy

    def initiate_model (self, atoms):
        
        ase_atoms = self.snapshot(atoms)
        target_energy = ase_atoms.calc.results['energy'] 
        target_forces = ase_atoms.calc.results['forces'] 
        target_stress = ase_atoms.calc.results['stress']
        
        # inducing_lce : List[AtomLCE]
        inducing_lce = kernel.get_unique_lces(self.atoms_lce)
        # Ke (natom,nind)
        Ke = kernel.get_gpr_energy (self.atoms_lce, inducing_lce)
        # Kf (3*natom,nind), Kv (6,nind)
        Kf, Kv = kernel.get_gpr_forces_virial (self.atoms_lce, inducing_lce)
        
        M = kernel.get_gpr (inducing_lce, inducing_lce)
        target_virial = target_stress.reshape(-1,1)*self.atoms.get_volume()
        self.gp_model = gpmodel.GPModel (
            ase_atoms = [ase_atoms],
            target_energy=jnp.array([target_energy]).reshape(1,1),
            target_forces=target_forces.reshape(-1,1),
            target_virial=target_virial,
            atoms_lce    = [self.atoms_lce],
            inducing_lce =inducing_lce,
            Ke=Ke,
            Kf=Kf,
            Kv=Kv,
            M=M,
            mu=jnp.empty((len(inducing_lce))),
            mean_weights={iz: 0.0 for iz in self.species},
            _is_kernel_virial=self._is_kernel_virial
        )
        
        self.log ("seed size: {}  {}".format (*self.data_size) )

        self.gp_model.make_munu (optimize=True)
        self.save_gpmodel ()

        # save a sgpr file
        self.write_sgpr_fn (self.fname_tape, ase_atoms)
        for lce in inducing_lce:
            self.write_sgpr_fn (self.fname_tape, lce)

    
    def get_covloss (self):
        # b (m, n)
        #b = self.gp_model.LL_inv @ self.Ke_sm.T 
        LL = jax.scipy.linalg.cholesky (self.gp_model.M, lower=True)
        LL_inv = jnp.linalg.inv (LL)
        b = jnp.einsum('ik,jk->ij', LL_inv, self.Ke_sm)
        c = (b*b).sum(axis=0)
        beta = (1.0-c)
        beta = jnp.where (beta > 0.0, jnp.sqrt(beta), 0.0)
        
        vscale = jnp.array ([self.gp_model.vscale_sqrt[iz] 
                             for iz in self.atoms.numbers])
    
        return beta*vscale
    
    
    def update_inducing (self):

        added_indices = []
        added_covloss = inf
        added = 0
        while True:
            beta = self.get_covloss ()
            q = jnp.argsort (beta)[::-1] # descending
            new_lce = []
            added_k = []
            for k in q[:10]:
                if k not in added_indices and beta[k] > self.ediff:
                    new_lce.append(self.atoms_lce[k])
                    added_k.append(k)
            
            if len(new_lce) > 0: #beta[k] > self.ediff:
                self.gp_model.add_inducing_lce (new_lce)

                Ke_sm = kernel.get_gpr (self.atoms_lce, new_lce)
                self.Ke_sm = jnp.hstack ([self.Ke_sm, Ke_sm])
                
                for k in added_k:
                    added_indices.append (k)
                    added += 1
                added_covloss = beta[added_k[-1]]
                
                # save a sgpr file
                for lce in new_lce:
                    self.write_sgpr_fn (self.fname_tape, lce)
            else:
                break 
        

        if added > 0:
            self.log(
                "added indu: {} -> size: {} {} details: {:.2g} ".format(
                    added, *self.data_size, added_covloss
                )
            )
            

        return added



    def update_data (self, atoms):
        
        ase_atoms = self.snapshot (atoms)
        target_energy = ase_atoms.calc.results['energy'] 
        target_forces = ase_atoms.calc.results['forces'] 
        target_virial = ase_atoms.calc.results['stress']*ase_atoms.get_volume()
        self.gp_model.add_atoms_data (
            ase_atoms=ase_atoms,
            atoms_lce=self.atoms_lce,
            target_energy=jnp.array([target_energy]),
            target_forces=target_forces,
            target_virial=target_virial
        )
        self.write_sgpr_fn (self.fname_tape, ase_atoms)
        
        
    def update (self):

        energy0 = jnp.einsum('ij,j->', self.Ke_sm, self.gp_model.mu)        
        m = self.update_inducing ()

        n = 0
        if m > 0:
            energy1 = jnp.einsum('ij,j->', self.Ke_sm, self.gp_model.mu)
            #print ('**checka**', self.step, abs(energy1-energy0))

            if abs(energy1-energy0) > 0.1:
                print ('**update_data**', self.step, energy1, energy0)
                self.update_data(self.atoms)
                n = 1
            
        
        if (m + n) > 0:
            self.log(
                "fit error (mean,mae): E: {:.2g} {:.2g}   F: {:.2g} {:.2g}   R2: {:.4g}".format(
                    *(float(v) for v in self.gp_model._stats)
                )
            )
        
        return m, n


    def save_gpmodel (self):

        if not os.path.isdir (self.pckl_dir):
            os.makedirs(self.pckl_dir)
            
        fname_model = self.pckl_dir + "/model"
        with open (fname_model, 'wb') as f:
            pickle.dump (self.gp_model, f)

        fname_info =  self.pckl_dir + "/info_stats" 
        with open (fname_info, 'w') as f:
            ndata, nind = self.data_size
            print ("data: {}, inducing: {}".format (ndata, nind), file=f)

            e1, e2, f1, f2, r2 = (float(v) for v in self.gp_model._stats)
            print (f'ediff -> mean: {e1} std: {e2}', file=f)
            print (f'fdiff -> mean: {f1} std: {f2}', file=f)
            print (f'R2: {r2}', file=f)

            print (f'Mean: {self.gp_model.mean_weights}', file=f)



    def include_data (self, data_traj, fname_dat):
        from ase.io import read 

        if type(data_traj) == str:
            print ('data_traj', data_traj)
            data_traj = read (data_traj, index=slice(None))
        
        fout = open(fname_dat, 'w', 1)


        _calc = self._calc 
        for iconf, atoms in enumerate(data_traj):
            enr = atoms.get_potential_energy ()
            stress = atoms.get_stress (include_ideal_gas=True)/units.GPa
            #print ('exact_enr', enr)
            self._calc = atoms.calc 
            atoms.set_calculator (self)
            enr2 = atoms.get_potential_energy ()
            stress2 = atoms.get_stress (include_ideal_gas=True)/units.GPa
            print ('energy[exact/Pred]/stress[exact/Pred]:', iconf+1, enr, enr2, stress[:3].mean(), stress2[:3].mean(), file=fout)
            
        self._calc = _calc 
        self.save_gpmodel ()
        

    def include_tape (self, fname_tape):
        fname_tape_abspath = os.path.abspath (fname_tape)
        if type(fname_tape) == str:
            if fname_tape_abspath == os.path.abspath(self.fname_tape):
                raise RuntimeError(
                    "ActiveCalculator can not include it own .sgpr tape!"
                )
        kernel_kw = self.kernel_kw
        read_sgpr_fn, _ = SgprIO (species=kernel_kw['species'], 
                                lmax=kernel_kw['lmax'], 
                                nmax=kernel_kw['nmax'], 
                                cutoff=kernel_kw['cutoff'],
                                max_neigh=kernel_kw['max_neigh'])
        _calc = self._calc 
        data = read_sgpr_fn(fname_tape_abspath)

        cdata = 0
        for cls, obj in data:
            if cls == "atoms":
                enr = obj.get_potential_energy ()
                self._calc = obj.calc 
                obj.set_calculator(self)
                enr2 = obj.get_potential_energy ()
                cdata += 1
                print ('pred_enr, (error)', cdata, enr2, enr2-enr)
            elif cls == "local":
                new_lce = [obj]
                Ke_sm = kernel.get_gpr (new_lce,
                                        self.gp_model.inducing_lce)
                LL = jax.scipy.linalg.cholesky (self.gp_model.M, lower=True)
                LL_inv = jnp.linalg.inv (LL)
                b = jnp.einsum ('ik,jk->ij', LL_inv, Ke_sm)
                c = (b*b).sum(axis=0)
                if obj.Z_i in self.gp_model.vscale_sqrt:
                    vscale = self.gp_model.vscale_sqrt[obj.Z_i]
                else:
                    vscale = float("inf")
                beta = (1.0-c)*vscale 
                beta = jnp.where (beta > 0.0, jnp.sqrt(beta), 0.0)

                if beta > self.ediff:
                    self.gp_model.add_inducing_lce (new_lce)
                    Ke_sm = kernel.get_gpr (self.atoms_lce, new_lce)
                    self.Ke_sm = jnp.hstack ([self.Ke_sm, Ke_sm])
                    print ('added local')
        self._calc = _calc 
