"""
This module defines an ASE interface to MBX
"""

import subprocess
import numpy as np
import json 

from ase.calculators import calculator 
from ase.calculators.calculator import Calculator
import ase.units as units
from ase.io import read

ANG2BOHR = 1.0/units.Bohr
BOHR2ANG = units.Bohr
RCUT_QM = 6.0*ANG2BOHR
RCUT_EF = 9.0*ANG2BOHR
RCUT_LR = 12.0*ANG2BOHR

RCUT2_QM = RCUT_QM*RCUT_QM
RCUT2_EF = RCUT_EF*RCUT_EF
RCUT2_LR = RCUT_LR*RCUT_LR

def fragment_update (atoms):
    """
    atoms is from ase.atoms
    length unit is A
    """
        
    crds = atoms.get_positions()
    natom = len (atoms)
    idx = np.arange (natom)
    znum = atoms.get_atomic_numbers()
    idx_H = (znum == 1)
    xbox, ybox, zbox, _, _, _ = atoms.cell.cellpar()
    box = np.array ([xbox, ybox, zbox])
    frg = {}
    
    for im, ia in enumerate(idx[~idx_H]):
        
        crd_i = crds[ia]
        tmp_frg = {'idx': [ia], 'pos': [crd_i] }
        # H1
        rij = crds[ia+1] - crd_i 
        ### PBC
        sij = (rij/box)
        dij = -np.where (sij>0.0, np.floor(sij+0.5), np.ceil(sij-0.5))*box 
        rij = rij + dij 
        ###
        tmp_frg['idx'].append (ia+1)
        tmp_frg['pos'].append (crd_i+rij)
        # H2
        rij = crds[ia+2] - crd_i 
        ### PBC
        sij = (rij/box)
        dij = -np.where (sij>0.0, np.floor(sij+0.5), np.ceil(sij-0.5))*box 
        rij = rij + dij 
        ###
        tmp_frg['idx'].append (ia+2)
        tmp_frg['pos'].append (crd_i+rij)
        mol_name = f"mol{im}"
        tmp_frg['pos'] = np.array(tmp_frg['pos'])
        frg[mol_name] = tmp_frg 


    return frg  


class MBX (Calculator):
    """
    Class for doing MBX calculations
    """

    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']
    discard_results_on_any_change = True 

    def __init__ (self, restart=None,
                  ignore_bad_restart_file=Calculator._deprecated,
                  label='mbx',
                  atoms=None,
                  command_mm='mbx_single_pbc.x',
                  **kwargs):
        
        self.set (**kwargs)
        self.results = {}
        
        self.directory = '.'

        self.command_mm = command_mm
                

        self.input_data = {
            "Note": "This is a configuration file",
            "MBX": {
                "box" : [10,0.0,0.0,  0.0,10.00,0.0,  0.0,0.0,10.0],
                "twobody_cutoff"   : 9.0,
                "threebody_cutoff" : 5.0,
                "dipole_tolerance" : 1E-16,
                "dipole_method"     : "cg",
                "alpha_ewald_elec" : 0.6,
                "grid_density_elec" : 2.5,
                "spline_order_elec" : 6,
                "alpha_ewald_disp" : 0.6, 
                "grid_density_disp" : 2.5,
                "spline_order_disp" : 6,
                "ff_mons"   : [],
                "connectivity_file" : "", 
                "ttm_pairs" : [],
                "ignore_dispersion" : [],
                "use_lennard_jones" : [],
                "nonbonded_file" : "",
                "monomers_file" : "",
                "ignore_1b_poly" : [],
                "ignore_2b_poly" : [],
                "ignore_3b_poly" : []
            },
            "i-pi" : {
                "port" : 34543,
                "localhost" : "localhost"
            }

        }

        Calculator.__init__ (self, restart, 
                            ignore_bad_restart_file,
                            label,
                            atoms,
                            **kwargs)
    
    def calculate (self, atoms=None, properties=['energy'], system_changes=None):

        Calculator.calculate (self, atoms, properties, system_changes)
        #print ("IN QMMM_CALC")
        mbx_frg = fragment_update (self.atoms)
        #print ('START MBX')
        self.write_mbx_input (self.atoms, mbx_frg)
        #
        #proc = subprocess.Popen (self.command, shell=True, cwd=self.directory)
        errorcode = subprocess.call (self.command_mm, shell=True, cwd=self.directory)
        #proc.wait()
        if errorcode:
            raise calculator.CalculationFailed(
                '{} in {} returned an error: {:d}'.format(
                    self.name, self.directory, errorcode))
        #print ('DONE SUBPROCESS MBX')
        self.read_mbx_results (self.atoms, mbx_frg)

    
    def write_mbx_input (self, atoms, mbx_frg):
        """
        Write updated coordinates to a file
        """
        self.nwat = len(atoms)//3
        self.vol = atoms.get_volume ()
        
        cell = atoms.get_cell()
        self.input_data['MBX']['box'] = [cell[i,j] for i in range(3) 
                                  for j in range(3)]
        filename = 'mbx_pbc.json'
        with open (filename, 'w') as outfile:
            json.dump (self.input_data, outfile, indent=4)

        
        """
        write XYZ coordinates
        Transform the upper coorindates to the lower coordinates
        """
        filename = 'input.nrg'
        fout = open(filename, 'w')
        #crds = atoms.get_positions ()

        print ("SYSTEM NRG",file=fout)
        
        for ifrg in mbx_frg:
            iO, iH1, iH2 = mbx_frg[ifrg]['idx']
            crd_O, crd_H1, crd_H2 = mbx_frg[ifrg]['pos']

            print ("MOLECULE", file=fout)
            print ("MONOMER h2o", file=fout)

            s, (x,y,z)= atoms.symbols[iO], crd_O
            print (("{:4s} {:14.8f} {:14.8f} {:14.8f}".format(s, x, y, z)), file=fout)
            s, (x,y,z)= atoms.symbols[iH1], crd_H1
            print (("{:4s} {:14.8f} {:14.8f} {:14.8f}".format(s, x, y, z)), file=fout)
            s, (x,y,z)= atoms.symbols[iH2], crd_H2
            print (("{:4s} {:14.8f} {:14.8f} {:14.8f}".format(s, x, y, z)), file=fout)
            print("ENDMON", file=fout) 
            print("ENDMOL", file=fout)

        print("ENDSYS",file=fout)
        fout.close()    
    
    
    def read_mbx_results (self, atoms, mbx_frg, filename='mbx.out'):
        
        with open(filename, 'r') as fd:
            lines = fd.readlines()
            self.results['energy'] = float(lines[0].split()[0]) * units.kcal/units.mol
            self.results['free_energy'] = self.results['energy']
            
            natoms = len(atoms)
            forces = np.zeros ( (natoms,3) )
            charges = np.zeros ( natoms )

            iline = 1
            for key in mbx_frg:
                iO, iH1, iH2 = mbx_frg[key]['idx']
                words = lines[iline].split()
                fx, fy, fz, chg = float(words[0]), float(words[1]), float(words[2]), float(words[3])
                forces[iO] = fx, fy, fz
                charges[iO] = chg

                words = lines[iline+1].split()
                fx, fy, fz, chg = float(words[0]), float(words[1]), float(words[2]), float(words[3])
                forces[iH1] = fx, fy, fz
                charges[iH1] = chg

                words = lines[iline+2].split()
                fx, fy, fz, chg = float(words[0]), float(words[1]), float(words[2]), float(words[3])
                forces[iH2] = fx, fy, fz
                charges[iH2] = chg

                charges[iO] = -(charges[iH1]+charges[iH2])

                iline += 3

            self.results['forces'] = np.array (forces) * (units.kcal / units.mol)/units.Ang 
            self.charges = charges
            words = lines[-1].split()
            vir = np.array([float(words[i]) for i in range (6)]) * units.kcal/units.mol # (xx, xy, xz, yy, yz, zz)
            virial = np.array ([ [vir[0], vir[1], vir[2]], [vir[1], vir[3], vir[4]], [vir[2], vir[4], vir[5]] ])

            # (xx, yy, zz, yz, xz, xy)
            stress = -np.array([virial[0,0], virial[1,1], virial[2,2], virial[1,2], virial[0,2], virial[0,1]])/self.vol
            #self.results['stress'] = 0.5*stress
            self.results['stress'] = stress


    def write_qm_mbx_input (self, atoms, qm_frg, mbx_frg):

        #crds = atoms.get_positions()*ANG2BOHR
        atom_names = np.array(atoms.get_chemical_symbols())
        cell = atoms.get_cell().reshape(3,3)
        box = np.array( [cell[0,0], cell[1,1], cell[2,2]])*ANG2BOHR
        chgs = self.charges

        pair_list_QM = [] # for dimer
        #pair_list_LR = [] # for long-range Coulomb
        mol_neigh_EF = {} # for embedding field
        jcel_dij = {}
        qm_list = list(qm_frg.keys())
        atom_dict = {}
        
        for imol in qm_list:
            mol_neigh_EF[imol] = {}
            pos = qm_frg[imol]['pos']*ANG2BOHR
            mol_neigh_EF[imol][imol] = [[int(idx), atom_names[idx], chgs[idx], list(pos[i])] 
                            for i, idx in enumerate(qm_frg[imol]['idx'])]
            
        num_qm = len(qm_list)
        
        if num_qm > 1:
            imol = qm_list[0]
            pos = qm_frg[imol]['pos']*ANG2BOHR
            atom_dict[imol] = [[int(idx), atom_names[idx], chgs[idx], list(pos[i])] 
                               for i, idx in enumerate(qm_frg[imol]['idx'])]
            jcel_dij[imol] = np.zeros (3)

            pairs_qm = [ (i,j) for i in range(num_qm) for j in range(i+1, num_qm)]
            for im, jm in pairs_qm:
                imol, jmol = qm_list[im], qm_list[jm]

                imol_atnm = atom_names[qm_frg[imol]['idx']]
                jmol_atnm = atom_names[qm_frg[jmol]['idx']]
                
                pos_i = qm_frg[imol]['pos']*ANG2BOHR
                pos_j = qm_frg[jmol]['pos']*ANG2BOHR

                mark_imol_O = imol_atnm == 'O'
                mark_jmol_O = jmol_atnm == 'O'

                imol_O_site = pos_i[mark_imol_O]
                jmol_O_site = pos_j[mark_jmol_O]

                l_add_QM = False 
                l_add_EF = False 
        
                dij = np.zeros (3)
                jcel_dij[jmol] = np.zeros (3)
                for crd_i in imol_O_site:
                    for crd_j in jmol_O_site:
                        pos_ij = crd_j - crd_i
                        sij = (pos_ij/box)
                        dij = -np.where (sij > 0.0, np.floor(sij+0.5), np.ceil(sij-0.5))*box 
                        rij = pos_ij + dij 
                        rij2 = np.einsum('i,i->', rij, rij)

                        if rij2 < RCUT2_QM:
                            l_add_QM = True
                        if rij2 < RCUT2_EF:
                            l_add_EF = True
                    
                
                if l_add_QM:
                    pair_list_QM.append ([imol, jmol])
                    
                if l_add_EF:
                    
                    mol_neigh_EF[imol][jmol] = \
                        [[int(idx), atom_names[idx], chgs[idx], list(pos_j[i]+dij)] 
                         for i, idx in enumerate(qm_frg[jmol]['idx'])]

                    mol_neigh_EF[jmol][imol] = \
                        [[int(idx), atom_names[idx], chgs[idx], list(pos_i[i]-dij)] 
                         for i, idx in enumerate(qm_frg[imol]['idx'])]

        
        for imol in qm_list:
            pos_i = qm_frg[imol]['pos']*ANG2BOHR
            mark_imol_O = atom_names[qm_frg[imol]['idx']] == 'O'
            imol_O_site = pos_i[mark_imol_O]
            
            for jmol in mbx_frg:
                # between imol and mbx_mol
                pos_j = mbx_frg[jmol]['pos']*ANG2BOHR
                mark_O = atom_names[mbx_frg[jmol]['idx']] == 'O'
                jmol_O_site = pos_j[mark_O]

                dij = np.zeros(3)
                l_add_QM = False 
                l_add_EF = False 
                
                for crd_i in imol_O_site:
                    for crd_j in jmol_O_site:
                        pos_ij = crd_j - crd_i
                        sij = (pos_ij/box)
                        dij = -np.where (sij > 0.0, np.floor(sij+0.5), np.ceil(sij-0.5))*box 
                        rij = pos_ij + dij 
                        rij2 = np.einsum('i,i->', rij, rij)
        
                        if rij2 < RCUT2_QM:
                            l_add_QM = True 
                        if rij2 < RCUT2_EF:
                            l_add_EF = True
                            
                
                if l_add_QM:
                    pair_list_QM.append ( [imol, jmol] )
                    
                if l_add_EF:
                    mol_neigh_EF[imol][jmol] = \
                        [[int(idx), atom_names[idx], chgs[idx], list(pos_j[i]+dij)] 
                         for i, idx in enumerate(mbx_frg[jmol]['idx'])] 


        filename = 'qmmm.json'
        with open (filename, 'w') as outfile:
            data = {}
            data['natoms'] = len(atoms)
            data['efield'] = mol_neigh_EF
            data['pair'] = pair_list_QM 
            
            
            json.dump (data, outfile, indent=4)


    def read_qm_mbx_results(self, atoms, filename='qmmm.out'):

        with open(filename, 'r') as fd:
            lines = fd.readlines()
            # Hartree --> eV
            #print ('energy', self.results['energy'])
            self.results['energy'] += float(lines[0].split()[0]) 
            #print ('updated energy', self.results['energy'])
            self.results['free_energy'] = self.results['energy']
            
            natoms = len(atoms)
            forces = np.zeros ( (natoms,3) )
            
            for iat, line in enumerate(lines[1:1+natoms]):               
                words = line.split()
                fx, fy, fz = float(words[0]), float(words[1]), float(words[2])
                forces[iat] = fx, fy, fz
            
            self.results['forces'] += forces 
            
            words = lines[-1].split()
            vir = np.array([float(words[i]) for i in range (6)]) # (xx, xy, xz, yy, yz, zz)
            virial = np.array ([ [vir[0], vir[1], vir[2]], [vir[1], vir[3], vir[4]], [vir[2], vir[4], vir[5]] ])

            # (xx, yy, zz, yz, xz, xy)
            stress = -np.array([virial[0,0], virial[1,1], virial[2,2], 
                                virial[1,2], virial[0,2], virial[0,1]])/self.vol
            self.results['stress'] += stress

if __name__ == "__main__":
    #from ase.io import read 
    import sys
    
    #atoms = read ('w256.xyz')
    atoms = read ('socket_send.xyz')
    cell = 19.716*np.eye(3)
    atoms.set_cell (cell)
    atoms.pbc = [1, 1, 1]

    atoms.calc = QMMM()
    volbox = atoms.get_volume()
    enr = atoms.get_potential_energy()
    frc = atoms.get_forces ()
    print ('enr', enr)
    print ('frc', frc[:6])
    sys.exit()
    #stress = atoms.get_stress ()
    if atoms.calc.rank == 0:
        print ('enr', enr)
        #print ('frc', frc[:3], frc.shape)
        #print ('frc', frc[-3:], frc.shape)
        print ('stress', stress)
        print ('stress/2', stress/2)
        #print ('virial', -stress*volbox)
    sys.exit()
    stress = atoms.calc.calculate_numerical_stress (atoms)
    print ('numerical_stress', stress)

    #recip = atoms.cell.reciprocal()
    crds = atoms.get_positions()
    #print ('crds', crds[:3])
    scaled_positions = atoms.cell.scaled_positions(crds)
    #print ('scaled crds', scaled_positions[:3])
    '''
    eps = 3.0e-5
    dedl = np.zeros ((3,3))

    for i in range (3):
        for j in range (3):
            cell2 = cell
            cell2[i,j] = cell[i,j] - eps 
            atoms.set_cell (cell2)
            xbox, ybox, zbox, alpha, beta, gamma = atoms.cell.cellpar()
            atoms.set_cell ([xbox, ybox, zbox, alpha, beta, gamma])
            pos = atoms.cell.cartesian_positions (scaled_positions)
            atoms.set_positions(pos)
            enr0 = atoms.get_potential_energy()

            cell2[i,j] = cell[i,j] + eps 
            atoms.set_cell (cell2)
            xbox, ybox, zbox, alpha, beta, gamma = atoms.cell.cellpar()
            atoms.set_cell ([xbox, ybox, zbox, alpha, beta, gamma])
            pos = atoms.cell.cartesian_positions (scaled_positions)
            atoms.set_positions(pos)
            enr1 = atoms.get_potential_energy()

            dedl[i,j] = 0.5*(enr1-enr0)/eps
            print ('i,j', i,j, dedl[i,j], enr0, enr1)

    print ('dE/dV', dedl)
    '''

    dedl = np.array ( [[-5.23411962, -2.4778249,   0.32486941],
            [-0.33260268, -5.40360099,  0.70126983],
            [-0.21465232, -0.02999351, -2.64665581]] )

    virn = np.einsum('kj,ki->ij', dedl, cell)
    print ('stress (numerical): dU/dV/V', virn/volbox)
    
