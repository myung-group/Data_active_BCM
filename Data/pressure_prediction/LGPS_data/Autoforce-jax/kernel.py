from threading import local
import jax
import jax.numpy as jnp
import sys 

def get_unique_lces (atoms_lce, expont=4):

    unique_lces = []
    unique = []
    for i, LS in enumerate(atoms_lce):
        is_unique = True
        for j in unique:
            IS = atoms_lce[j]
            if IS.Z_i == LS.Z_i:
                val = (LS.p_sp * IS.p_sp).sum()**expont
                if val >= 0.95:
                    is_unique = False 
                    break 
        if is_unique:
            unique_lces.append (LS)
            unique.append (i)
    
    return unique_lces

# 
def get_gpr (lce_left, 
             lce_right,
             expont=4):
    
    @jax.jit 
    def local_gpr (p_sp, Z_i):

        return jnp.array([ jnp.where(IS.Z_i == Z_i,
                           (p_sp*IS.p_sp).sum()**expont,
                            0.0) 
                for IS in lce_right   ])
    
    p_sp = jnp.array([LS.p_sp for LS in lce_left])
    Z_i = jnp.array([LS.Z_i for LS in lce_left])
    
    return jax.vmap (local_gpr) (p_sp, Z_i)
    
        

def get_gpr_energy (lce_left,
                    lce_right, 
                    expont=4):

    # cov (n_lce, n_inducing)
    cov = get_gpr (lce_left, 
                   lce_right,
                   expont)
    # cov (1, n_inducing)
    return jnp.array(cov.sum(axis=0)).reshape(1,-1)



def get_gpr_forces_virial_slow (lce_left,
                           lce_right,
                           expont=4):
    
    natoms = len (lce_left)
    vir_idx = jnp.array([0, 4, 8, 5, 2, 1])


    def local_gpr_forces_virial (p_sp, Z_i):

        frc = jnp.zeros ( (natoms, 3) )
        vir = jnp.zeros ( (3,3) )

        for LS in lce_left:
            cc = jnp.where (Z_i == LS.Z_i, (p_sp*LS.p_sp).sum(), 0.0)
            dg = jnp.einsum('ijkl,ij->kl', LS.dp_sp, p_sp)
            fr = expont*cc**(expont-1)*dg 
            frc = frc.at[LS.atom_i].add (fr.sum(axis=0))
            frc = frc.at[LS.atom_js].add (-fr)
            vir += jnp.einsum('ij,ik->jk', fr, LS.rij)

        vir = vir.reshape(-1)

        return frc.reshape(-1), vir[vir_idx] 

    
    # 
    if len(lce_right) > 1:
        p_sp = jnp.array([IS.p_sp for IS in lce_right])
        Z_i = jnp.array([IS.Z_i for IS in lce_right])
        gpr_frc, gpr_vir = jax.vmap(local_gpr_forces_virial) (p_sp, Z_i)
    else:
        gpr_frc, gpr_vir = local_gpr_forces_virial (lce_right[0].p_sp, 
                                                    lce_right[0].Z_i)
        gpr_frc = gpr_frc.reshape(1,-1)
        gpr_vir = gpr_vir.reshape(1,-1)

    return gpr_frc.T, gpr_vir.T



def get_gpr_forces_virial (lce_left,
                            lce_right,
                            expont=4):
    
    natoms = len (lce_left)
    vir_idx = jnp.array([0, 4, 8, 5, 2, 1])

    @jax.jit
    def local_gpr_forces_virial (p_sp, dp_sp, 
                                 atom_i, Z_i, atom_js, rij,
                                 p_sp_left, Z_i_left):

        # rij = rj - ri
        #for IS in lce_right_batch:
        nind = p_sp_left.shape[0]
        frc = jnp.zeros ( (nind, natoms, 3) )
        
        cc = jnp.where (Z_i == Z_i_left, jnp.einsum('jk,ijk->i',p_sp,p_sp_left), 0.0)
        dg = jnp.einsum('ijkl,oij->okl', dp_sp, p_sp_left)
        cc1 = expont*cc**(expont-1)
        # g_ij = gj - gi
        gij = jnp.einsum('o,okl->okl',cc1, dg)
        
        frc = frc.at[:,atom_i].add (gij.sum(axis=1))
        frc = frc.at[:,atom_js].add (-gij)
        # this stress virial is gij * rij instead of pressure virial = fij * rij
        vir = jnp.einsum('oij,ik->ojk', gij, rij)
        
        return frc.reshape(nind,-1), vir.reshape(nind,-1)[:,vir_idx]

    
    #
    p_sp = jnp.array([LS.p_sp for LS in lce_left])
    dp_sp = jnp.array([LS.dp_sp for LS in lce_left])
    rij = jnp.array([LS.rij for LS in lce_left])
    atom_i = jnp.array([LS.atom_i for LS in lce_left])
    atom_js = jnp.array([LS.atom_js for LS in lce_left])
    Z_i = jnp.array([LS.Z_i for LS in lce_left])
    
    #(natom, nind, natom*3)
    nind = len(lce_right)
    nbatch = 20
    gpr_frc_vstack = None
    gpr_vir_vstack = None
    for ist0 in range (0, nind, nbatch):
        ied0 = jnp.where (ist0+nbatch < nind, ist0+nbatch, nind)
        lce_right_batched = lce_right[ist0:ied0]
        p_sp_left = jnp.array([IS.p_sp for IS in lce_right_batched])
        Z_i_left = jnp.array([IS.Z_i for IS in lce_right_batched])

        #(natom, nind, natom*3)
        gpr_frc, gpr_vir = jax.vmap (local_gpr_forces_virial, 
                                     in_axes=(0,0,0,0,0,0,None, None)) (p_sp, dp_sp, 
                                             atom_i, Z_i, atom_js, rij, 
                                             p_sp_left, Z_i_left)
        
        if gpr_frc_vstack is None:
            gpr_frc_vstack = gpr_frc.sum(axis=0)
            gpr_vir_vstack = gpr_vir.sum(axis=0)
        else:
            gpr_frc_vstack = jnp.vstack([gpr_frc_vstack, gpr_frc.sum(axis=0)])
            gpr_vir_vstack = jnp.vstack([gpr_vir_vstack, gpr_vir.sum(axis=0)])

    
    return gpr_frc_vstack.T, gpr_vir_vstack.T

