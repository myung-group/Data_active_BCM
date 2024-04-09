import jax
import jax.numpy as jnp
from spherical_harmonics import Ylm
import sys
import numpy as np

def cart2sph (pos):
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    rxy_sq = x*x + y*y #jnp.atleast_1d (x*x + y*y)
    rxy = jnp.sqrt (rxy_sq)
    r = jnp.sqrt (rxy_sq + z*z)
    theta = jnp.arctan2 (rxy,z)
    phi = jnp.arctan2 (y, x)

    return r, theta, phi


def poly_cut (cutoff):

    @jax.jit
    def compute (dij):
        # dij (1, nloc)
        step = jnp.where (dij < cutoff, jnp.ones_like(dij), jnp.zeros_like(dij))
        fval = (1.0-dij/cutoff)**2
        gval = -2*(1.0-dij/cutoff)/cutoff
        return step*fval, step*gval 
    
    return compute 



def factorial (n):
    a = jnp.arange (n) + 1
    return jnp.prod (a)



def SeSoap (species, lmax, nmax, cutoff):

    radial_fun = poly_cut(cutoff)
    
    _fac = jnp.array ([factorial(n) for n in range (lmax+nmax+1)])
    
    one = jnp.ones ( (lmax+1, lmax+1) )
    Yr = 2*jnp.tril(one) - jnp.eye(lmax+1)
    Yi = 2*(jnp.triu(one)- jnp.eye(lmax+1))
    
    aa = jnp.array ([[
        1.0/((2.*l+1) * 2**(2*n+l)*_fac[n]*_fac[n+l])
        for l in range (lmax+1)]
        for n in range (nmax+1)
    ]) # a_nl in Eq. (18) and sqrt(2 * l + 1) in Eq. (20)
    # a(nmax+1, lmax+1)
    # nnl (nmax+1, nmax+1, lmax+1) = a * a'
    nnl = jnp.sqrt (aa[None]*aa[:,None])
    
    nspec = len(species)
    
    compute_ylm_rl = Ylm (lmax)
    
    rad_n = 2*jnp.arange (nmax+1, dtype=jnp.float64)
	#rad_n = np.array (nmax+1, dtype=jnp.float64)
	#rad_n = 2*jnp.arange (rad_n)
    
	# (Descriptor_SoSoap)
    @jax.jit
    def compute (cij, Z_js):
        '''
        loc : struct LocalData: loc.rij : |Rj - Ri|: jnp.ndarray (nloc)
        '''
        # Default Radii: if H (Z=1), 0.5, otherwise 1.0
        units = jnp.where (Z_js == 1, 0.5, 1.0)
        rij = cij/units.reshape(-1,1)
        # dij (nloc)
        dij, theta, phi = cart2sph (rij)
        
        dn = dij[None]**rad_n[:,None]
        
        rf, d_rf = radial_fun(units*dij)
        jnp_exp = jnp.exp (-0.5*dij*dij)
        rf_exp = rf*jnp_exp 
        # (nmax+1, nloc)
        #f = rf*jnp.exp(-0.5*dij*dij)*(dij**n)
        f = jnp.einsum('j,ij->ij', rf_exp, dn)
        
        ## Grad_Radial
        d_jnp_exp = -dij*jnp_exp 
        d_rad = (units*d_rf)*jnp_exp + rf*d_jnp_exp
        ## 
        
        # Spherical Harmonic Part 
        # r**l Ylm
        # Y_rl (lmax+1, lmax+1, nloc)
        Y_rl, dY_rl = compute_ylm_rl (dij, phi, theta)
        
        # ff (nmax+1, lmax+1, lmax+1, nloc)
        #ff = rad.reshape (nmax+1, 1, 1, -1)*Y_rl.reshape (1, lmax+1, lmax+1, -1)
        ff = jnp.einsum('il,jkl->ijkl', f, Y_rl)
        
        #idx = jnp.arange (dij.shape[0]) # nloc
        c = jnp.array ([jnp.where(num==Z_js, ff, 0).sum(axis=-1)
                        for num in species])
        
        # Eq. (19) I = \sum_n C (nspec, n, l, m) C(nspec', n, l, m')
        nnp = (c[None,:,None]*c[:,None,:,None])
        p = (nnp*Yr).sum(axis=-1) + (nnp*Yi).sum(axis=-2)
        p = p*nnl
        
        # GRAD ##
        d_dn = dij[None]**(rad_n[:,None]-1) #(nmax+1, nloc)
        df = jnp.einsum('j,ij->ij',d_rad,  dn) \
            + jnp.einsum('j,i,ij->ij',rf_exp, rad_n, d_dn)
        
        df = df[...,None] * rij / dij[:, None]
        
        dc = (df[:,None,None]*Y_rl[None,...,None] + 
              f[:,None,None,:,None]*dY_rl[None])
        
        dc = jnp.array(
            [(Z_js==num)[:,None]*dc for num in species],
            dtype=dij.dtype)
        
        dnnp = (
                c[None, :, None, ..., None, None] * dc[:, None, :, None]
                + dc[None,:,None,] * c[:, None, :, None, ..., None, None]
            )
        dp = (dnnp * Yr[..., None, None]).sum(axis=-3) + \
             (dnnp * Yi[..., None, None]).sum(axis=-4)
        dp = dp*nnl[...,None,None]/units.reshape(-1,1)
        
        ###
        norm = jnp.linalg.norm (p)
        p = p / norm 
        dp = dp / norm
        
        dp = dp - \
            p[...,None,None]*(p[...,None,None]*dp).sum(axis=(0,1,2,3,4))
        
        p_sp = p.reshape (nspec**2, -1) # Eq. (20)
        dp_sp = dp.reshape (nspec**2, -1, *rij.shape)
       
        return (p_sp, dp_sp)
    
    return compute
