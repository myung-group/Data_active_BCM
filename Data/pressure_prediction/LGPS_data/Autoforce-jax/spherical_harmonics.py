import jax
import jax.numpy as jnp
import sys 

@jax.jit
def sph_harm_to_cart(sin_theta, cos_theta,
                     sin_phi, cos_phi,
                     F_theta, F_phi):
    
    F_phi = jnp.nan_to_num(F_phi/sin_theta)

    F_x = cos_theta * cos_phi * F_theta - sin_phi * F_phi
    F_y = cos_theta * sin_phi * F_theta + cos_phi * F_phi
    F_z = - sin_theta * F_theta
    
    return jnp.stack([F_x, F_y, F_z], axis=-1)

def Ylm (lmax):

    Yoo = jnp.sqrt(1.0/(4.0*jnp.pi))
    alp_al =  2 * [[]] + [
            jnp.array(
                [
                    jnp.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
                    for m in range(l - 1)
                ]
            )[:, None]
            for l in range(2, lmax + 1)
        ]
    alp_bl = 2 * [[]] + [
            jnp.array(
                [
                    -jnp.sqrt(
                        ((l - 1.0) ** 2 - m * m) / (4 * (l - 1.0) ** 2 - 1)
                    )
                    for m in range(l - 1)
                ]
            )[:, None]
            for l in range(2, lmax + 1)
        ]

    alp_cl = [jnp.sqrt(2.0 * l + 1.0) for l in range(lmax + 1)]
    alp_dl = [[]] + [
            -jnp.sqrt(1.0 + 1.0 / (2.0 * l)) for l in range(1, lmax + 1)
        ]

    # indices: for traversing diagonals
    II = [[l + k for l in range(lmax - k + 1)] for k in range(lmax + 1)]
    JJ = [[l for l in range(lmax - k + 1)] for k in range(lmax + 1)]

    # l,m tables
    l_table = jnp.array(
            [
                [l for m in range(l)] + [m for m in range(l, lmax + 1)]
                for l in range(lmax + 1)
            ]
        )[:, :, None]
    m_table = jnp.zeros_like(l_table)
    for m in range(lmax + 1):
        m_table = m_table.at[II[m], JJ[m]].set(m)
        m_table = m_table.at[JJ[m], II[m]].set(m)
    

    # lower triangle indices
    one = jnp.ones( (lmax + 1, lmax + 1) )
    sign = (-jnp.tril(one) + jnp.eye(lmax+1) + jnp.triu(one))[..., None]

    # l,m related coeffs
    coef = jnp.sqrt((
            ((l_table - m_table) * (l_table + m_table) * (2.0 * l_table + 1.))
            / (2 * l_table - 1)
        )[1:, 1:])

    @jax.jit 
    def compute_ylm_rl (dij, phi, theta):

        _r = dij
        r2 = _r*_r
        sin_theta = jnp.sin(theta)
        cos_theta = jnp.cos(theta)
        sin_phi = jnp.sin(phi)
        cos_phi = jnp.cos(phi)

        r_sin_theta = _r*sin_theta
        r_cos_theta = _r*cos_theta 

        # Associated Legendre polynomials
        alp = [[jnp.full_like(sin_theta, Yoo)]]
        for l in range(1, lmax + 1):
            alp += [
                [
                    alp_al[l][m]
                    * (
                        r_cos_theta * alp[l - 1][m]
                        + r2 * alp_bl[l][m] * alp[l - 2][m]
                    )
                    for m in range(l - 1)
                ]
                + [alp_cl[l] * r_cos_theta * alp[l - 1][l - 1]]
                + [alp_dl[l] * r_sin_theta * alp[l - 1][l - 1]]
            ]

        # sin, cos of m*phi
        sin = [jnp.zeros_like(sin_phi), sin_phi]
        cos = [jnp.ones_like(cos_phi), cos_phi]
        for m in range(2, lmax + 1):
            s = sin_phi * cos[-1] + cos_phi * sin[-1]
            c = cos_phi * cos[-1] - sin_phi * sin[-1]
            sin += [s]
            cos += [c]


        # Spherical Harmonics
        nloc = sin_theta.shape[0]
        Yr = jnp.hstack(
            [
                jnp.vstack(
                    [(alp[l][m] * cos[m]).reshape(1, nloc) 
                     for m in range(l, -1, -1)]
                    + [jnp.zeros( (lmax - l, nloc))]
                ).reshape(lmax + 1, 1, nloc)
                for l in range(lmax + 1)
            ]
        )
        Yr = jnp.transpose (Yr, (1,0,2))
        Yi = jnp.hstack(
            [
                jnp.vstack(
                    [(alp[l][m] * sin[m]).reshape(1, nloc) 
                     for m in range(l, -1, -1)]
                    + [jnp.zeros( (lmax - l, nloc) )]
                ).reshape(lmax + 1, 1, nloc)
                for l in range(lmax + 1)
            ]
        )
        Ylm = Yr + Yi

        Y_theta = jnp.nan_to_num(cos_theta/sin_theta) * l_table * Ylm
        Y_theta = Y_theta.at[1:, 1:].add (-jnp.nan_to_num(_r * Ylm[:-1, :-1] * coef / sin_theta))
        Y_phi = jnp.transpose(Ylm,(1, 0, 2)) * sign * m_table 

        dYlm = sph_harm_to_cart(
                    sin_theta,
                    cos_theta,
                    sin_phi,
                    cos_phi,
                    Y_theta / _r,
                    Y_phi / _r
                )
        
        return Ylm, dYlm
    
    return compute_ylm_rl
