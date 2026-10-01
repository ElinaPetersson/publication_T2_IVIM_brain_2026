""" Functions to generate MR signal and corresponding Jacobians based on IVIM parameters. 
"""

import numpy as np
import numpy.typing as npt

SIVIM_REGIME = 'sIVIM'


def monoexp(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the monoexponential e^(-b*D).
    
    Arguments:
        b: vector of b-values [s/mm2]
        D: ND array of diffusion coefficients [mm2/s]

    Output:
        S: (N+1)D array of signal values
    """
    [b, D] = at_least_1d([b, D])
    if TE is None or T2d is None:
        S = np.exp(-np.outer(D, b))
        return np.reshape(S, list(D.shape) + [b.size]) # reshape as np.outer flattens D if ndim > 1
    else:
        [TE, T2d] = at_least_1d([TE, T2d])
        b = at_lest_right_dim_b(b)
        TE = at_lest_right_dim_te(TE)
        [D,T2d] = at_lest_right_dim_tissue([D,T2d])
        return np.exp(-D*b) * np.exp(-TE/T2d) # [N,nb]

def kurtosis(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], K: npt.NDArray[np.float64], TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the kurtosis signal representation.
    
    Arguments: 
        b: vector of b-values [s/mm2]
        D: ND array of diffusion coefficients [mm2/s]
        K: ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S: (N+1)D array of signal values
    """        
    [b, D, K] = at_least_1d([b, D, K])
    if TE is None or T2d is None:
        Slin = monoexp(b, D)
        Squad = np.exp(np.reshape(np.outer(D, b)**2, list(D.shape) + [b.size]) * K[..., np.newaxis]/6)
        return Slin * Squad
    else:
        [TE, T2d] = at_least_1d([TE, T2d])
        b = at_lest_right_dim_b(b)
        TE = at_lest_right_dim_te(TE)
        [D,K,T2d] = at_lest_right_dim_tissue([D,K,T2d])
        Slin = monoexp(b, D, TE, T2d)
        Squad = np.exp((D*b)**2*K/6)
        return Slin * Squad

def sIVIM(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] = 1, K: npt.NDArray[np.float64] = 0, TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None, T2p: npt.NDArray[np.float64] | None = None, Covterm: bool = False, H: npt.NDArray[np.float64] | None = None ) -> npt.NDArray[np.float64]:
    """
    Return MR signal based on the simplified IVIM (sIVIM) model.
    
    Arguments: 
        b:  vector of b-values [s/mm2]
        D:  ND array of diffusion coefficients [mm2/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        S:  (N+1)D array of signal values
    """
    [b, D, f, S0] = at_least_1d([b, D, f, S0])
    if TE is None or T2d is None:
        return S0[..., np.newaxis] * ((1-f[..., np.newaxis]) * kurtosis(b, D, K) + np.reshape(np.outer(f, b==0), list(f.shape) + [b.size]))
    else:
        [TE, T2d] = at_least_1d([TE, T2d])
        b = at_lest_right_dim_b(b)
        TE = at_lest_right_dim_te(TE)
        [D,f,K,T2d,T2p,S0] = at_lest_right_dim_tissue([D,f,K,T2d,T2p,S0])
        if not Covterm:
            return S0*((1-f) * kurtosis(b,D,K,TE,T2d) + f * np.exp(-TE/T2p)*(b==0))
        else:
            [H] = at_lest_right_dim_tissue([H])
            return S0*((1-f) * kurtosis(b,D,K,TE,T2d) * np.exp(-TE*b*H*D/T2d) + f * np.exp(-TE/T2p)*(b==0))


def monoexp_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """ 
    Return the Jacobian matrix for the monoexponential expression.
    
    S(b) = exp(-b*D)

    Arguments:
        b: vector of b-values [s/mm2]
        D: ND array of diffusion coefficients [mm2/s]

    Output: 
        J: Jacobian matrix
    """
    if TE is None or T2d is None:
        # warning! alternative to b[np.newaxis,:] may be needed
        J = (monoexp(b, D) * -b[np.newaxis, :])[...,np.newaxis] # D is the only parameter, but we still want the last dimension
        return J
    else:
        b = at_lest_right_dim_b(b)
        TE = at_lest_right_dim_te(TE)
        [D,T2d] = at_lest_right_dim_tissue([D,T2d])
        dSdD = monoexp(b,D,TE,T2d)*-b # [N,nb]*[1,nb]
        dSdT2 = monoexp(b,D,TE,T2d)*TE/T2d**2 # [N,nb]*[1,nte] and nb shoud be the same as nte since b contains repeats
        return [dSdD, dSdT2]

def kurtosis_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], K: npt.NDArray[np.float64], TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """ 
    Return the Jacobian matrix for the kurtosis expression.

    S(b) = exp(-b*D + b**2*D**2*K/6)

    Arguments:
        b: vector of b-values [s/mm2]
        D: ND array of diffusion coefficients [mm2/s]
        K: ND array of kurtosis coefficients (same shape as D or scalar)

    Output: 
        J: Jacobian matrix
    """

    [b,D,K] = at_least_1d([b,D,K])
    if TE is None or T2d is None:
        J = np.stack([
                    kurtosis(b,D,K)*(-b[np.newaxis, :]+2*np.reshape(np.outer(D*K,b**2)/6,list(D.shape) + [b.size])),
                    kurtosis(b,D,K)*np.reshape(np.outer(D, b)**2/6, list(D.shape) + [b.size])
                    ], axis=-1)
        return J
    else:
        b = at_lest_right_dim_b(b)
        [D,K,T2d] = at_lest_right_dim_tissue([D,K,T2d])
        dSdD = kurtosis(b,D,K,TE,T2d)*(2*D*b**2*K/6-b) 
        dSdK = kurtosis(b,D,K,TE,T2d)*(b*D)**2/6
        dSdT2 = kurtosis(b,D,K,TE,T2d)*TE/T2d**2
        return [dSdD, dSdK, dSdT2]

def sIVIM_jacobian(b: npt.NDArray[np.float64], D: npt.NDArray[np.float64], f: npt.NDArray[np.float64], S0: npt.NDArray[np.float64] | None = None, K: npt.NDArray[np.float64] | None = None, TE: npt.NDArray[np.float64] | None = None, T2d: npt.NDArray[np.float64] | None = None, T2p: npt.NDArray[np.float64] | None = None,  Covterm: bool = False, H: npt.NDArray[np.float64] | None = None) -> npt.NDArray[np.float64]:
    """
    Return the Jacobian matrix for the simplified IVIM (sIVIM) model.
    
    S(b) = S0((1-f)*exp(-b*D+b^2*D^2*K/6)+fδ(b))

    Arguments: 
        b:  vector of b-values [s/mm2]
        D:  ND array of diffusion coefficients [mm2/s]
        f:  ND array of perfusion fractions (same shape as D or scalar)
        S0: (optional) ND array of signal values at b == 0 (same shape as D or scalar)
        K:  (optional) ND array of kurtosis coefficients (same shape as D or scalar)

    Output:
        J:  Jacobian matrix
    """

    [b, D, f] = at_least_1d([b, D, f])

    if TE is None or T2d is None or T2p is None:
        if K is None:
            dSdD = (1-f)[..., np.newaxis] * monoexp_jacobian(b,D)[..., 0]
            dSdf = -monoexp(b,D) + (b==0)[np.newaxis, :]
        else:
            [K] = at_least_1d([K])
            dSdD = (1-f)[..., np.newaxis] * kurtosis_jacobian(b,D,K)[..., 0]
            dSdf = -kurtosis(b, D, K) + (b==0)[np.newaxis, :] 
            dSdK = (1-f)[..., np.newaxis] * kurtosis_jacobian(b,D,K)[..., 1]

        if S0 is None:
            if K is None:
                J_list = [dSdD, dSdf]
            else:
                J_list = [dSdD, dSdf, dSdK]
        else:
            [S0] = at_least_1d([S0])
            if K is None:
                dSdS0 = sIVIM(b, D, f)
            else:
                dSdS0 = sIVIM(b, D, f, K=K)
            dSdD *= S0[..., np.newaxis]
            dSdf *= S0[..., np.newaxis]
            if K is None:
                J_list = [dSdD, dSdf, dSdS0]
            else:
                J_list = [dSdD, dSdf, dSdS0, dSdK * S0[..., np.newaxis]]

        J = np.stack(J_list, axis=-1)
        
        return J
    else: # [x*y*z,nb,nte,np]
        b = at_lest_right_dim_b(b)
        TE = at_lest_right_dim_te(TE)
        [D,f,T2d,T2p] = at_lest_right_dim_tissue([D,f,T2d,T2p])
        if not Covterm:
            if K is None:
                dSdD = (1-f) * monoexp_jacobian(b,D,TE,T2d)[0]
                dSdf = -monoexp(b,D,TE,T2d) + np.exp(-TE/T2p) * (b==0)
                dSdT2d = (1-f) * monoexp_jacobian(b,D,TE,T2d)[1]
                dSdT2p = f * TE/T2p**2 * np.exp(-TE/T2p) * (b==0)
                if S0 is None:
                    J_list = [dSdD,dSdf,dSdT2d,dSdT2p]
                else:
                    [S0] = at_least_1d([S0])
                    [S0] = at_lest_right_dim_tissue([S0])
                    dSdD *= S0
                    dSdf *= S0
                    dSdT2d *= S0
                    dSdT2p *= S0 
                    dSdS0 = sIVIM(b, D, f,TE=TE,T2d=T2d,T2p=T2p)
                    J_list = [dSdD,dSdf,dSdS0,dSdT2d,dSdT2p]
            else:
                [K] = at_least_1d([K])
                [K] = at_lest_right_dim_tissue([K])
                dSdD = (1-f) * kurtosis_jacobian(b,D,K,TE,T2d)[0]
                dSdf = -kurtosis(b, D, K,TE,T2d) + np.exp(-TE/T2p)*(b==0)
                dSdK = (1-f) * kurtosis_jacobian(b,D,K,TE,T2d)[1]
                dSdT2d = (1-f) * kurtosis_jacobian(b,D,K,TE,T2d)[2]
                dSdT2p = f * TE/T2p**2 * np.exp(-TE/T2p) * (b==0)
                if S0 is None:
                    J_list = [dSdD,dSdf,dSdK,dSdT2d,dSdT2p]
                else:
                    [S0] = at_least_1d([S0])
                    [S0] = at_lest_right_dim_tissue([S0])
                    dSdD *= S0
                    dSdf *= S0
                    dSdK *= S0
                    dSdT2d *= S0
                    dSdT2p *= S0 
                    dSdS0 = sIVIM(b, D, f,K=K,TE=TE,T2d=T2d,T2p=T2p)
                    J_list = [dSdD,dSdf,dSdS0,dSdK,dSdT2d,dSdT2p]
        elif Covterm and H is not None:
           raise NotImplementedError('Covterm not implemented yet')
        J = np.stack(J_list, axis=-1)
        return J

def at_lest_right_dim_tissue(pars: list) -> list:
    """ Makes sure that each tissue parameter has the correct dimension: [N,1] """
    for i, par in enumerate(pars):
        pars[i] = np.array(par).ravel()[...,np.newaxis]
    return pars

def at_lest_right_dim_b(b: npt.NDArray) -> npt.NDArray:
    """ Makes sure that b-vector has the correct dimension: [1,nb*nte] """
    b = b.ravel()[np.newaxis,...]
    return b

def at_lest_right_dim_te(te: npt.NDArray) -> npt.NDArray:
    """ Makes sure that te vector has the correct dimension: [1,nte*nb] """
    te = te.ravel()[np.newaxis,...]
    return te

def at_least_1d(pars: list) -> list:
    """ Check that each parameter is atleast one dimension in shape. """
    for i, par in enumerate(pars):
        pars[i] = np.atleast_1d(par)
    return pars


'''
Test of the jacobian calculation for the sIVIM model. 
The Jacobian is calculated using finite differences and compared to the analytical Jacobian. 
The test is performed for a range of parameters and b-values. The test passes if the two Jacobians are close within a specified tolerance.
'''
# from OPT_44_models_T2 import sIVIM,sIVIM_jacobian
# import numpy as np
# import matplotlib.pyplot as plt
# from OPT_02_opt_input_pars import usr_input

# b = np.array([0,10,100,500,1000])
# TE = np.array([0.05])
# rtol = 1e-6
# atol_models = 1e-8
# atol_jac = 1e-4

# f_ = np.stack([np.stack([np.linspace(0.01,0.03,5) for _ in range(5)]).T for _ in range(5)]).ravel()
# D_ = np.ones_like(f_)*0.8e-3
# K_ = np.stack([np.stack([np.linspace(0.5,1.5,5) for _ in range(5)]) for _ in range(5)]).ravel()
# T2d_ = np.ones_like(f_)*usr_input['T2d']
# T2p_ = np.ones_like(f_)*usr_input['T2p']
# S0_ = np.stack([np.stack([np.linspace(0.5,1.5,5) for _ in range(5)]) for _ in range(5)]).ravel()

# for D, f, S0, K, T2d, T2p in zip(D_, f_, S0_, K_, T2d_, T2p_):
#     y = sIVIM(b,D,f,S0,K,TE,T2d,T2p)
#     Jlist = [(sIVIM(b,D*(1+rtol),f,S0,K,TE,T2d,T2p) - y) / (np.atleast_1d(D)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f*(1+rtol),S0,K,TE,T2d,T2p) - y) / (np.atleast_1d(f)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0*(1+rtol),K,TE,T2d,T2p) - y) / (np.atleast_1d(S0)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0,K*(1+rtol),TE,T2d,T2p) - y) / (np.atleast_1d(K)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0,K,TE,T2d*(1+rtol),T2p) - y) / (np.atleast_1d(T2d)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0,K,TE,T2d,T2p*(1+rtol)) - y) / (np.atleast_1d(T2p)[..., np.newaxis]*rtol)]

#     Japp = np.stack(Jlist, axis = -1)
#     Jmy = sIVIM_jacobian(b,D,f,S0,K,TE,T2d,T2p)
#     np.testing.assert_allclose(Japp, Jmy, rtol, atol_jac)

# # Med K=0
# for D, f, S0, K, T2d, T2p in zip(D_, f_, S0_, K_*0, T2d_, T2p_):
#     y = sIVIM(b,D,f,S0,K,TE,T2d,T2p)
#     Jlist = [(sIVIM(b,D*(1+rtol),f,S0,K,TE,T2d,T2p) - y) / (np.atleast_1d(D)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f*(1+rtol),S0,K,TE,T2d,T2p) - y) / (np.atleast_1d(f)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0*(1+rtol),K,TE,T2d,T2p) - y) / (np.atleast_1d(S0)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0,K,TE,T2d*(1+rtol),T2p) - y) / (np.atleast_1d(T2d)[..., np.newaxis]*rtol),
#              (sIVIM(b,D,f,S0,K,TE,T2d,T2p*(1+rtol)) - y) / (np.atleast_1d(T2p)[..., np.newaxis]*rtol)]
#     Japp = np.stack(Jlist, axis = -1)
#     Jmy = sIVIM_jacobian(b,D,f,S0,None,TE,T2d,T2p)
#     np.testing.assert_allclose(Japp, Jmy, rtol, atol_jac)

'''
No error so the jacobian calculation is correct.
'''