# Size of variable arrays:
sizeAlgebraic = 16
sizeStates = 4
sizeConstants = 32
import sys
from math import *
from numpy import *

from SALib.analyze import morris
from SALib.sample.morris import sample
from SALib.test_functions import Sobol_G, Ishigami
from SALib.util import read_param_file
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, \
                                sample_histograms
import matplotlib.pyplot as plt
import numpy as np


def createLegends():
    legend_states = [""] * sizeStates
    legend_rates = [""] * sizeStates
    legend_algebraic = [""] * sizeAlgebraic
    legend_voi = ""
    legend_constants = [""] * sizeConstants
    legend_voi = "time in component environment (ms)"
    legend_constants[0] = "epi in component environment (dimensionless)"
    legend_constants[1] = "endo in component environment (dimensionless)"
    legend_constants[2] = "mcell in component environment (dimensionless)"
    legend_states[0] = "u in component membrane (dimensionless)"
    legend_algebraic[0] = "Vm in component membrane (mV)"
    legend_constants[3] = "V_0 in component membrane (mV)"
    legend_constants[4] = "V_fi in component membrane (mV)"
    legend_algebraic[8] = "J_fi in component fast_inward_current (per_ms)" # Na
    legend_algebraic[14] = "J_so in component slow_outward_current (per_ms)" # K
    legend_algebraic[15] = "J_si in component slow_inward_current (per_ms)" # Ca
    legend_algebraic[1] = "J_stim in component membrane (per_ms)"
    legend_algebraic[2] = "m in component m (dimensionless)"
    legend_constants[5] = "u_m in component m (dimensionless)"
    legend_algebraic[4] = "p in component p (dimensionless)"
    legend_constants[6] = "u_p in component p (dimensionless)"
    legend_algebraic[3] = "q in component q (dimensionless)"
    legend_constants[13] = "u_q in component q (dimensionless)"
    legend_algebraic[5] = "r in component r (dimensionless)"
    legend_constants[15] = "u_r in component r (dimensionless)"
    legend_constants[16] = "tau_fi in component fast_inward_current (ms)"
    legend_constants[17] = "u_u in component fast_inward_current (dimensionless)"
    legend_states[1] = "v in component fast_inward_current_v_gate (dimensionless)"
    legend_algebraic[6] = "v_inf in component fast_inward_current_v_gate (dimensionless)"
    legend_algebraic[9] = "tau_v_minus in component fast_inward_current_v_gate (ms)"
    legend_constants[30] = "tau_v1_minus in component fast_inward_current_v_gate (ms)"
    legend_constants[12] = "tau_v2_minus in component fast_inward_current_v_gate (ms)"
    legend_constants[7] = "tau_v_plus in component fast_inward_current_v_gate (ms)"
    legend_algebraic[11] = "tau_o in component slow_outward_current (ms)"
    legend_constants[14] = "tau_o1 in component slow_outward_current (ms)"
    legend_constants[31] = "tau_o2 in component slow_outward_current (ms)"
    legend_algebraic[13] = "tau_so in component slow_outward_current (ms)"
    legend_constants[18] = "tau_so1 in component slow_outward_current (ms)"
    legend_constants[19] = "tau_so2 in component slow_outward_current (ms)"
    legend_constants[20] = "k_so in component slow_outward_current (dimensionless)"
    legend_constants[21] = "u_so in component slow_outward_current (dimensionless)"
    legend_constants[22] = "tau_si in component slow_inward_current (ms)"
    legend_states[2] = "w in component slow_inward_current_w_gate (dimensionless)"
    legend_states[3] = "s in component slow_inward_current_s_gate (dimensionless)"
    legend_algebraic[10] = "w_inf in component slow_inward_current_w_gate (dimensionless)"
    legend_constants[23] = "tau_winf in component slow_inward_current_w_gate (ms)"
    legend_constants[24] = "wstar_inf in component slow_inward_current_w_gate (dimensionless)"
    legend_algebraic[12] = "tau_w_minus in component slow_inward_current_w_gate (ms)"
    legend_constants[25] = "tau_w1_minus in component slow_inward_current_w_gate (ms)"
    legend_constants[26] = "tau_w2_minus in component slow_inward_current_w_gate (ms)"
    legend_constants[27] = "k_w_minus in component slow_inward_current_w_gate (dimensionless)"
    legend_constants[28] = "u_w_minus in component slow_inward_current_w_gate (dimensionless)"
    legend_constants[29] = "tau_w_plus in component slow_inward_current_w_gate (ms)"
    legend_algebraic[7] = "tau_s in component slow_inward_current_s_gate (ms)"
    legend_constants[8] = "tau_s1 in component slow_inward_current_s_gate (ms)"
    legend_constants[11] = "tau_s2 in component slow_inward_current_s_gate (ms)"
    legend_constants[9] = "k_s in component slow_inward_current_s_gate (dimensionless)"
    legend_constants[10] = "u_s in component slow_inward_current_s_gate (dimensionless)"
    legend_rates[0] = "d/dt u in component membrane (dimensionless)"
    legend_rates[1] = "d/dt v in component fast_inward_current_v_gate (dimensionless)"
    legend_rates[2] = "d/dt w in component slow_inward_current_w_gate (dimensionless)"
    legend_rates[3] = "d/dt s in component slow_inward_current_s_gate (dimensionless)"
    return (legend_states, legend_algebraic, legend_voi, legend_constants)

def initConsts():
    constants = [0.0] * sizeConstants; states = [0.0] * sizeStates;
    constants[0] = 1        # epi in component environment (dimensionless)
    constants[1] = 0        # endo in component environment (dimensionless)
    constants[2] = 0        # mcell in component environment (dimensionless)
    states[0] = 0           # u in component membrane (dimensionless)
    constants[3] = -83      # V_0 in component membrane (mV)
    constants[4] = 2.7      # V_fi in component membrane (mV)
    constants[5] = 0.3      # u_m in component m (dimensionless)
    constants[6] = 0.13     # u_p in component p (dimensionless)
    states[1] = 1           # v in component fast_inward_current_v_gate (dimensionless)
    constants[7] = 1.45     # tau_v_plus in component fast_inward_current_v_gate (ms)
    states[2] = 1           # w in component slow_inward_current_w_gate (dimensionless)
    states[3] = 0           # s in component slow_inward_current_s_gate (dimensionless)
    constants[8] = 2.7342   # tau_s1 in component slow_inward_current_s_gate (ms)
    constants[9] = 2.0994   # k_s in component slow_inward_current_s_gate (dimensionless)
    constants[10] = 0.9087  # u_s in component slow_inward_current_s_gate (dimensionless)
    constants[11] = custom_piecewise([equal(constants[0] , 1.00000), 16.0000 , equal(constants[1] , 1.00000), 2.00000 , True, 4.00000])
    # tau_s2 in component slow_inward_current_s_gate (ms)"
    constants[12] = custom_piecewise([equal(constants[0] , 1.00000), 1150.00 , equal(constants[1] , 1.00000), 10.0000 , True, 1.45000])
    # tau_v2_minus in component fast_inward_current_v_gate (ms)"
    constants[13] = custom_piecewise([equal(constants[0] , 1.00000), 0.00600000 , equal(constants[1] , 1.00000), 0.0240000 , True, 0.100000])
    # u_q in component q (dimensionless)"
    constants[14] = custom_piecewise([equal(constants[0] , 1.00000), 400.000 , equal(constants[1] , 1.00000), 470.000 , True, 410.000])
    # tau_o1 in component slow_outward_current (ms)"
    constants[15] = custom_piecewise([equal(constants[0] , 1.00000), 0.00600000 , equal(constants[1] , 1.00000), 0.00600000 , True, 0.00500000])
    # u_r in component r (dimensionless)"
    constants[16] = custom_piecewise([equal(constants[0] , 1.00000), 0.110000 , equal(constants[1] , 1.00000), 0.104000 , True, 0.0780000])
    # tau_fi in component fast_inward_current (ms)"
    constants[17] = custom_piecewise([equal(constants[0] , 1.00000), 1.55000 , equal(constants[1] , 1.00000), 1.56000 , True, 1.61000])
    # u_u in component fast_inward_current (dimensionless)"
    constants[18] = custom_piecewise([equal(constants[0] , 1.00000), 30.0200 , equal(constants[1] , 1.00000), 40.0000 , True, 91.0000])
    # tau_so1 in component slow_outward_current (ms)"
    constants[19] = custom_piecewise([equal(constants[0] , 1.00000), 0.996000 , equal(constants[1] , 1.00000), 1.20000 , True, 0.800000])
    # tau_so2 in component slow_outward_current (ms)"
    constants[20] = custom_piecewise([equal(constants[0] , 1.00000), 2.04600 , equal(constants[1] , 1.00000), 2.00000 , True, 2.10000])
    # k_so in component slow_outward_current (dimensionless)"
    constants[21] = custom_piecewise([equal(constants[0] , 1.00000), 0.650000 , equal(constants[1] , 1.00000), 0.650000 , True, 0.600000])
    # u_so in component slow_outward_current (dimensionless)"
    constants[22] = custom_piecewise([equal(constants[0] , 1.00000), 1.88750 , equal(constants[1] , 1.00000), 2.90130 , True, 3.38490])
    # tau_si in component slow_inward_current (ms)"
    constants[23] = custom_piecewise([equal(constants[0] , 1.00000), 0.0700000 , equal(constants[1] , 1.00000), 0.0273000 , True, 0.0100000])
    # tau_winf in component slow_inward_current_w_gate (ms)"
    constants[24] = custom_piecewise([equal(constants[0] , 1.00000), 0.940000 , equal(constants[1] , 1.00000), 0.780000 , True, 0.500000])
    # wstar_inf in component slow_inward_current_w_gate (dimensionless)"
    constants[25] = custom_piecewise([equal(constants[0] , 1.00000), 60.0000 , equal(constants[1] , 1.00000), 6.00000 , True, 70.0000])
    # tau_w1_minus in component slow_inward_current_w_gate (ms)"
    constants[26] = custom_piecewise([equal(constants[0] , 1.00000), 15.0000 , equal(constants[1] , 1.00000), 140.000 , True, 8.00000])
    # tau_w2_minus in component slow_inward_current_w_gate (ms)"
    constants[27] = custom_piecewise([equal(constants[0] , 1.00000), 65.0000 , equal(constants[1] , 1.00000), 200.000 , True, 200.000])
    # k_w_minus in component slow_inward_current_w_gate (dimensionless)"
    constants[28] = custom_piecewise([equal(constants[0] , 1.00000), 0.0300000 , equal(constants[1] , 1.00000), 0.0160000 , True, 0.0160000])
    # u_w_minus in component slow_inward_current_w_gate (dimensionless)"
    constants[29] = custom_piecewise([equal(constants[0] , 1.00000), 200.000 , equal(constants[1] , 1.00000), 280.000 , True, 280.000])
    # tau_w_plus in component slow_inward_current_w_gate (ms)"
    constants[30] = custom_piecewise([equal(constants[0] , 1.00000), 60.0000 , equal(constants[1] , 1.00000), 75.0000 , True, 80.0000])
    # tau_v1_minus in component fast_inward_current_v_gate (ms)"
    constants[31] = custom_piecewise([equal(constants[0] , 1.00000), 6.00000 , equal(constants[1] , 1.00000), 6.00000 , True, 7.00000])
    # tau_o2 in component slow_outward_current (ms)"
    return (states, constants)

def computeRates(voi, states, constants):
    rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
    algebraic[4] = custom_piecewise([less(states[0] , constants[6]), 0.00000 , True, 1.00000])
    algebraic[7] = (1.00000-algebraic[4])*constants[8]+algebraic[4]*constants[11]
    rates[3] = ((1.00000+tanh(constants[9]*(states[0]-constants[10])))/2.00000-states[3])/algebraic[7]
    algebraic[2] = custom_piecewise([less(states[0] , constants[5]), 0.00000 , True, 1.00000])
    algebraic[6] = custom_piecewise([less(states[0] , constants[13]), 1.00000 , True, 0.00000])
    algebraic[3] = custom_piecewise([less(states[0] , constants[13]), 0.00000 , True, 1.00000])
    algebraic[9] = algebraic[3]*constants[12]+(1.00000-algebraic[3])*constants[30]
    rates[1] = ((1.00000-algebraic[2])*(algebraic[6]-states[1]))/algebraic[9]-(algebraic[2]*states[1])/constants[7]
    algebraic[5] = custom_piecewise([less(states[0] , constants[15]), 0.00000 , True, 1.00000])
    algebraic[10] = (1.00000-algebraic[5])*(1.00000-(states[0]*1.00000)/constants[23])+algebraic[5]*constants[24]
    algebraic[12] = constants[25]+((constants[26]-constants[25])*(1.00000+tanh(constants[27]*(states[0]-constants[28]))))/2.00000
    rates[2] = ((1.00000-algebraic[5])*(algebraic[10]-states[2]))/algebraic[12]-(algebraic[5]*states[2])/constants[29]
    algebraic[8] = (-algebraic[2]*states[1]*(states[0]-constants[5])*(constants[17]-states[0]))/constants[16]
    algebraic[11] = (1.00000-algebraic[5])*constants[14]+algebraic[5]*constants[31]
    algebraic[13] = constants[18]+((constants[19]-constants[18])*(1.00000+tanh(constants[20]*(states[0]-constants[21]))))/2.00000
    algebraic[14] = (states[0]*(1.00000-algebraic[4]))/algebraic[11]+algebraic[4]/algebraic[13]
    algebraic[15] = (-algebraic[4]*states[2]*states[3])/constants[22]
    algebraic[1] = custom_piecewise([greater_equal(voi , 100.000) & less_equal(voi , 101.000), -1.00000 , True, 0.00000])
    rates[0] = -(algebraic[8]+algebraic[14]+algebraic[15]+algebraic[1])
    return(rates)

def computeAlgebraic(constants, states, voi):
    algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
    states = array(states)
    voi = array(voi)
    algebraic[4] = custom_piecewise([less(states[0] , constants[6]), 0.00000 , True, 1.00000])
    algebraic[7] = (1.00000-algebraic[4])*constants[8]+algebraic[4]*constants[11]
    algebraic[2] = custom_piecewise([less(states[0] , constants[5]), 0.00000 , True, 1.00000])
    algebraic[6] = custom_piecewise([less(states[0] , constants[13]), 1.00000 , True, 0.00000])
    algebraic[3] = custom_piecewise([less(states[0] , constants[13]), 0.00000 , True, 1.00000])
    algebraic[9] = algebraic[3]*constants[12]+(1.00000-algebraic[3])*constants[30]
    algebraic[5] = custom_piecewise([less(states[0] , constants[15]), 0.00000 , True, 1.00000])
    algebraic[10] = (1.00000-algebraic[5])*(1.00000-(states[0]*1.00000)/constants[23])+algebraic[5]*constants[24]
    algebraic[12] = constants[25]+((constants[26]-constants[25])*(1.00000+tanh(constants[27]*(states[0]-constants[28]))))/2.00000
    algebraic[8] = (-algebraic[2]*states[1]*(states[0]-constants[5])*(constants[17]-states[0]))/constants[16]
    algebraic[11] = (1.00000-algebraic[5])*constants[14]+algebraic[5]*constants[31]
    algebraic[13] = constants[18]+((constants[19]-constants[18])*(1.00000+tanh(constants[20]*(states[0]-constants[21]))))/2.00000
    algebraic[14] = (states[0]*(1.00000-algebraic[4]))/algebraic[11]+algebraic[4]/algebraic[13]
    algebraic[15] = (-algebraic[4]*states[2]*states[3])/constants[22]
    algebraic[1] = custom_piecewise([greater_equal(voi , 100.000) & less_equal(voi , 101.000), -1.00000 , True, 0.00000])
    algebraic[0] = constants[3]+states[0]*(constants[4]-constants[3])
    #legend_algebraic[8] = "J_fi in component fast_inward_current (per_ms)" # Na
    #legend_algebraic[14] = "J_so in component slow_outward_current (per_ms)" # K
    #legend_algebraic[15] = "J_si in component slow_inward_current (per_ms)" # Ca
#     Na = algebraic[8]
#     K = algebraic[14]
#     Ca = algebraic[15]
#     vMax = max(algebraic[0])
#     vMin = min(algebraic[0])
#     indexV = np.where(algebraic[0]==np.max(algebraic[0]))
#     np.set_printoptions(precision=14)
#     #indexV.astype(np.float64)
#     print(indexV)
#     print(algebraic[0][121])
#     print(algebraic[0][indexV][0])
#     print(voi[0])
#     print(voi[indexV])
    getAmpPerc(voi, algebraic, 0.9)
    return algebraic

def getAmp(voi,algebraic):    
    idxVMax = np.where(algebraic[0]==np.max(algebraic[0]))[0][0] 
    idxVMin = np.where(algebraic[0]==np.min(algebraic[0]))[0][0] 
    np.set_printoptions(precision=14)
    vMax = algebraic[0][idxVMax]
    vMin = algebraic[0][idxVMin]
    vAmp = vMax - vMin
    tStart = voi[0]
    tEnd = voi[idxVMax]
    tGap = tEnd - tStart
    print('vMax = ',vMax, 'vMin = ',vMin, 'vAmp = ',vAmp, 'tGap = ',tGap)
    return vAmp, vMax, vMin, idxVMax, tGap

def getAmpPerc(voi,algebraic,percentage): 
    vAmp, vMax, vMin, idxVMax, tGap = getAmp(voi, algebraic)   
    vAmpPerc = vAmp * (1 - percentage) + vMin
    vAmpLeft = algebraic[0][0:idxVMax]
    vAmpRight = algebraic[0][idxVMax+1:]
    idxVStart = (np.abs(vAmpLeft-vAmpPerc)).argmin()
    idxVEnd = (np.abs(vAmpRight-vAmpPerc)).argmin()
    tStart = voi[idxVStart]
    tEnd = voi[idxVEnd]
    tGap = tEnd - tStart
    #print('vStart-1',algebraic[0][idxVMax-1])
    print('vStart',vAmpLeft[idxVStart], 'vEnd',vAmpRight[idxVEnd], 'tStart', tStart, 'tEnd', tEnd)
    #print('vStart+1',algebraic[0][idxVMax+1])   
    #print('vEnd-1',vAmpRight[idxVMin-1])
    
    return vAmp,tGap

def custom_piecewise(cases):
    """Compute result of a piecewise function"""
    return select(cases[0::2],cases[1::2])

def solve_model():
    """Solve model with ODE solver"""
    from scipy.integrate import ode
    # Initialise constants and state variables
    (init_states, constants) = initConsts()

    # Set timespan to solve over
    voi = linspace(0, 1000, 1200)

    # Construct ODE object to solve
    r = ode(computeRates)
    r.set_integrator('vode', method='bdf', atol=1e-006, rtol=1e-006, max_step=0.1)
    r.set_initial_value(init_states, voi[0])
    r.set_f_params(constants)

    # Solve model
    states = array([[0.0] * len(voi)] * sizeStates)
    states[:,0] = init_states
    for (i,t) in enumerate(voi[1:]):
        if r.successful():
            r.integrate(t)
            states[:,i+1] = r.y
        else:
            break

    # Compute algebraic variables
    algebraic = computeAlgebraic(constants, states, voi)
    return (voi, states, algebraic)

def plot_model(voi, states, algebraic):
    """Plot variables against variable of integration"""
    import pylab
    (legend_states, legend_algebraic, legend_voi, legend_constants) = createLegends()
    pylab.figure(1)
    pylab.plot(voi,vstack((states,algebraic)).T)
    pylab.xlabel(legend_voi)
    pylab.legend(legend_states + legend_algebraic, loc='best')
    pylab.show()

if __name__ == "__main__":
    (voi, states, algebraic) = solve_model()
    plot_model(voi, states, algebraic)
