# Size of variable arrays:
sizeAlgebraic = 16
sizeStates = 4
sizeConstants = 32
sizeInputs = 30
import sys, getopt
from math import *
from numpy import *
import random
import pylab
import matplotlib.pyplot as plt


class Bueno:

    def createLegends(self):
        legend_states = [""] * 0
        legend_rates = [""] * sizeStates
        legend_algebraic = [""] * 1
        legend_voi = "time in component environment (ms)"
        legend_constants = [""] * sizeConstants        
        legend_algebraic[0] = "Vm in component membrane (mV)"        
        return (legend_states, legend_algebraic, legend_voi, legend_constants)
        
    def initConsts(self):
        constants = [0.0] * sizeConstants; states = [0.0] * sizeStates; 
        constants[0] = 1  # epi in component environment (dimensionless)
        constants[1] = 0  # endo in component environment (dimensionless)
        constants[2] = 0  # mcell in component environment (dimensionless)
        states[0] = 0  # u in component membrane (dimensionless)
        constants[3] = -83  # V_0 in component membrane (mV)
        constants[4] = 2.7  # V_fi in component membrane (mV)
        constants[5] = 0.3  # u_m in component m (dimensionless)
        constants[6] = 0.13  # u_p in component p (dimensionless)
        states[1] = 1  # v in component fast_inward_current_v_gate (dimensionless)
        constants[7] = 1.45  # tau_v_plus in component fast_inward_current_v_gate (ms)
        states[2] = 1  # w in component slow_inward_current_w_gate (dimensionless)
        states[3] = 0  # s in component slow_inward_current_s_gate (dimensionless)
        constants[8] = 2.7342  # tau_s1 in component slow_inward_current_s_gate (ms)
        constants[9] = 2.0994  # k_s in component slow_inward_current_s_gate (dimensionless)
        constants[10] = 0.9087  # u_s in component slow_inward_current_s_gate (dimensionless)
        constants[11] = self.custom_piecewise([equal(constants[0] , 1.00000), 16.0000 , equal(constants[1] , 1.00000), 2.00000 , True, 4.00000])
        # tau_s2 in component slow_inward_current_s_gate (ms)"
        constants[12] = self.custom_piecewise([equal(constants[0] , 1.00000), 1150.00 , equal(constants[1] , 1.00000), 10.0000 , True, 1.45000])
        # tau_v2_minus in component fast_inward_current_v_gate (ms)"
        constants[13] = self.custom_piecewise([equal(constants[0] , 1.00000), 0.00600000 , equal(constants[1] , 1.00000), 0.0240000 , True, 0.100000])
        # u_q in component q (dimensionless)"
        constants[14] = self.custom_piecewise([equal(constants[0] , 1.00000), 400.000 , equal(constants[1] , 1.00000), 470.000 , True, 410.000])
        # tau_o1 in component slow_outward_current (ms)"
        constants[15] = self.custom_piecewise([equal(constants[0] , 1.00000), 0.00600000 , equal(constants[1] , 1.00000), 0.00600000 , True, 0.00500000])
        # u_r in component r (dimensionless)"
        constants[16] = self.custom_piecewise([equal(constants[0] , 1.00000), 0.110000 , equal(constants[1] , 1.00000), 0.104000 , True, 0.0780000])
        # tau_fi in component fast_inward_current (ms)"
        constants[17] = self.custom_piecewise([equal(constants[0] , 1.00000), 1.55000 , equal(constants[1] , 1.00000), 1.56000 , True, 1.61000])
        # u_u in component fast_inward_current (dimensionless)"
        constants[18] = self.custom_piecewise([equal(constants[0] , 1.00000), 30.0200 , equal(constants[1] , 1.00000), 40.0000 , True, 91.0000])
        # tau_so1 in component slow_outward_current (ms)"
        constants[19] = self.custom_piecewise([equal(constants[0] , 1.00000), 0.996000 , equal(constants[1] , 1.00000), 1.20000 , True, 0.800000])
        # tau_so2 in component slow_outward_current (ms)"
        constants[20] = self.custom_piecewise([equal(constants[0] , 1.00000), 2.04600 , equal(constants[1] , 1.00000), 2.00000 , True, 2.10000])
        # k_so in component slow_outward_current (dimensionless)"
        constants[21] = self.custom_piecewise([equal(constants[0] , 1.00000), 0.650000 , equal(constants[1] , 1.00000), 0.650000 , True, 0.600000])
        # u_so in component slow_outward_current (dimensionless)"
        constants[22] = self.custom_piecewise([equal(constants[0] , 1.00000), 1.88750 , equal(constants[1] , 1.00000), 2.90130 , True, 3.38490])
        # tau_si in component slow_inward_current (ms)"
        constants[23] = self.custom_piecewise([equal(constants[0] , 1.00000), 0.0700000 , equal(constants[1] , 1.00000), 0.0273000 , True, 0.0100000])
        # tau_winf in component slow_inward_current_w_gate (ms)"
        constants[24] = self.custom_piecewise([equal(constants[0] , 1.00000), 0.940000 , equal(constants[1] , 1.00000), 0.780000 , True, 0.500000])
        # wstar_inf in component slow_inward_current_w_gate (dimensionless)"
        constants[25] = self.custom_piecewise([equal(constants[0] , 1.00000), 60.0000 , equal(constants[1] , 1.00000), 6.00000 , True, 70.0000])
        # tau_w1_minus in component slow_inward_current_w_gate (ms)"
        constants[26] = self.custom_piecewise([equal(constants[0] , 1.00000), 15.0000 , equal(constants[1] , 1.00000), 140.000 , True, 8.00000])
        # tau_w2_minus in component slow_inward_current_w_gate (ms)"
        constants[27] = self.custom_piecewise([equal(constants[0] , 1.00000), 65.0000 , equal(constants[1] , 1.00000), 200.000 , True, 200.000])
        # k_w_minus in component slow_inward_current_w_gate (dimensionless)"
        constants[28] = self.custom_piecewise([equal(constants[0] , 1.00000), 0.0300000 , equal(constants[1] , 1.00000), 0.0160000 , True, 0.0160000])
        # u_w_minus in component slow_inward_current_w_gate (dimensionless)"
        constants[29] = self.custom_piecewise([equal(constants[0] , 1.00000), 200.000 , equal(constants[1] , 1.00000), 280.000 , True, 280.000])
        # tau_w_plus in component slow_inward_current_w_gate (ms)"
        constants[30] = self.custom_piecewise([equal(constants[0] , 1.00000), 60.0000 , equal(constants[1] , 1.00000), 75.0000 , True, 80.0000])
        # tau_v1_minus in component fast_inward_current_v_gate (ms)"
        constants[31] = self.custom_piecewise([equal(constants[0] , 1.00000), 6.00000 , equal(constants[1] , 1.00000), 6.00000 , True, 7.00000])
        # tau_o2 in component slow_outward_current (ms)"
        return (states, constants)
        
    def initTitles(self, percentage):
        title = [''] * 10
        title[1] = 'voltage at the starting point at the percentage of ' + str(percentage * 100) + '%'
        title[2] = 'voltage at the end point at the percentage of ' + str(percentage * 100) + '%'
        title[3] = 'time at the starting point at the percentage of ' + str(percentage * 100) + '%'
        title[4] = 'time at the end point at the percentage of ' + str(percentage * 100) + '%'
        title[5] = 'time duration at the percentage of ' + str(percentage * 100) + '%'
        title[6] = 'amplitude voltage'
        title[7] = 'maximum voltage'
        title[8] = 'minimum voltage'
        title[9] = 'time duration'        
        return title
    
    def initLabels(self):
        x_labels = [''] * sizeInputs
        x_labels[0] = "Vm"  # in component membrane (mV)"
        x_labels[1] = "V_0"  # in component membrane (mV)"
        x_labels[2] = "V_fi"  # in component membrane (mV)"
        x_labels[3] = "u_m"  # in component m (dimensionless)"
        x_labels[4] = "u_p"  # in component p (dimensionless)"
        x_labels[5] = "tau_v_plus"  # in component fast_inward_current_v_gate (ms)"
        x_labels[6] = "tau_s1"  # in component slow_inward_current_s_gate (ms)"
        x_labels[7] = "k_s"  # in component slow_inward_current_s_gate (dimensionless)"
        x_labels[8] = "u_s"  # in component slow_inward_current_s_gate (dimensionless)"
        x_labels[9] = "tau_s2"  # in component slow_inward_current_s_gate (ms)"
        x_labels[10] = "tau_v2_minus"  # in component fast_inward_current_v_gate (ms)"
        x_labels[11] = "u_q"  # in component q (dimensionless)"
        x_labels[12] = "tau_o1"  # in component slow_outward_current (ms)"
        x_labels[13] = "u_r"  # in component r (dimensionless)"
        x_labels[14] = "tau_fi"  # in component fast_inward_current (ms)"
        x_labels[15] = "u_u"  # in component fast_inward_current (dimensionless)"
        x_labels[16] = "tau_so1"  # in component slow_outward_current (ms)"
        x_labels[17] = "tau_so2"  # in component slow_outward_current (ms)"
        x_labels[18] = "k_so"  # in component slow_outward_current (dimensionless)"
        x_labels[19] = "u_so"  # in component slow_outward_current (dimensionless)"
        x_labels[20] = "tau_si"  # in component slow_inward_current (ms)"
        x_labels[21] = "tau_winf"  # in component slow_inward_current_w_gate (ms)"
        x_labels[22] = "wstar_inf"  # in component slow_inward_current_w_gate (dimensionless)"
        x_labels[23] = "tau_w1_minus"  # in component slow_inward_current_w_gate (ms)"
        x_labels[24] = "tau_w2_minus"  # in component slow_inward_current_w_gate (ms)"
        x_labels[25] = "k_w_minus"  # in component slow_inward_current_w_gate (dimensionless)"
        x_labels[26] = "u_w_minus"  # in component slow_inward_current_w_gate (dimensionless)"
        x_labels[27] = "tau_w_plus"  # in component slow_inward_current_w_gate (ms)"
        x_labels[28] = "tau_v1_minus"  # in component fast_inward_current_v_gate (ms)"
        x_labels[29] = "tau_o2"  # in component slow_outward_current (ms)"
        return x_labels
    
    def computeRates(self, voi, states, constants):
        rates = [0.0] * sizeStates; algebraic = [0.0] * sizeAlgebraic
        algebraic[4] = self.custom_piecewise([less(states[0] , constants[6]), 0.00000 , True, 1.00000])
        algebraic[7] = (1.00000 - algebraic[4]) * constants[8] + algebraic[4] * constants[11]
        rates[3] = ((1.00000 + tanh(constants[9] * (states[0] - constants[10]))) / 2.00000 - states[3]) / algebraic[7]
        algebraic[2] = self.custom_piecewise([less(states[0] , constants[5]), 0.00000 , True, 1.00000])
        algebraic[6] = self.custom_piecewise([less(states[0] , constants[13]), 1.00000 , True, 0.00000])
        algebraic[3] = self.custom_piecewise([less(states[0] , constants[13]), 0.00000 , True, 1.00000])
        algebraic[9] = algebraic[3] * constants[12] + (1.00000 - algebraic[3]) * constants[30]
        rates[1] = ((1.00000 - algebraic[2]) * (algebraic[6] - states[1])) / algebraic[9] - (algebraic[2] * states[1]) / constants[7]
        algebraic[5] = self.custom_piecewise([less(states[0] , constants[15]), 0.00000 , True, 1.00000])
        algebraic[10] = (1.00000 - algebraic[5]) * (1.00000 - (states[0] * 1.00000) / constants[23]) + algebraic[5] * constants[24]
        algebraic[12] = constants[25] + ((constants[26] - constants[25]) * (1.00000 + tanh(constants[27] * (states[0] - constants[28])))) / 2.00000
        rates[2] = ((1.00000 - algebraic[5]) * (algebraic[10] - states[2])) / algebraic[12] - (algebraic[5] * states[2]) / constants[29]
        algebraic[8] = (-algebraic[2] * states[1] * (states[0] - constants[5]) * (constants[17] - states[0])) / constants[16]
        algebraic[11] = (1.00000 - algebraic[5]) * constants[14] + algebraic[5] * constants[31]
        algebraic[13] = constants[18] + ((constants[19] - constants[18]) * (1.00000 + tanh(constants[20] * (states[0] - constants[21])))) / 2.00000
        algebraic[14] = (states[0] * (1.00000 - algebraic[4])) / algebraic[11] + algebraic[4] / algebraic[13]
        algebraic[15] = (-algebraic[4] * states[2] * states[3]) / constants[22]
        algebraic[1] = self.custom_piecewise([greater_equal(voi , 100.000) & less_equal(voi , 101.000), -1.00000 , True, 0.00000])
        rates[0] = -(algebraic[8] + algebraic[14] + algebraic[15] + algebraic[1])
        return(rates)
        
    def computeAlgebraic(self, constants, states, voi):
        algebraic = array([[0.0] * len(voi)] * sizeAlgebraic)
        states = array(states)
        voi = array(voi)        
        algebraic[0] = constants[3] + states[0] * (constants[4] - constants[3])
        return algebraic
        
    def getAmp(self, voi, algebraic):
        """Calculate amplitude of voltage"""
        idxVMax = where(algebraic[0] == max(algebraic[0]))[0][0]  # Find the index of maximum value of voltage
        idxVMin = where(algebraic[0] == min(algebraic[0]))[0][0]  # Find the index of minimum value of voltage
        set_printoptions(precision=14)
        vMax = algebraic[0][idxVMax]  # Get the maximum voltage
        vMin = algebraic[0][idxVMin]  # Get the minimum voltage
        vAmp = vMax - vMin  # Calculate the amplitude value
        tStart = voi[0]  # Get the time starting from 0
        tEnd = voi[idxVMax]  # Get the time ending at the maximum voltage
        tGap = tEnd - tStart  # Calculate the duration from the start to the end
        # print('vMax = ',vMax, 'vMin = ',vMin, 'vAmp = ',vAmp, 'tGap = ',tGap)
        return vAmp, vMax, vMin, idxVMax, tGap
        
    def getAmpPerc(self, voi, algebraic, percentage): 
        """Calculate amplitude of voltage by different percentages"""
        vAmp, vMax, vMin, idxVMax, tGapAmp = self.getAmp(voi, algebraic)  # Get the amplitude of voltage
        vAmpPerc = vAmp * (1 - percentage) + vMin  # Calculate the value of voltage by percentage
        vAmpLeft = algebraic[0][0:idxVMax]  # Split the voltage array into a left and a right from the maximum point 
        vAmpRight = algebraic[0][idxVMax + 1:]
        idxVStart = (abs(vAmpLeft - vAmpPerc)).argmin()  # Calculate which value is closest to the percentage of the voltage and 
        idxVEnd = (abs(vAmpRight - vAmpPerc)).argmin()  # Return the two indexes of the two voltages in the duration
        tStart = voi[idxVStart]  # Calculate the time starting at the first value of voltage
        tEnd = voi[idxVEnd + idxVMax + 1]  # Calculate the time ending at the second value of voltage
        tGap = tEnd - tStart  # Calculate the duration of the two points - 
        # print('vStart-1',algebraic[0][idxVStart-1])           # for testing the voltage, by using the neighboring voltage
        # print('vStart',algebraic[0][idxVStart])
        # print('vStart+1',algebraic[0][idxVStart+1])   
        # print('vEnd-1',vAmpRight[idxVEnd-1])
        # print('vEnd',vAmpRight[idxVEnd])
        # print('vEnd+1',vAmpRight[idxVEnd+1])    
        # print('vStart = ',vAmpLeft[idxVStart], 'vEnd = ',vAmpRight[idxVEnd], 'tStart = ', tStart, 'tEnd = ', tEnd)
                        
        return vAmpLeft[idxVStart], vAmpRight[idxVEnd], tStart, tEnd, tGap, vAmp, vMax, vMin, tGapAmp
        
    def custom_piecewise(self, cases):
        """Compute result of a piecewise function"""
        return select(cases[0::2], cases[1::2])
        
    def solve_model(self, constants):
        """Solve model with ODE solver"""
        from scipy.integrate import ode
        # Initialise constants and state variables  
        init_states = self.initConsts()[0]
        # Set timespan to solve over
        voi = linspace(0, 1300, 5000)
    
        # Construct ODE object to solve
        r = ode(self.computeRates)
        r.set_integrator('vode', method='bdf', atol=1e-006, rtol=1e-006, max_step=0.1)
        r.set_initial_value(init_states, voi[0])
        r.set_f_params(constants)
    
        # Solve model
        states = array([[0.0] * len(voi)] * sizeStates)
        states[:, 0] = init_states
        for (i, t) in enumerate(voi[1:]):
            if r.successful():
                r.integrate(t)
                states[:, i + 1] = r.y
            else:
                break
    
        # Compute algebraic variables
        algebraic = self.computeAlgebraic(constants, states, voi)
        return (voi, states, algebraic)
    
    def morris(self, p, percentage):
        
        delta = 1 / (p - 1)  # Calculate delta according to Morris
        xi = arange(0, p, delta)  # Initiate x base values 
        
        print('Morris parameters - ', 'delta:', delta, ', p(level):', p, ', AP percentage:', percentage)
        
        y = zeros((30 - 25 + 25), dtype=object)  # Debug number can be reduced     + 25 
        # for idx_r in range(r):        
            # print('Calculating y when r is', idx_r, ':')
        x = self.initConsts()[1]
        r_time = 0
        (voi, states, algebraic) = self.solve_model(x)  # Calculate y0 base
        plots = [algebraic] * sizeInputs
        plots[0] = algebraic
        y[r_time] = self.getAmpPerc(voi, algebraic, percentage)
        # print('x',r_time,':',x)
        print('y', r_time, ':', y[r_time])
        
        for idx in range(3, len(x) - 25 + 25):  # Debug number can be reduced            
            x[idx] = x[idx] + delta  # Calculate x'
            (voi, states, algebraic) = self.solve_model(x)  # Calculate y'
            plots[r_time + 1] = algebraic
            y[r_time + 1] = self.getAmpPerc(voi, algebraic, percentage)
            # print('x',r_time,':',x)
            print('y', r_time + 1, ':', y[r_time + 1])
            r_time = r_time + 1    
              
        print('Calculating y finished.')  
        # Outputs(9 outputs): vAmpLeft[idxVStart], vAmpRight[idxVEnd], tStart, tEnd, tGap, vAmp, vMax, vMin, tGapAmp
        
        # Calculate elementary effect
        ee = zeros((30 - 1 - 25 + 25), dtype=object)  # Debug number can be reduced
        
        y_pre = array([], dtype=float32)
        for y_idx in range(len(y)):
            val = array(y[y_idx], dtype=float32)
            if(len(y_pre) == 0):
                y_pre = val
            else:
                ee[y_idx - 1] = (abs(val - y_pre)) / delta  # ee = (y' - y) / delta
            # print('ee',y_idx,':', ee[r_idx][y_idx - 1])
        print('Calculating EE finished.')    
        # print('length of ee:', len(ee))       
        
        self.get_results(ee, [5, 6], percentage, plots, True)
    
    def get_results(self, ee, var, percentage, plots, ordered):  # the number of var can be chosen from 1 to 9 to get different outputs  
    
        ord = zeros((len(ee), 9), dtype=object)
        for idx in range(len(ee)):
            if(isinstance(ee[idx], float)): 
                ord[idx] = ee[idx]
            else:
                ord[idx] = ee[idx].tolist()
                
        title = self.initTitles(percentage)
        label = self.initLabels()
        
        plt.figure(figsize=(30, 30))
        plt.tight_layout() 
        plt.subplots_adjust(wspace=0, hspace=0.5)
        idx = 0
        for idx in range(len(var)):
            val_list = (ord[:, var[idx] - 1:var[idx]].T)[0] 
            label_list = [''] * (len(ee))
            sorted_list = dict() 
            for ee_idx in range(len(ee)):
                label_list[ee_idx] = 'x' + str(ee_idx + 1)  # + '\n' + label[ee_idx];
                sorted_list[label_list[ee_idx]] = val_list[ee_idx]        
            
            if(ordered):
                sorted_list = dict(sorted(sorted_list.items(), key=lambda item:item[1], reverse=True))
                label_list = list(sorted_list.keys())
                val_list = sorted_list.values()
           
            plt.subplot(len(var) * 100 + 10 + (idx + 1))
            plt.title('Elementary Effect of ' + title[var[idx]])
            # plt.ylabel('$\ee$')  
            # plt.xticks(rotation=45)
            plt.xlim((-1, len(val_list) + 1))
            bars = plt.bar(range(len(val_list)), val_list, color='steelblue', tick_label=label_list, alpha=0.75)
            for bar in bars:
                plt.text(bar.get_x() + bar.get_width() / 2, 0.55 * bar.get_height(), '%f' % float(bar.get_height()), ha='center', rotation=45, va='bottom', fontsize=6)

        plt.show()  
        
        #
        plt.figure(figsize=(30, 30))
        plt.tight_layout() 
        plt.title('Action Potential at the percentage of ' + str(percentage * 100) + '%')
        (legend_states, legend_algebraic, legend_voi, legend_constants) = self.createLegends()
        plt.xlabel(legend_voi)
        plt.ylabel(legend_algebraic[0])
        plt.xlim(0, 2000)
        for plot in plots:
            plt.plot(vstack((plot)).T)
            plt.legend(label, loc=1, borderaxespad=0.5, fontsize='7')
        plt.show()  
        

class Command:
    
    """\
    ------------------------------------------------------------
    USE: python <PROGNAME> (options) -- e.g. Bueno.py -l 20 -p 0.1
    OPTIONS:
        -h : print this help message
        -l : the level of trajectory for Morris method (default: 20)
        -p : the percentage of the amplitude of Action Potential (default: 0)
        -o FILE : output results to file FILE (default: output to stdout)
    ------------------------------------------------------------\
        """
        
    def __init__(self):
        level = 20
        percentage = 0
        opts, args = getopt.getopt(sys.argv[1:], 'hl:p:o:')        
        for opt, arg in opts:
        
            if '-h' == opt:
                self.printHelp()
            
            if '-l' == opt:
                self.level = int(arg)
                
            if '-p' == opt:
                self.percentage = float(arg)
                if(self.percentage < 0 or self.percentage > 1.0):
                    print("*** ERROR: arguments, the percentage is between 0.0 and 1.0 ***", file=sys.stderr)
                
            if len(args) > 0:
                print("*** ERROR: arguments, please check the options below ***", file=sys.stderr)
                self.printHelp()
                
    def printHelp(self):        
        help = self.__doc__.replace('<PROGNAME>', sys.argv[0], 1)
        print(help, file=sys.stderr)
        sys.exit()    


if __name__ == "__main__":
    config = Command()
    # (init_states, constants) = initConsts()   
    # plot_model(voi, states, algebraic)
    # (voi, states, algebraic) = solve_model()
    # getAmp(voi, algebraic)
    # getAmpPerc(voi, algebraic, 0.5)
    print('Morris starts.')
    bueno = Bueno()
    bueno.morris(config.level, config.percentage)    
    print('Morris ends.')

