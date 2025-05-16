import numpy as np
import qutip as qt
from operator import add
from tqdm import notebook
import line_profiler
import matplotlib.pyplot as plt
profile = line_profiler.LineProfiler()
import threading
from scipy.stats import norm
import multiprocessing

def plot_estimate_hist(time, estimate_signal, True_signal, N_states, title, save=False, filename='test'):
    #fig, ax = plt.subplots(figsize=(16, 8))

    c = plt.imshow(estimate_signal.T, aspect='auto', extent=[0, 
                    time[-1], -1/2, N_states -1/2], origin='lower', 
                    cmap='Blues', vmin=0, vmax=1, interpolation='none')
    plt.plot(time, True_signal, label='True', color='red')
    #ax.set_xlabel('$\gamma t$')
    plt.legend()
    plt.colorbar(c)
    plt.title(title)
    if save:
        file_path =  + f'{filename}.png'
        plt.savefig(file_path)
    plt.show()

def plot_estimate_trajectory(time, N_states, estimator, title, save=False, filename='test'):
    fig, ax = plt.subplots(1, 3, figsize=(16, 8), sharex=True)
    for i in range(N_states):
        ax[0].plot(time, estimator.P_n_rho[:, i])
        ax[1].plot(time, estimator.P_n_E[:, i])
        ax[2].plot(time, estimator.P_n_PQS[:, i])
    
    #ax[0].legend()
    #ax[1].legend()
    #ax[2].legend()

    plt.tight_layout()
    plt.title(title)
    if save:
        plt.savefig(f'{filename}.png')
   
    plt.show()



class PQSSolver:
    def __init__(self, H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, phi=0, detector_effecency_w=None, detector_effecency_c=None, timedependent_H=False, debug=False, shot_noise=None):
        
        self.H = H
        self.rho_0 = rho_0 
        self.times = times
        self.N_t = len(times)
        self.dt = times[1] - times[0]
        self.index = 0
        self.c_ops = c_ops
        self.c_ops_numbers = len(c_ops[0])
        self.sc_ops_w = sc_ops_w #wieiner prosess
        self.sc_ops_w_numbers = len(sc_ops_w[0])
        self.sc_ops_c = sc_ops_c #jump prosess prosess
        self.sc_ops_c_numbers = len(sc_ops_c[0])
        
        self.e_ops_numbers = len(e_ops[0])
        self.rho = []#np.zeros((self.N_t, *np.shape(rho_0)), dtype=complex)
        self.rho.append(self.rho_0)
        self.phi = phi
        self.expectation = np.zeros((self.N_t, self.e_ops_numbers), dtype=complex)
        self.e_ops = e_ops
        self.timedependent_H = timedependent_H
        self.measurment = False
        self.system_size = 0#int(rho_0.shape[0])  ### broken
        self.debug = debug
        self.N_states = 0
        

        
        self.nu_0 = np.zeros(self.c_ops_numbers, dtype=float)

        if shot_noise == None:
            self.shot_noise = np.zeros(self.sc_ops_c_numbers)
        else:
            self.shot_noise = shot_noise

        if detector_effecency_w == None:
            self.nu_w = np.ones(self.sc_ops_w_numbers)
        else:
            self.nu_w = detector_effecency_w

        if detector_effecency_c == None:
            self.nu_c = np.ones(self.sc_ops_c_numbers)
        else:
            self.nu_c = detector_effecency_c
        
        

    def add_delta_W(self, delta_W):
        
        self.delta_W = delta_W
    
    def add_std_vec(self, std_vec):
        
        self.std_vec = std_vec
    
    def variance_oporator(self, c, rho):
        """Calculates the variance of the output oporator"""
        variance = self.expectation_value((c + c.dag()) ** 2, rho) - self.expectation_value(c + c.dag(), rho) ** 2
        return variance
    
    def ensemble_expectation_value(self, c, rho_full):
        """Calculates the expectation value of oporator"""
        if self.N_states == 0:
            expectation_value_ensamble = self.expectation_value(c + c.dag(), rho_full)
        else:
            expectation_value_ensamble = 0
            for i in range(self.N_states):
                expectation_value_ensamble += self.expectation_value(c + c.dag(), rho_full[i])
        
        return expectation_value_ensamble
        
    
    def D_operator(self, c, rho):
        """Calculates the D superoperator, witch is the linblad dissipation terms"""
        D = c * rho * c.dag() - 1 / 2 * (c.dag() * c * rho + rho * c.dag() * c)
        return D
    

    def D_operator_dag(self, c, E):
        """Calculates the D superoperator, witch is the linblad dissipation terms"""
        D = c.dag() * E * c - 1 / 2 * (c.dag() * c * E + E * c.dag() * c)
        return 
    
    
    def lindblad_terms(self, rho, oporators, oporator_number):
        """Calculates the lindblad operator"""

        lindblad = 0
        for i in range(oporator_number):
            lindblad += self.D_operator(oporators[i], rho)

        return lindblad
    
    
    def H_operator(self, c, rho, rho_full):
        """Calculates the H superoperator, witch is the superoporator for the backaction of the measurement"""
        
        expectation_value_ensamble = self.ensemble_expectation_value(c + c.dag(), rho_full)
        #print('expectation value ensamble', expectation_value_ensamble)
       
        H = c * rho + rho * c.dag() - expectation_value_ensamble * rho # here this is the ensable average of the expectation value
        return H
    
    def H_operator_dag(self, c, E, E_full):
        """Calculates the H superoperator, witch is the superoporator for the backaction of the measurement"""

        expectation_value_ensamble = self.ensemble_expectation_value(c + c.dag(), E_full)

        H = c.dag() * E + E * c - expectation_value_ensamble * E
        return H
    
    def H_operator_counting(self, c, rho, rho_full):
        """Calculates the H superoperator, witch is the superoporator for the backaction of the measurement"""

        expectation_value_ensamble = self.ensemble_expectation_value((c.dag() * c), rho_full)
        
        #print('expectation value ensamble', expectation_value_ensamble)
       
        H = c * rho * c.dag() / expectation_value_ensamble - rho # here this is the ensable average of the expectation value
        return H
    
    def H_operator_counting_dag(self, c, E, E_full):
        """Calculates the H superoperator, witch is the superoporator for the backaction of the measurement"""
        
        expectation_value_ensamble = self.expectation_value((c.dag() * c), E_full)

        H = c.dag() * E * c / expectation_value_ensamble - E 
        return H

    def M_operator(self, c, rho):
        """Calculates the M superoperator, witch is the superoporator for the backaction of the measurement"""

        M = c * rho + rho * c.dag()
        return M
    
    def backaction_terms(self, rho, rho_full, sc_ops, dw):
        """Calculates the backaction of the measurements"""

        backaction = 0
        for i in range(self.sc_ops_numbers):
            backaction += self.H_operator(sc_ops[i], rho, rho_full) * dw[i]      


        return backaction
    

    def backaction_terms_dag(self, E, E_full, sc_ops, dw):
        """Calculates the backaction of the measurements"""

        backaction = 0
        for i in range(self.sc_ops_numbers):
            backaction += self.H_operator_dag(sc_ops[i], E, E_full) * dw[i]        

        return backaction
    

        
    def func(self, H, rho, rho_full, c_ops, sc_ops, dw, split=False):
        """The function to be solved"""
        
        Hamiltion_term = - 1j * (H * rho - rho * H) * self.dt
            

        lindblad_contribution = self.lindblad_terms(rho, c_ops, self.c_ops_numbers) * self.dt
        measurment_backaction = self.lindblad_terms(rho, sc_ops, self.sc_ops_numbers) * self.dt

        if split:
            return Hamiltion_term + lindblad_contribution + measurment_backaction, self.backaction_terms(rho, rho_full, sc_ops, dw)
        else:
            measurment_backaction += self.backaction_terms(rho, rho_full, sc_ops, dw)
            #return measurment_backaction + lindblad_contribution
            return Hamiltion_term + lindblad_contribution + measurment_backaction


        
    def stocastic_split_function(self):
        """Splits the stocastic eqation dx = f(x)dt + g(x)dw in to f(x) and g(x)"""
        
        def f(rho, H, c_ops, sc_ops):
            Hamiltion_term = -1j * (H * rho - rho * H)
            lindblad_contribution = self.lindblad_terms(rho, c_ops, self.c_ops_numbers)
            measurment_lindblad = self.lindblad_terms(rho, sc_ops, self.sc_ops_numbers)
            return Hamiltion_term + lindblad_contribution + measurment_lindblad
            
        def g(rho, rho_full, sc_ops, j):
            measurment_backaction = self.H_operator(sc_ops[j], rho, rho_full)
            return measurment_backaction
        
        return f, g
    

    def g_diff(self, l, l_p, sc_ops, nu, rho):
        """Calculates the differential matrix of g"""
        matrix = np.zeros((self.system_size, self.system_size), dtype=complex)
        sc = sc_ops.full()
        
        rho_full = rho.full()
        matrix[:, l_p] += sc[:, l] # indexing problems, as when indexing a qutip tensor, it wil index the densor matrixes insted of just the rwo and coloume
        matrix[l, :] += sc[l_p]
        matrix += (sc[l_p, l] + sc[l_p, l]) * rho_full
        matrix[l, l_p] += self.expectation_value(sc_ops + sc_ops.dag(), rho)
        matrix = qt.Qobj(nu * matrix, dims=rho.dims)
        return matrix


    def Lg(self, j_1, j_2, g, rho, rho_full, sc_ops):
        """Calculates the Lg matrix"""
        matrix = qt.qzero_like(rho)
        
        for l in range(self.system_size):
            for l_p in range(self.system_size):
                matrix = matrix + g(rho, rho_full, sc_ops, j_1).full()[l][l_p] * self.g_diff(l, l_p, sc_ops[j_2], self.nu[j_2], rho)

        return matrix

    
    def Milstein_next_step(self, H, rho, rho_full, c_ops, sc_ops, dw):
        """Calculates the next values with the Milstein methode"""
        
        f, g = self.stocastic_split_function()
        d_rho = f(rho, H, c_ops, sc_ops) * self.dt
        for j_1 in range(self.sc_ops_numbers):
            d_rho += g(rho, rho_full, sc_ops, j_1) * dw[j_1]
            d_rho += - 1/2 * self.Lg(j_1, j_1, g, rho, rho_full, sc_ops) * self.dt
            for j_2 in range(self.sc_ops_numbers):
                d_rho += self.Lg(j_1, j_2, g, rho, rho_full, sc_ops) * dw[j_1] * dw[j_2]

        return d_rho

    def RK_stocastic_next_step(self, delta_dw, index):
        """Calculates the next values with the Sticastic Runge-Kutta methode"""
        
        a, b = self.func(index, self.rho[index], delta_dw, split=True)
        
        Y_est = self.rho[index] + a + self.backaction_terms(self.rho[index], index, self.std_vec)

        d_rho = a + b + 1/2 * (self.backaction_terms(Y_est, index, (self.delta_W ** 2 - self.dt) * np.sqrt(self.dt)) - self.backaction_terms(self.rho[index], index, (self.delta_W ** 2 - self.dt) * np.sqrt(self.dt)))

        return d_rho

    def Euler_next_step(self, H, rho, rho_full, c_ops, sc_ops, dw):
        """Calculates the next values with the Euler methode"""
        
        d_rho = self.func(H, rho, rho_full, c_ops, sc_ops, dw)
        
        return d_rho
    
    def Kraus_next_step(self, rho, nu, M_dy, L):
        """Calculates the next values with the Kraus methode"""
        
        rho_new = M_dy * rho * M_dy.dag() + self.kraus_c_ops(rho, nu, L, self.sc_ops_w_numbers) * self.dt
        return rho_new
    
    def Kraus_next_step_E(self, H, rho, c_ops, sc_ops, nu, dY):
        """Calculates the next values with the Kraus methode"""
        
        M, L = self.Kraus_operator(H, rho, c_ops, sc_ops, nu, self.c_ops_numbers, self.sc_ops_numbers)
        M_dy = self.Kraus_M_dy(M, L, nu, dY, self.sc_ops_numbers)
        rho_new = M_dy * rho * M_dy.dag() + self.kraus_sc_ops(rho, nu, L) * self.dt
        
        return rho_new
    

    def Kraus_operator(self, H, rho, c_ops, sc_ops, c_ops_number, sc_ops_number):
        """Calculates the Kraus operator"""
        M_0 = qt.qeye_like(rho)
        M_0 += -1j * H * self.dt
        for i in range(sc_ops_number):
            M_0 += -1/2 * sc_ops[i].dag() * sc_ops[i] * self.dt
        for i in range(c_ops_number):
            M_0 += -1/2 * c_ops[i].dag() * c_ops[i] * self.dt

        S = M_0.dag() * M_0
        for i in range(sc_ops_number):
            S += sc_ops[i].dag() * sc_ops[i] * self.dt
        for i in range(c_ops_number):
            S += c_ops[i].dag() * c_ops[i] * self.dt

        S_inv_sqrt = (S.inv()).sqrtm()

        M = M_0 * S_inv_sqrt
        L = []
        for i in range(sc_ops_number):
            L.append(sc_ops[i] * S_inv_sqrt)

        return M, L
    
    def Kraus_operator_simple(self, H, rho, c_ops, sc_ops, c_ops_number, sc_ops_number):
        """Calculates the Kraus operator"""
        M_0 = qt.qeye_like(rho)
        M_0 += -1j * H * self.dt
        for i in range(sc_ops_number):
            M_0 += -1/2 * sc_ops[i].dag() * sc_ops[i] * self.dt
        for i in range(c_ops_number):
            M_0 += -1/2 * c_ops[i].dag() * c_ops[i] * self.dt
        
        return M_0, sc_ops
    
    def kraus_c_ops(self, rho, nu, L, L_number):
        """Calculates the kraus superoperator"""
        L_rho = 0
        for i in range(L_number):
            L_rho += (1 - nu[i]) * L[i] * rho * L[i].dag()
        
        return L_rho
    
    def kraus_c_ops_dif(self, rho, rho_dif, nu, L, L_dif, L_number):
        """Calculates the differential kraus superoperator"""
        L_rho_dif = 0
        for i in range(L_number):
            L_rho_dif += (1 - nu[i]) * (L_dif[i] * rho * L[i].dag() + L[i] * rho_dif * L[i].dag() + L[i] * rho * L_dif[i].dag())
        
        return L_rho_dif

    def next_step(self, dw, H, rho, rho_full, c_ops, sc_ops, methode='Euler', order=1, dY=None):
        """Calculates the next values with the given methode"""

        if methode == 'Euler':
            d_rho = self.Euler_next_step(H, rho, rho_full, c_ops, sc_ops, dw) # working
        elif methode == 'Adam Bashforth':
            d_rho = self.Adam_Bashforth_next_step(H, rho, c_ops, sc_ops, dw, order) # broken
        elif methode == 'Milstein':
            d_rho = self.Milstein_next_step(H, rho, rho_full, c_ops, sc_ops, dw) #broken
        elif methode == 'RK stocastic':
            d_rho = self.RK_stocastic_next_step(H, rho, c_ops, sc_ops, self.delta_W) #broken
        elif methode == 'Kraus':                        
            rho = self.Kraus_next_step(H, rho, c_ops, sc_ops, self.nu, dY)
            return rho
        
        return d_rho


    def expectation_value(self, c, rho):
        """Calculates the expectation value of the output oporator"""

        expectation = ((c * rho)).tr()
        return expectation
    
    
    def Kraus_M_dy(self, M_0, L, nu, dY, L_number):
        """Calculates the Kraus operator"""
        M = M_0
        for i in range(L_number):
            M += np.sqrt(nu[i]) * dY[i] * L[i]
        
        return M
    
    def Kraus_M_dy_mix(self, H, c_ops, sc_ops_w, sc_ops_c, dY):
        """Calculates the Kraus operator for the case where there boath is a jump and a wieiner prosess"""
        M_dy = qt.qeye_like(self.rho_0) - 1j * H * self.dt
       

        for i in range(self.c_ops_numbers):
            M_dy += -1/2 * c_ops[i].dag() * c_ops[i] * self.dt
        

        for i in range(self.sc_ops_w_numbers):
            M_dy += -1/2 * sc_ops_w[i].dag() * sc_ops_w[i] * self.dt
        
        
        for i in range(self.sc_ops_c_numbers):
            M_dy += -1/2 * sc_ops_c[i].dag() * sc_ops_c[i] * self.dt
        
        
        for i in range(self.sc_ops_w_numbers):
            M_dy += dY[i] * sc_ops_w[i]
        

        return M_dy
    
    

    
    def Kraus_mix_next_step(self, M_dy, rho, c_ops, sc_ops_w, sc_ops_c, nu_w, nu_c, nu_0, shot_noise, dN):
        """Calculates the next values with the Kraus methode"""
        index = np.where(dN == 1)[0][0]
        
        if dN[-1] == 1:
            rho_tilde = rho
        else:
            rho_tilde = self.detection_counting_rho(rho, sc_ops_c[index], nu_c[index], shot_noise[index])


        rho_new = M_dy * rho_tilde * M_dy.dag() + (self.kraus_c_ops(rho_tilde, nu_0, c_ops, self.c_ops_numbers) 
                                           + self.kraus_c_ops(rho_tilde, nu_w, sc_ops_w, self.sc_ops_w_numbers)
                                           + self.kraus_c_ops(rho_tilde, nu_c, sc_ops_c, self.sc_ops_c_numbers)) * self.dt

        return rho_new
    
    
    
    def detection_counting_rho(self, rho, sc_ops_c, nu_c, shot_noise):
        
        rho_tilde = shot_noise * rho + nu_c * sc_ops_c * rho * sc_ops_c.dag()
        norm_factor = shot_noise + nu_c * (sc_ops_c * rho * sc_ops_c.dag()).tr()
        rho_tilde = rho_tilde / norm_factor
        
        return rho_tilde
    
    
    
    


         
class Experiment_simulation(PQSSolver):
    def __init__(self, H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, phi=0, detector_effecency_w=None, detector_effecency_c=None, timedependent_H=False, debug=False):
        super().__init__(H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, phi, detector_effecency_w, detector_effecency_c, timedependent_H, debug=debug)

    def gennerate_outputsignal(self, dw, rho, c, nu):
        """Calculates the output signal of the detection from stocastic dw"""
        dY = np.sqrt(nu) * dw + nu * self.expectation_value(c + c.dag(), rho) * self.dt
        return dY
    
    def gennerate_outputsignal_mixed(self, dw, rho, sc_ops_w, nu):
        """Calculates the output signal for all the detections form winer prosses measurments"""
        
        dY = np.zeros(self.sc_ops_w_numbers)
        for i in range(self.sc_ops_w_numbers):
            dY[i] = np.sqrt(nu[i]) * dw[i] + nu[i] * self.expectation_value(sc_ops_w[i] + sc_ops_w[i].dag(), rho) * self.dt
        
        return dY

    def gennerate_outputsignal_Kraus(self, M_0, rho, L, nu, L_ops_number):
        """Calculates the output signal of the detection from stocastic dw for the Kraus methode"""
        dY, M_dy = self.sample_Kraus_pdf(rho, M_0, L, nu, L_ops_number)
        dY *= np.sqrt(self.dt)
        return dY, M_dy
    
    def Kras_pdf(self, s, rho, M_0, L, nu, L_ops_number):
        """Calculates the probability density function for the Kraus methode"""
        M_s = self.Kraus_M_dy(M_0, L, nu, s * self.dt, L_ops_number)
        matrix = M_s * rho * M_s.dag() + self.kraus_sc_ops(rho, nu, L) * self.dt
        pdf = (matrix * matrix.dag()).tr()
        for i in range(L_ops_number):
            pdf *= norm.pdf(s[i])
        return pdf, M_s
    
    def sample_Kraus_pdf(self, rho, M_0, L, nu, L_ops_number, max_steps=100):
        """Samples the probability density function for the Kraus methode"""
        for i in range(max_steps):
            s = np.zeros(L_ops_number)
            hight = 1
            for i in range(L_ops_number):
                s[i] = np.random.normal(0, 1)
                hight *= norm.pdf(s[i])
            
            safty_higth = 1 + 5 * self.dt
            control = np.random.uniform(0, hight * safty_higth)
            pdf, M_dy = self.Kras_pdf(s, rho, M_0, L, nu, L_ops_number)
            if hight * safty_higth < pdf:
                print('warning, pdf value exceeded the control gaussian')
            if control <= pdf:
                return s, M_dy
        print('warning, max steps reached')
            
    def calculate_expectation_values(self, rho, e_ops, index):
        """Calculates the expectation value of the output oporator"""
        
        for i in range(self.e_ops_numbers):
            self.expectation[index][i] = self.expectation_value(e_ops[i], rho)

    def gennerate_dw(self, noise_factor=1):
        """Generates the stocastic dw"""
        dw = np.random.normal(0, np.sqrt(self.dt) * noise_factor, (self.N_t, self.sc_ops_w_numbers))
        #dw[:, self.measurment_type] = np.zeros((self.N_t))
        return dw
    
    def calculate_dN(self, rho, sc_ops_c):
        """Calculates the dN"""
        
        probability = np.zeros(self.sc_ops_c_numbers + 1)
        for i in range(self.sc_ops_c_numbers):
            probability[i] = (self.shot_noise[i] + self.nu_c[i] * self.expectation_value(sc_ops_c[i].dag() * sc_ops_c[i], rho)) * self.dt
        probability[-1] = 1 - np.sum(probability)
        index = np.random.choice(self.sc_ops_c_numbers + 1, p=probability)
        dN = np.zeros(self.sc_ops_c_numbers + 1)
        
        dN[index] = 1
        return dN

    def solve(self, methode='Kraus', type_='homodyne', noise_factor=1, order=1):
        """Solves the stocastic master equation with detection"""

        if type_ == 'homodyne':
            dw = self.gennerate_dw(noise_factor)
        if type_ == 'counting':
            dw = np.zeros((self.N_t, self.sc_ops_w_numbers))
            theshold = np.random.uniform(0, 1, 1)[0]
            colapse_operator = np.random.randint(0, self.sc_ops_w_numbers, 1)[0]
            dY = np.zeros((self.N_t, self.sc_ops_w_numbers))
            state_norm = 1
            
        self.dw = dw
        self.delta_W = dw - np.roll(dw, 1, axis=0)
        self.std_vec = np.ones_like(dw) * np.sqrt(self.dt)
        
        dY = np.zeros((self.N_t, self.sc_ops_w_numbers))
        #if methode == 'Kraus':
        #    M_0, L = self.Kraus_operator_simple(self.H[0], self.rho[0], self.c_ops, self.sc_ops, self.c_ops_numbers, self.sc_ops_numbers)
        self.calculate_expectation_values(self.rho[0], self.e_ops[0], 0)
        for index in range(self.N_t-1):

            if methode == 'Kraus':
                M_0, L = self.Kraus_operator(self.H[index], self.rho[index], self.c_ops[index], self.sc_ops_w[index], self.c_ops_numbers, self.sc_ops_w_numbers)
                #dY[index], M_dy = self.gennerate_outputsignal_Kraus(M_0, self.rho[index], L, self.nu, self.sc_ops_numbers)
                if type_ == 'homodyne':
                    for i in range(self.sc_ops_w_numbers):
                        dY[index][i] = self.gennerate_outputsignal(dw[index][i], self.rho[index], self.sc_ops_w[index][i], self.nu_w[i])  
                    M_dy = self.Kraus_M_dy(M_0, L, self.nu_w, dY[index], self.sc_ops_w_numbers)
                    rho_new = self.Kraus_next_step(self.rho[index], self.nu_w, M_dy, L)
                    self.normelization_Kraus(rho_new, index)
                elif type_ == 'counting':
                    if np.sum(dY[index]) == 1:
                        rho_new = self.colapse_state(self.rho[index], self.sc_ops_w[index][colapse_operator])
                        #print('index', index, rho_new, rho_new.tr())
                        self.normelization_Kraus(rho_new, index)
                        state_norm = 1
                        colapse_operator = np.random.randint(0, self.sc_ops_w_numbers, 1)[0]
                        theshold = np.random.uniform(0, 1, 1)[0]
                    else:
                        rho_new = self.Kraus_next_step(self.rho[index], self.nu_w, M_0, L)
                        state_norm = self.normelization_Kraus(rho_new, index, type_='counting')
                        
                    if state_norm <= theshold:
                        dY[index + 1, colapse_operator] = 1
                        dw[index + 1, colapse_operator] = 1

                        


                    
            else:
                for i in range(self.sc_ops_w_numbers):
                    dY[index][i] = self.gennerate_outputsignal(dw[index][i], self.rho[index], self.sc_ops_w[index][i], self.nu_w[i])  
                d_rho = self.next_step(dw[index], self.H[index], self.rho[index], self.rho[index], 
                                            self.c_ops[index], self.sc_ops_w[index], methode=methode, order=order)
                rho_new = self.normelization(d_rho, index)
                self.rho.append(rho_new)
            self.calculate_expectation_values(self.rho[index + 1], self.e_ops[index + 1], index + 1)

        self.detection_record = dY
        self.dw = dw

    def solve_mixed(self):
        """solves the stocastic master equation with a mix of counting and homodyne detection"""
        dw = self.gennerate_dw()
        self.dw = dw
        dY = np.zeros((self.N_t, self.sc_ops_w_numbers))
        dN = np.zeros((self.N_t, self.sc_ops_c_numbers + 1))
        self.normfactor = np.zeros(self.N_t)
        self.calculate_expectation_values(self.rho[0], self.e_ops[0], 0)
        for index in range(self.N_t-1):
            dN[index] = self.calculate_dN(self.rho[index], self.sc_ops_c[index])
            dY[index] = self.gennerate_outputsignal_mixed(dw[index], self.rho[index], self.sc_ops_w[index], self.nu_w)
            M_dy = self.Kraus_M_dy_mix(self.H[index], self.c_ops[index], self.sc_ops_w[index], self.sc_ops_c[index], dY[index])
            rho_new = self.Kraus_mix_next_step(M_dy, self.rho[index], self.c_ops[index], self.sc_ops_w[index], self.sc_ops_c[index], self.nu_w, self.nu_c, self.nu_0, self.shot_noise, dN[index])
            self.normelization_Kraus_mixed(rho_new, index)
            self.calculate_expectation_values(self.rho[index + 1], self.e_ops[index + 1], index + 1)
        


        self.detection_record = dY
        self.dw = dw
        self.dY = dY
        self.dN = dN

    def colapse_state(self, rho, sc_op):
        """Colapses the state"""
        rho_new = sc_op * rho * sc_op.dag()
        return rho_new

    def normelization(self, d_rho, index):
        rho_new = self.rho[index] + d_rho
        rho_new = rho_new / rho_new.tr()
        return rho_new
        #self.rho.append(rho_new)
    
    def normelization_Kraus(self, rho_new, index, type_='homodyne'):
        new_norm = rho_new.tr()
        if type_ == 'counting':
            self.rho.append(rho_new)
            return new_norm
        
        rho_new = rho_new / new_norm
        self.rho.append(rho_new)

    def normelization_Kraus_mixed(self, rho_new, index):
        norm_factor = rho_new.tr()
        rho_new = rho_new / norm_factor
        self.rho.append(rho_new)
        self.normfactor[index] = norm_factor
        
    def comutator(self, A, B):
        return A * B - B * A

    def anticomutator(self, A, B):
        return A * B + B * A
    
    def reset(self):
        self.rho = []
        self.rho.append(self.rho_0)
        self.expectation = np.zeros((self.N_t, self.e_ops_numbers), dtype=complex)
        self.index = 0


    
        

    
        
        
class Experiment_Fisher_estimation(Experiment_simulation):
    def __init__(self, H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, peram_number, c_ops_dif, H_dif, sc_ops_w_dif, sc_ops_c_dif, phi=0, detector_effecency_c=None, detector_effecency_w=None, timedependent_H=False, debug=False):
        super().__init__(H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, phi, detector_effecency_c=detector_effecency_c, detector_effecency_w=detector_effecency_w, timedependent_H=timedependent_H, debug=debug)
        self.parm_number = peram_number
        self.c_ops_dif = c_ops_dif
        self.sc_ops_w_dif = sc_ops_w_dif
        self.sc_ops_c_dif = sc_ops_c_dif
        self.H_dif = H_dif

    def D_operator_dif(self, c, c_dif, rho, rho_dif):
        D = c * rho * c_dif.dag() + c_dif * rho * c.dag() + c * rho_dif * c.dag() - 1/2 * self.anticomutator(c.dag() * c_dif + c_dif.dag() * c, rho
                                ) - 1/2 * self.anticomutator(c.dag() * c + c.dag() * c_dif, rho_dif)
        return D

    def H_operator_dif(self, c, c_dif, rho, rho_dif):
        H = c * rho_dif + rho_dif * c.dag() + c_dif * rho + rho * c_dif.dag() - self.expectation_value(c + c.dag(), rho) * rho_dif
        return H

    def calculate_fisher_information(self, N_fisher, sigma_z_1, type_='homodyne'):
        """Calculates the Fisher information"""
        self.sigma_z_1 = sigma_z_1
        rho_t_tr = np.zeros((N_fisher, self.N_t, self.parm_number), dtype=complex)
        d_rho_t_tr = np.zeros((N_fisher, self.N_t, self.parm_number), dtype=complex)
        #print('c_op', self.c_ops_numbers)
        #print('sc_op', self.sc_ops_numbers)
        termes = np.zeros((N_fisher, self.N_t, 1 + self.c_ops_numbers + 2 * self.sc_ops_w_numbers), dtype=complex)
        for i in range(N_fisher): # This is the number of trajectories used to estimate the fisher information
            super().reset()
            super().solve(type_=type_)
            rho_t_tr[i], d_rho_t_tr[i], termes[i] = self.solve_fisher_tr(self.c_ops_dif, self.sc_ops_w_dif, self.H_dif, type_=type_)

        rho_t_tr = np.swapaxes(rho_t_tr, 0, 1)
        rho_t_tr = np.swapaxes(rho_t_tr, 1, 2)
        rho_t_tr = np.swapaxes(rho_t_tr, 0, 1)
        #print('rho_t_tr', rho_t_tr.shape)
        d_rho_t_tr = np.swapaxes(d_rho_t_tr, 0, 1)
        d_rho_t_tr = np.swapaxes(d_rho_t_tr, 1, 2)
        d_rho_t_tr = np.mean(d_rho_t_tr, axis=2)
        termes = np.swapaxes(termes, 0, 1)

        termes = np.mean(termes, axis=1)
        
        fisher_information_mean, fisher_information_std = fisher_calculator(self.parm_number, self.N_t, N_fisher, rho_t_tr)
        #for i in range(self.N_t):
        #for k in range(self.parm_number):
        #    for l in range(self.parm_number):    
        #        fisher_information[k, l] = rho_t_tr[k] * rho_t_tr[l]   

        #fisher_information_mean = np.mean(fisher_information, axis=3)
        #fisher_information_std = np.std(fisher_information, axis=3) / np.sqrt(N_fisher)
        return fisher_information_mean, fisher_information_std, d_rho_t_tr, termes, rho_t_tr
    
    def calculate_fisher_information_mixed(self, N_fisher):
        """Calculates the Fisher information for mixed dectection methodes"""

        rho_t_tr = np.zeros((N_fisher, self.N_t, self.parm_number), dtype=complex)
        
        for i in range(N_fisher):
            super().reset()
            super().solve_mixed()
            rho_t_tr[i] = self.solve_fisher_tr_mixed(self.c_ops, self.c_ops_dif, self.sc_ops_w, self.sc_ops_w_dif, self.sc_ops_c, self.sc_ops_c_dif, self.H, self.H_dif, self.dY, self.dN)

        rho_t_tr = np.swapaxes(rho_t_tr, 0, 1)
        rho_t_tr = np.swapaxes(rho_t_tr, 1, 2)
        rho_t_tr = np.swapaxes(rho_t_tr, 0, 1)

        fisher_information_mean, fisher_information_std = fisher_calculator(self.parm_number, self.N_t, N_fisher, rho_t_tr)

        return fisher_information_mean, fisher_information_std, rho_t_tr

    def fisher_d_rho_i_t(self, c_ops, c_ops_dif, sc_ops, sc_ops_dif, H, H_dif, rho, rho_i_t, dw, type_='homodyne'):
        d_rho_i_t = qt.qzero_like(self.rho_0)

        termes = np.zeros((1 + self.c_ops_numbers + 2 * self.sc_ops_w_numbers), dtype=complex)

        d_rho_i_t += (- 1j * self.comutator(H, rho_i_t) - 1j * self.comutator(H_dif, rho)) * self.dt
        termes[0] = ((- 1j * self.comutator(H, rho_i_t) - 1j * self.comutator(H_dif, rho)) * self.dt).tr()
        #print('comutator term', ((- 1j * self.comutator(H, rho_i_t) - 1j * self.comutator(H_dif, rho)) * self.dt).tr())
        
        for i in range(self.c_ops_numbers):
            d_rho_i_t += self.D_operator_dif(c_ops[i], c_ops_dif[i], rho, rho_i_t) * self.dt
            termes[i + 1] = (self.D_operator_dif(c_ops[i], c_ops_dif[i], rho, rho_i_t) * self.dt).tr()
            #print(f'colaplse term {i}', (self.D_operator_dif(c_ops[i], c_ops_dif[i], rho, rho_i_t) * self.dt).tr())
        
        for i in range(self.sc_ops_w_numbers):
            d_rho_i_t += self.D_operator_dif(sc_ops[i], sc_ops_dif[i], rho, rho_i_t) * self.dt
            termes[i + 1 + self.c_ops_numbers] = (self.D_operator_dif(sc_ops[i], sc_ops_dif[i], rho, rho_i_t) * self.dt).tr()

        if type_ == 'homodyne':
            for i in range(self.sc_ops_w_numbers):
                d_rho_i_t += self.H_operator_dif(sc_ops[i], sc_ops_dif[i], rho, rho_i_t) * dw[i]
                termes[2 * i + 1 + 1 + self.c_ops_numbers] = (self.H_operator_dif(sc_ops[i], sc_ops_dif[i], rho, rho_i_t) * dw[i]).tr()
                #print(f'measurment D term {i+1} :', (self.D_operator_dif(sc_ops[i], sc_ops_dif[i], rho, rho_i_t) * self.dt).tr())
                #print(f'measurment H term {i+1} :', (self.H_operator_dif(sc_ops[i], sc_ops_dif[i], rho, rho_i_t) * dw[i]).tr())
        
        elif type_ == 'counting':
            for i in range(self.sc_ops_w_numbers):
                d_rho_i_t += (sc_ops[i] * rho_i_t * sc_ops[i].dag() + sc_ops_dif[i] * rho * sc_ops[i].dag() + sc_ops[i] * rho_i_t * sc_ops_dif[i].dag() - rho_i_t) * dw[i]
                termes[2 * i + 1 + 1 + self.c_ops_numbers] = ((sc_ops[i] * rho_i_t * sc_ops[i].dag() + sc_ops_dif[i] * rho * sc_ops[i].dag() + sc_ops[i] * rho_i_t * sc_ops_dif[i].dag() - rho_i_t) * dw[i]).tr()
            
        #print('d_rho_i_t', d_rho_i_t)

        return d_rho_i_t, termes
    
    def fisher_d_rho_i_t_mixed(self, c_ops, c_ops_dif, sc_ops_w, sc_ops_w_dif, sc_ops_c, sc_ops_c_dif, H, H_dif, rho, rho_i_t, dw):
        d_rho_i_t = qt.qzero_like(self.rho_0)

        d_rho_i_t += (- 1j * self.comutator(H, rho_i_t) - 1j * self.comutator(H_dif, rho)) * self.dt

        for i in range(self.c_ops_numbers):
            d_rho_i_t += self.D_operator_dif(c_ops[i], c_ops_dif[i], rho, rho_i_t) * self.dt
        
    def solve_fisher_tr(self, c_ops_dif,  sc_ops_dif, H_dif, type_='homodyne'):
        """Solves the stocastic master equation with detection"""
        
        rho_i_t_tr = np.zeros((self.N_t, self.parm_number), dtype=complex)
        d_rho_i_t_tr = np.zeros((self.N_t, self.parm_number), dtype=complex)
        termes = np.zeros((self.N_t, 1 + self.c_ops_numbers + 2 * self.sc_ops_w_numbers), dtype=complex)
        for i in range(self.parm_number):
            rho_i_t = qt.qzero_like(self.rho_0)
            rho_i_t_tr_ = np.zeros(self.N_t, dtype=complex)
            d_rho_i_t_tr_ = np.zeros(self.N_t, dtype=complex)
            for index in range(self.N_t-1):
                d_rho_i_t, term = self.fisher_d_rho_i_t(self.c_ops[index], c_ops_dif[index][i], self.sc_ops_w[index], sc_ops_dif[index][i], 
                                                  self.H[index], H_dif[index][i], self.rho[index], rho_i_t, self.dw[index], type_=type_)
                #print("d_rho_i_t 0", (self.sigma_z_1 * self.rho[index] + self.rho[index] * self.sigma_z_1).tr())
                rho_i_t = rho_i_t + d_rho_i_t
                rho_i_t_tr_[index + 1] = rho_i_t.tr()
                d_rho_i_t_tr_[index + 1] = d_rho_i_t.tr()
                termes[index + 1] = term
            rho_i_t_tr[:, i] = rho_i_t_tr_
            d_rho_i_t_tr[:, i] = d_rho_i_t_tr_
        
        return rho_i_t_tr, d_rho_i_t_tr, termes
    
    def solve_fisher_tr_mixed(self, c_ops, c_ops_dif, sc_ops_w, sc_ops_w_dif, sc_ops_c, sc_ops_c_dif, H, H_dif, dY, dN):
        """Solves the stocastic equation for fisher information with mixed detection"""

        rho_i_t_tr = np.zeros((self.N_t, self.parm_number), dtype=complex)
        

        for i in range(self.parm_number):
            rho_it = qt.qzero_like(self.rho_0)
            rho_it_tr_ = np.zeros(self.N_t, dtype=complex)
            
            for index in range(self.N_t-1):
                M_dy_dif = self.Kraus_M_dy_mix_dif(H_dif[index][i], c_ops[index], c_ops_dif[index][i], sc_ops_w[index], sc_ops_w_dif[index][i], sc_ops_c[index], sc_ops_c_dif[index][i], dY[index])
                M_dy = self.Kraus_M_dy_mix(H[index], c_ops[index], sc_ops_w[index], sc_ops_c[index], dY[index])
                rho_it = self.Kraus_mix_next_step_dif(M_dy, M_dy_dif, self.rho[index], rho_it, c_ops[index], c_ops_dif[index][i], sc_ops_w[index], sc_ops_w_dif[index][i], sc_ops_c[index], sc_ops_c_dif[index][i], self.nu_w, self.nu_c, self.nu_0, self.shot_noise, dN[index])
                rho_it = rho_it / self.normfactor[index]
                rho_it_tr_[index + 1] = rho_it.tr()
            rho_i_t_tr[:, i] = rho_it_tr_
        return rho_i_t_tr
                



    def Kraus_M_dy_mix_dif(self, H_dif, c_ops, c_ops_dif, sc_ops_w, sc_ops_w_dif, sc_ops_c, sc_ops_c_dif, dY):
        """Calculates the differential of the Kraus operator for the case where there boath is a jump and a wieiner prosess"""
        M_dy_dif = -1j * H_dif * self.dt

        for i in range(self.c_ops_numbers):
            M_dy_dif += -1/2 * c_ops_dif[i].dag() * c_ops[i] * self.dt - 1/2 * c_ops[i].dag() * c_ops_dif[i] * self.dt

        for i in range(self.sc_ops_w_numbers):
            M_dy_dif += -1/2 * sc_ops_w_dif[i].dag() * sc_ops_w[i] * self.dt - 1/2 * sc_ops_w[i].dag() * sc_ops_w_dif[i] * self.dt
            M_dy_dif += dY[i] * sc_ops_w_dif[i]

        for i in range(self.sc_ops_c_numbers):
            M_dy_dif += -1/2 * sc_ops_c_dif[i].dag() * sc_ops_c[i] * self.dt - 1/2 * sc_ops_c[i].dag() * sc_ops_c_dif[i] * self.dt
        
        return M_dy_dif
    
    def fisher_inside_gen(self, c_ops_dif,  sc_ops_dif, H_dif, rho_i_t_tr, d_rho_i_t_tr, termes):
        
        
        
        def function(self, i):
            rho_i_t = qt.qzero_like(self.rho_0)
            rho_i_t_tr_ = np.zeros(self.N_t, dtype=complex)
            d_rho_i_t_tr_ = np.zeros(self.N_t, dtype=complex)
            for index in range(self.N_t-1):
                d_rho_i_t, term = self.fisher_d_rho_i_t(self.c_ops[index], c_ops_dif[index][i], self.sc_ops[index], sc_ops_dif[index][i], 
                                                  self.H[index], H_dif[index][i], self.rho[index], rho_i_t, self.dw[index])
                #print("d_rho_i_t 0", (self.sigma_z_1 * self.rho[index] + self.rho[index] * self.sigma_z_1).tr())
                rho_i_t = rho_i_t + d_rho_i_t
                rho_i_t_tr_[index + 1] = rho_i_t.tr()
                d_rho_i_t_tr_[index + 1] = d_rho_i_t.tr()
                termes[index + 1] = term
            rho_i_t_tr[:, i] = rho_i_t_tr_
            d_rho_i_t_tr[:, i] = d_rho_i_t_tr_

    def Kraus_mix_next_step_dif(self, M_dy, M_dy_dif, rho, rho_dif, c_ops, c_ops_dif, sc_ops_w, sc_ops_w_dif, sc_ops_c, sc_ops_c_dif, nu_w, nu_c, nu_0, shot_noise, dN):
        """Calculates the next values with the Kraus methode for mixed detection for the fisher estimation"""
        index = np.where(dN == 1)[0][0]
        
        if dN[-1] == 1:
            rho_tilde = rho
            rho_tilde_dif = rho_dif
        else:
            rho_tilde = self.detection_counting_rho(rho, sc_ops_c[index], nu_c[index], shot_noise[index])
            rho_tilde_dif = self.detection_counting_rho_dif(rho, rho_dif, sc_ops_c[index], sc_ops_c_dif[index], nu_c[index], shot_noise[index])

        
        rho_new = M_dy_dif * rho_tilde * M_dy.dag() + M_dy * rho_tilde_dif * M_dy.dag() + M_dy * rho_tilde * M_dy_dif.dag() +(
                  self.kraus_c_ops_dif(rho_tilde, rho_tilde_dif, nu_0, c_ops, c_ops_dif, self.c_ops_numbers)
                + self.kraus_c_ops_dif(rho_tilde, rho_tilde_dif, nu_w, sc_ops_w, sc_ops_w_dif, self.sc_ops_w_numbers)
                + self.kraus_c_ops_dif(rho_tilde, rho_tilde_dif, nu_c, sc_ops_c, sc_ops_c_dif, self.sc_ops_w_numbers)) * self.dt
        

        
        return rho_new
    
    def detection_counting_rho_dif(self, rho, rho_dif, sc_ops_c, sc_ops_c_dif, nu_c, shot_noise):
        rho_tilde_dif = shot_noise * rho_dif + nu_c * (sc_ops_c_dif * rho * sc_ops_c.dag() + sc_ops_c * rho_dif * sc_ops_c.dag() + sc_ops_c * rho * sc_ops_c_dif.dag())
        norm_factor = shot_noise + nu_c * (sc_ops_c * rho * sc_ops_c.dag()).tr()
        rho_tilde_dif = rho_tilde_dif / norm_factor

        return rho_tilde_dif












def solve_fisher_parrallel(self, c_ops_dif,  sc_ops_dif, H_dif, N_parrallel):
    """Solves the stocastic master equation with detection"""
        
    pool = multiprocessing.Pool(N_parrallel)


    rho_i_t_tr = np.zeros((N_t, parm_number), dtype=complex)
    d_rho_i_t_tr = np.zeros((N_t, parm_number), dtype=complex)
    termes = np.zeros((N_t, 1 + c_ops_numbers + 2 * sc_ops_numbers), dtype=complex)


    out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
        
    return rho_i_t_tr, d_rho_i_t_tr, termes

           
def fisher_calculator(parm_number, N_t, N_fisher, rho_t_tr):
    """Calculates the Fisher information from rho_i"""
    fisher_information = np.zeros((parm_number, parm_number, N_t, N_fisher), dtype=complex) 
    for k in range(parm_number):
        for l in range(parm_number):    
            fisher_information[k, l] = rho_t_tr[k] * rho_t_tr[l]   

    fisher_information_mean = np.mean(fisher_information, axis=3)
    fisher_information_std = np.std(fisher_information, axis=3) / np.sqrt(N_fisher)
    return fisher_information_mean, fisher_information_std


class Experiment_estimation(PQSSolver): 
    def __init__(self, H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, N_states, markov_matrix, index_list, e_ops, phi=0, detector_effecency=None, timedependent_H=False, debug=False):
        
        super().__init__(H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, phi, detector_effecency, timedependent_H, debug=debug)
        self.N_states = N_states
        self.index_list = index_list
        self.markov_matrix = markov_matrix
        self.P_n_rho = np.zeros((self.N_t, self.N_states), dtype=complex)
        self.P_n_E = np.zeros((self.N_t, self.N_states), dtype=complex)
        self.P_n_PQS = np.zeros((self.N_t, self.N_states), dtype=complex)

    def import_expectation_values(self, expectation_values):
        self.exp = expectation_values
    
    def normelization_individual(self, d_rho, rho):
        rho_new = []
        tr = 0
        for i in range(self.N_states):
            rho_new.append(rho[i] + d_rho[i])
            tr_1 = rho_new[i].tr()
            tr += tr_1
            rho_new[i] = rho_new[i] / tr_1
        return rho_new
        

    def normelization(self, d_rho, rho):
        rho_new = []
        tr = 0
        for i in range(self.N_states):
            rho_new.append(rho[i] + d_rho[i])
            tr_1 = rho_new[i].tr()
            tr += tr_1
        
        for i in range(self.N_states):    
            rho_new[i] = rho_new[i] / tr
        return rho_new
    
    def normelization_Kraus(self, rho_new):
        tr = 0
        for i in range(self.N_states):
            tr += rho_new[i].tr()
        
        for i in range(self.N_states):    
            rho_new[i] = rho_new[i] / tr
        return rho_new
        

    def calculate_expectation_values(self, rho, e_ops, index):
        """Calculates the expectation value of the output oporator"""
        
        for i in range(self.e_ops_numbers):
            self.expectation[index][i] = 0
            for j in range(self.N_states):
                self.expectation[index][i] += self.expectation_value(e_ops[j][i], rho[j])
        

    def func_E(self, H, E, E_full, c_ops, sc_ops, dw, split=False):
        """The function to be solved"""

        
        Hamiltion_term = 1j * (H * E - E * H) * self.dt
            

        lindblad_contribution = self.lindblad_terms_dag(E, c_ops, self.c_ops_numbers) * self.dt
        measurment_backaction = self.lindblad_terms_dag(E, sc_ops, self.sc_ops_numbers) * self.dt

        if split:
            return Hamiltion_term + lindblad_contribution + measurment_backaction, self.backaction_terms_dag(E, E_full, sc_ops, dw)
        else:
            measurment_backaction += self.backaction_terms_dag(E, E_full, sc_ops, dw)
            return Hamiltion_term + lindblad_contribution + measurment_backaction

    def markov_oporator(self, rho, d_rho, r):
        """Calculates the markov oporator"""
        
        for i in range(self.N_states):
            for j in range(self.N_states):
                if i != j:
                    d_rho[i] += r[i, j] * rho[j] * self.dt
                    d_rho[j] += - r[i, j] * rho[j] * self.dt
            
        return d_rho

    def solve(self, dY, methode='Euler', order=1):
        """Solves the stocastic master equation with detection"""
        
        dv = np.zeros((self.N_t, self.sc_ops_w_numbers)) # it is denoted dv as it is the estimate of the winer increment dw
        self.dv = dv
        self.delta_W = np.zeros((self.N_t, self.sc_ops_w_numbers))
        self.std_vec = np.ones_like(dv) * np.sqrt(self.dt)
        
        #dv = dY
        if methode == 'Kraus':
            M_0 = []
            L = []
            for i in range(self.N_states):
                M_0_i, L_i = self.Kraus_operator(self.H[i], self.rho_0[i], self.c_ops[i], self.sc_ops_w[i], self.c_ops_numbers, self.sc_ops_w_numbers)
                M_0.append(M_0_i)
                L.append(L_i)


        for index in range(self.N_t):

            for i in range(self.sc_ops_w_numbers):
                
                dv[index][i] = self.estimate_dw(dY[index][i], self.rho[index], self.sc_ops_w, self.nu_w[i], self.N_states, i)
                #dv[index][i] = self.estimate_dw_exp(dY[index][i], self.rho[index], self.sc_ops, self.nu[i], self.N_states, i, self.exp[index][0])
                self.delta_W[index][i] = dv[index][i] - dv[index-1][i]
            
            if methode == 'Kraus':
                rho_new = []
                for i in range(self.N_states):
                    #M_0, L = self.Kraus_operator_simple(self.H[i], self.rho[index][i], self.c_ops[i], self.sc_ops[i], self.c_ops_numbers, self.sc_ops_numbers)
                    M_dy = self.Kraus_M_dy(M_0[i], L[i], self.nu_w, dY[index], self.sc_ops_w_numbers)
                    rho_new.append(self.Kraus_next_step(self.rho[index][i], self.nu_w, M_dy, L[i]))
                rho_new = self.markov_oporator(self.rho[index], rho_new, self.markov_matrix)
                self.rho.append(self.normelization_Kraus(rho_new))
            else:
                d_rho = []
                for i in range(self.N_states):
                    d_rho.append(self.next_step(dv[index], self.H[i], self.rho[index][i], self.rho[index], 
                                                        self.c_ops[i], self.sc_ops_w[i], methode=methode, order=order))
                d_rho = self.markov_oporator(self.rho[index], d_rho, self.markov_matrix)
                self.rho.append(self.normelization(d_rho, self.rho[index]))
            
            

            
            #self.rho.append(self.normelization(d_rho, self.rho[index]))
            self.calculate_expectation_values(self.rho[index], self.e_ops, index)
            self.probability_rho(index)
        
        
        self.dv_rho = dv

        self.detection_record = dY
        

        


    def solve_E(self, dY, methode='Euler', order=1):
        """Solves the stocastic master equation for the effect matrix with detection"""
        # The signal type should always be full, as it is only relevant when estimating
        
        self.E = []
        E_0 = qt.qeye_like(self.rho_0[0])
        E_0 = E_0 / E_0.tr() / self.N_states
        E_HMM = []
        for i in range(self.N_states):
            E_HMM.append(E_0)
        self.E.append(E_HMM)

        #print('E_0 shape', np.shape(E_HMM))

        

        dv = np.zeros((self.N_t, self.sc_ops_w_numbers)) # it is denoted dv as it is the estimate of the winer increment dw
        
        for index in range(self.N_t): # Be aware that here we are indexing backwards in time, 
            index_reverse = self.N_t - index - 1 # it is E and dv that are indexed backwards in time, so all the other tings use index_reverse
            
            for i in range(self.sc_ops_w_numbers):
                dv[index][i] = self.estimate_dw(dY[index_reverse][i], self.E[index], self.sc_ops_w, self.nu_w[i], self.N_states, i) # do I need rho for the estimate?
            

            if methode == 'Kraus':
                E_new = []
                for i in range(self.N_states):
                    E_new.append(self.next_step_E(dv[index], self.H[i], self.E[index][i], self.E[index], self.c_ops[i], self.sc_ops_w[i], methode=methode, order=order))

            else:
                d_E = []
                for i in range(self.N_states):
                    d_E.append(self.next_step_E(dv[index], self.H[i], self.E[index][i], self.E[index], self.c_ops[i], self.sc_ops_w[i], methode=methode, order=order))
                d_E = self.markov_oporator(self.E[index], d_E, self.markov_matrix.T, self.index_list)
            self.E.append(self.normelization(d_E, self.E[index]))


        
        self.E = self.E[::-1] # reverse the list to get the correct order of the elements
        self.dv_E = dv # reverse the list to get the correct order of the elements
        

                
        for index in range(self.N_t):
            self.probability_E(index)

    def lindblad_terms_dag(self, E, oporators, oporator_number):
        """Calculates the lindblad operator"""

        lindblad = 0
        
        for i in range(oporator_number):
            lindblad += self.D_operator_dag(oporators[i], E)

        return lindblad

    def Euler_next_step_E(self, H, E, E_full, c_ops, sc_ops, dw):
        """Calculates the next values with the Euler methode"""

        d_E = self.func_E(H, E, E_full, c_ops, sc_ops, dw)
        
        return d_E

    def next_step_E(self, dw, H, E, E_full, c_ops, sc_ops, methode='Euler', order=1):
        """Calculates the next values with the given methode"""

        if methode == 'Euler':
            d_E = self.Euler_next_step_E(H, E, E_full, c_ops, sc_ops, dw) #working
        elif methode == 'Adam Bashforth':
            d_E = self.Adam_Bashforth_next_step_E(H, E, c_ops, sc_ops, dw, order) # broken 
        elif methode == 'Milstein':
            d_E = self.Milstein_next_step_E(H, E, E_full, c_ops, sc_ops, dw) # broken 
        elif methode == 'RK stocastic':
            d_E = self.RK_stocastic_next_step_E(H, E, c_ops, sc_ops, dw) # broken
        elif methode == 'Kraus':
            E_new = self.Kraus_next_step_E(H, E, c_ops, sc_ops, self.nu, dw)
 
        return d_E
    




    def Milstein_next_step_E(self, H, E, E_full, c_ops, sc_ops, dw):
        """Calculates the next values with the Milstein methode"""
        
        f, g = self.stocastic_split_function_E()
        d_E = f(E, H, c_ops, sc_ops) * self.dt
        for j_1 in range(self.sc_ops_numbers):
            d_E += g(E, E_full, sc_ops, j_1) * dw[j_1]
            d_E += - 1/2 * self.Lg(j_1, j_1, g, E, E_full, sc_ops) * self.dt
            for j_2 in range(self.sc_ops_numbers):
                d_E += self.Lg(j_1, j_2, g, E, E_full, sc_ops) * dw[j_1] * dw[j_2]

        return d_E
    
    def stocastic_split_function_E(self):
        """Splits the stocastic eqation dx = f(x)dt + g(x)dw in to f(x) and g(x)"""
        
        def f(E, H, c_ops, sc_ops):
            Hamiltion_term = 1j * (H * E - E * H)
            lindblad_contribution = self.lindblad_terms(E, c_ops, self.c_ops_numbers)
            measurment_lindblad = self.lindblad_terms(E, sc_ops, self.sc_ops_numbers)
            return Hamiltion_term + lindblad_contribution + measurment_lindblad
            
        def g(E, E_full, sc_ops, j):
            measurment_backaction = self.H_operator_dag(sc_ops[j], E, E_full)
            return measurment_backaction
        
        return f, g
        
    def Adam_Bashforth_next_step_E(self, dw, index, index_reverse, order):
        """Calculates the next values with the Adam Bashforth methode"""
        
        order = min(order, index)

        d_E = self.func_E(index, index_reverse, self.E[index], dw)
        coeficents = self.calculate_Adam_Bashforth_coeficents(order)

        for i in range(1, order):
            d_E += coeficents[i] * self.func_E(index - i, index_reverse - i, self.E[index - i], dw)

        return d_E
    
    def RK_stocastic_next_step_E(self, delta_dw, index, index_reverse):
        """Calculates the next values with the Sticastic Runge-Kutta methode"""
        
        a, b = self.func_E(index, index_reverse, self.rho[index], delta_dw, split=True)
        
        Y_est = self.E[index] + a + self.backaction_terms_dag(self.E[index], index, index_reverse, self.std_vec)

        delta_complicated = (self.delta_W ** 2 - self.dt) * np.sqrt(self.dt) 
        

        d_rho = a + b + 1/2 * (self.backaction_terms_dag(Y_est, index, index_reverse, delta_complicated) - self.backaction_terms_dag(self.rho[index], index, index_reverse, delta_complicated))

        return d_rho

    def extract_rho_n(self, rho, n):
        """Extracts the density matrix of the qubit from the density matrix of the qubit and HMM"""

        if self.N_states == 1:
            return rho

        rho_n = rho[n]
        return rho_n
    
    def extract_E_n(self, E, n):
        """Extracts the E of the qubit from the E matrix of the qubit and HMM"""

        if self.N_states == 1:
            return E
        
        E_n = E[n]
        return E_n
    
    def probability_rho(self, index):
        """Calculates the probability of the HMM states given the density matrix"""
        for i in range(self.N_states):
            p_n = (self.extract_rho_n(self.rho[index], i)).tr()
            self.P_n_rho[index, i] = p_n
    
    def probability_E(self, index):
        """Calculates the probability of the HMM states given the effects matrix"""
        for i in range(self.N_states):
            p_n = (self.extract_E_n(self.E[index], i)).tr()
            self.P_n_E[index, i] = p_n 
    
    def probability_PQS(self, index):
        """Calculates the probability of the HMM states given both the density and effects matrix"""

        for i in range(self.N_states):
            p_n = (self.extract_rho_n(self.rho[index], i) * self.extract_E_n(self.E[index], i)).tr()
            self.P_n_PQS[index, i] = p_n
        norm_factor = np.sum(self.P_n_PQS[index])
        self.P_n_PQS[index] = self.P_n_PQS[index] / norm_factor

    def estimate_dw(self, dY, rho, sc, nu, N_states, sc_index):
        """Calculates the estimate of the stocastic dw from the output signal"""
        exp = 0
        for i in range(N_states):
            exp += self.expectation_value(sc[i][sc_index] + sc[i][sc_index].dag(), rho[i])
        #print('sc', sc[0][sc_index])
        #print('rho', rho[0])
        #exp = self.expectation_value(sc[0][sc_index] + sc[0][sc_index].dag(), rho[0])

        dw = dY / np.sqrt(nu)  - np.sqrt(nu) * exp * self.dt
        return dw
    
    def estimate_dw_exp(self, dY, rho, sc, nu, N_states, sc_index, exp):
        """Calculates the estimate of the stocastic dw from the output signal"""
        #exp = 0
        #for i in range(N_states):
        #    exp += self.expectation_value(sc[i][sc_index] + sc[i][sc_index].dag(), rho[i])
        #print('sc', sc[0][sc_index])
        #print('rho', rho[0])
        #exp = self.expectation_value(sc[0][sc_index] + sc[0][sc_index].dag(), rho[0])

        dw = dY / np.sqrt(nu)  - np.sqrt(nu) * exp * self.dt
        return dw

    

    def solve_PQS(self, dY, methode='Euler', order=1):
        """Solves the stocastic master equation for the density matrix and the effects matrix with detection record"""

        

        self.solve(dY, methode=methode, order=order)
        self.solve_E(dY, methode=methode, order=order)

        
        for index in range(self.N_t):
            self.probability_PQS(index)
    
    def solve_PQS_multi(self, dY, methode='Euler', order=1):
        """Solves the stocastic master equation for the density matrix and the effects matrix with detection record"""

        
        
        forward = threading.Thread(target=self.solve, args=(dY, methode, order,))
        
        backward = threading.Thread(target=self.solve_E, args=(dY, methode, order,))
        
        forward.start()
        backward.start()

        forward.join()
        backward.join() 

            
        for index in range(self.N_t):
            self.probability_PQS(index)
    
    
    




    def run_parralel_statistics(function, arguments, N_t):
    
        worker_1 = threading.Thread(target=function, args=arguments)
        worker_2 = threading.Thread(target=function, args=arguments)
        worker_3 = threading.Thread(target=function, args=arguments)
        worker_4 = threading.Thread(target=function, args=arguments)
        worker_5 = threading.Thread(target=function, args=arguments)
        worker_6 = threading.Thread(target=function, args=arguments)
        worker_7 = threading.Thread(target=function, args=arguments)
        worker_8 = threading.Thread(target=function, args=arguments)

        for i in range(int(N_t/8)):
            worker_1.start()
            worker_2.start()
            worker_3.start()
            worker_4.start()
            worker_5.start()
            worker_6.start()
            worker_7.start()
            worker_8.start()

            worker_1.join()
            worker_2.join()
            worker_3.join()
            worker_4.join()
            worker_5.join()
            worker_6.join()
            worker_7.join()
            worker_8.join()
            




