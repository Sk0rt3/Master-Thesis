import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plot_estimate_hist(time, estimate_signal, True_signal, N_states, title, save=False, filename='test', cmap='plasma'):
    #fig, ax = plt.subplots(figsize=(16, 8))
    figure(figsize=(24, 4))
    c = plt.imshow(estimate_signal.T, aspect='auto', extent=[0, 
                    time[-1], -1/2, N_states -1/2], origin='lower', 
                    cmap=cmap, vmin=0, vmax=1, interpolation='none')
    plt.plot(time, True_signal, label='True', color='red')
    plt.xlabel('$\gamma t$')
    plt.ylabel('$\Omega / \gamma$')
    plt.legend()
    plt.colorbar(c)
    plt.title(title)
    if save:
        file_path =  + f'{filename}.png'
        plt.savefig(file_path)
    plt.show()

class PQSSolver:
    def __init__(self, H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, detector_effecency_w=None, detector_effecency_c=None, shot_noise=None):
        """
        H: Hamiltonian as a function of time
        rho_0: Initial density matrix with qutip format
        times: time array numpy array
        c_ops: list of collapse operators as a function of time
        sc_ops_w: list of jump operators for the wieiner prosess as a function of time
        sc_ops_c: list of jump operators for the counting prosess as a function of time
        e_ops: list of expectation value operators as a function of time
        detector_effecency_w: list of detector effecency for the wieiner prosess
        detector_effecency_c: list of detector effecency for the counting prosess
        shot_noise: list of shot noise for the counting prosess
        
        Initializes the solver"""
        
        self.H = H
        self.rho_0 = rho_0 
        self.rho_0_size = rho_0
        self.times = times
        self.N_t = len(times)
        self.dt = times[1] - times[0]
        self.index = 0
        self.c_ops = c_ops
        self.c_ops_numbers = len(c_ops(0))
        self.sc_ops_w = sc_ops_w #wieiner prosess
        self.sc_ops_w_numbers = len(sc_ops_w(0))
        self.sc_ops_c = sc_ops_c #jump prosess prosess
        self.sc_ops_c_numbers = len(sc_ops_c(0))
        self.e_ops_numbers = len(e_ops(0))
        self.rho = []#np.zeros((self.N_t, *np.shape(rho_0)), dtype=complex)
        self.rho.append(self.rho_0)
        self.expectation = np.zeros((self.N_t, self.e_ops_numbers), dtype=complex)
        self.e_ops = e_ops
        self.system_size = 0#int(rho_0.shape[0])  ### broken
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

    def calculate_expectation_values(self, rho, e_ops, index):
        """Calculates the expectation value of the output oporator"""
        
        for i in range(self.e_ops_numbers):
            self.expectation[index][i] = self.expectation_value(e_ops[i], rho)

    def expectation_value(self, c, rho):
        """Calculates the expectation value of the output oporator"""

        expectation = ((c * rho)).tr()
        return expectation
    
    def Kraus_M_dy_mix(self, H, c_ops, sc_ops_w, sc_ops_c, nu_w, dY):
        """Calculates the Kraus operator for the case where there boath is a jump and a wieiner prosess"""
        M_dy = qt.qeye_like(self.rho_0_size) - 1j * H * self.dt
       

        for i in range(self.c_ops_numbers):
            M_dy += -1/2 * c_ops[i].dag() * c_ops[i] * self.dt
        

        for i in range(self.sc_ops_w_numbers):
            M_dy += -1/2 * sc_ops_w[i].dag() * sc_ops_w[i] * self.dt
        
        
        for i in range(self.sc_ops_c_numbers):
            M_dy += -1/2 * sc_ops_c[i].dag() * sc_ops_c[i] * self.dt
        
        
        for i in range(self.sc_ops_w_numbers):
            M_dy += np.sqrt(nu_w[i]) * dY[i] * sc_ops_w[i]
        

        return M_dy
    
    def Kraus_mix_next_step(self, M_dy, rho, c_ops, sc_ops_w, sc_ops_c, nu_w, nu_c, nu_0, shot_noise, dN, c_ops_numbers, sc_ops_w_numbers, sc_ops_c_numbers):
        """Calculates the next values with the Kraus methode"""
        index = np.where(dN == 1)[0][0]
        
        if dN[-1] == 1:
            rho_tilde = rho
            norm_factor = 1
        else:
            rho_tilde, norm_factor = self.detection_counting_rho(rho, sc_ops_c[index], nu_c[index], shot_noise[index])

        
        rho_new = M_dy * rho_tilde * M_dy.dag() + (self.kraus_c_ops(rho_tilde, nu_0, c_ops, c_ops_numbers) 
                                           + self.kraus_c_ops(rho_tilde, nu_w, sc_ops_w, sc_ops_w_numbers)
                                           + self.kraus_c_ops(rho_tilde, nu_c, sc_ops_c, sc_ops_c_numbers)) * self.dt

        return rho_new, norm_factor
    
    def kraus_c_ops(self, rho, nu, L, L_number):
        """Calculates the kraus superoperator"""
        L_rho = 0
        for i in range(L_number):
            L_rho += (1 - nu[i]) * L[i] * rho * L[i].dag()
        
        return L_rho
    
    def detection_counting_rho(self, rho, sc_ops_c, nu_c, shot_noise):
        """Calculates the density matrix for counting"""
        
        rho_tilde = shot_noise * rho + nu_c * sc_ops_c * rho * sc_ops_c.dag()
        norm_factor = shot_noise + nu_c * (sc_ops_c * rho * sc_ops_c.dag()).tr()
        rho_tilde = rho_tilde / norm_factor
        
        return rho_tilde, norm_factor
    
    



        

class Experiment_simulation(PQSSolver):
    def __init__(self, H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, detector_effecency_w=None, detector_effecency_c=None, shot_noise=None):
        super().__init__(H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, detector_effecency_w, detector_effecency_c, shot_noise=shot_noise)
        
        

    def solve_mixed(self):
        """solves the stocastic master equation with a mix of counting and homodyne detection"""
        dw = self.gennerate_dw()
        self.dw = dw
        dY = np.zeros((self.N_t, self.sc_ops_w_numbers))
        dN = np.zeros((self.N_t, self.sc_ops_c_numbers + 1))
        self.normfactor = np.zeros(self.N_t)
        self.normfactor_counting = np.zeros(self.N_t)
        self.calculate_expectation_values(self.rho[0], self.e_ops(0), 0)
        for index in range(self.N_t-1):
            dN[index] = self.calculate_dN(self.rho[index], self.sc_ops_c(index))
            dY[index] = self.gennerate_outputsignal_mixed(dw[index], self.rho[index], self.sc_ops_w(index), self.nu_w)
            M_dy = self.Kraus_M_dy_mix(self.H(index), self.c_ops(index), self.sc_ops_w(index), self.sc_ops_c(index), self.nu_w, dY[index])
            rho_new, normfactor_count = self.Kraus_mix_next_step(M_dy, self.rho[index], self.c_ops(index), self.sc_ops_w(index), self.sc_ops_c(index), self.nu_w, self.nu_c, self.nu_0, self.shot_noise, dN[index], self.c_ops_numbers, self.sc_ops_w_numbers, self.sc_ops_c_numbers)
            if index > 0:
                self.normfactor_counting[index] = normfactor_count * self.normfactor_counting[index-1]
            else:
                self.normfactor_counting[index] = normfactor_count
            self.normelization_Kraus_mixed(rho_new, index)
            self.calculate_expectation_values(self.rho[index + 1], self.e_ops(index + 1), index + 1)
        dN[self.N_t - 1] = self.calculate_dN(self.rho[self.N_t - 1], self.sc_ops_c(self.N_t - 1))
        dY[self.N_t - 1] = self.gennerate_outputsignal_mixed(dw[self.N_t - 1], self.rho[self.N_t - 1], self.sc_ops_w(self.N_t - 1), self.nu_w)

        


        self.detection_record = dY
        self.dw = dw
        self.dY = dY
        self.dN = dN
    

    def gennerate_dw(self):
        """Generates the stocastic dw"""
        dw = np.random.normal(0, np.sqrt(self.dt) , (self.N_t, self.sc_ops_w_numbers))
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

    def gennerate_outputsignal_mixed(self, dw, rho, sc_ops_w, nu):
        """Calculates the output signal for all the detections form winer prosses measurments"""
        
        dY = np.zeros(self.sc_ops_w_numbers)
        for i in range(self.sc_ops_w_numbers):
            dY[i] = np.sqrt(nu[i]) * dw[i] + nu[i] * self.expectation_value(sc_ops_w[i] + sc_ops_w[i].dag(), rho) * self.dt
        
        return dY
    

    def reset(self):
        self.rho = []
        self.rho.append(self.rho_0)
        self.expectation = np.zeros((self.N_t, self.e_ops_numbers), dtype=complex)
        self.index = 0
        self.normfactor = np.zeros(self.N_t)
        self.normfactor_counting = np.zeros(self.N_t)

    def normelization_Kraus_mixed(self, rho_new, index):
        norm_factor_rest = rho_new.tr()
        rho_new = rho_new / norm_factor_rest
        self.rho.append(rho_new)
        if index == 0:
            self.normfactor[index] = norm_factor_rest
        else:
            self.normfactor[index] = self.normfactor[index - 1] * norm_factor_rest
        



class Experiment_Fisher_estimation(Experiment_simulation):
    def __init__(self, H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, peram_number, c_ops_dif, H_dif, sc_ops_w_dif, sc_ops_c_dif, detector_effecency_c=None, detector_effecency_w=None, shot_noise=None):
        super().__init__(H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, detector_effecency_c=detector_effecency_c, detector_effecency_w=detector_effecency_w, shot_noise=shot_noise)
        self.parm_number = peram_number
        self.c_ops_dif = c_ops_dif
        self.sc_ops_w_dif = sc_ops_w_dif
        self.sc_ops_c_dif = sc_ops_c_dif
        self.H_dif = H_dif

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

    def solve_fisher_tr_mixed(self, c_ops, c_ops_dif, sc_ops_w, sc_ops_w_dif, sc_ops_c, sc_ops_c_dif, H, H_dif, dY, dN):
        """Solves the stocastic equation for fisher information with mixed detection"""

        rho_i_t_tr = np.zeros((self.N_t, self.parm_number), dtype=complex)
        rho_unnorm = []
        for i in range(self.N_t):
            rho_unnorm.append(self.rho[i] * (self.normfactor[i]) * self.normfactor_counting[i])
        

        for i in range(self.parm_number):
            rho_it = qt.qzero_like(self.rho_0)
            rho_it_tr_ = np.zeros(self.N_t, dtype=complex)
            
            for index in range(self.N_t-1):
                M_dy_dif = self.Kraus_M_dy_mix_dif(H_dif(index, i), c_ops(index), c_ops_dif(index, i), sc_ops_w(index), sc_ops_w_dif(index, i), sc_ops_c(index), sc_ops_c_dif(index, i), dY[index])
                M_dy = self.Kraus_M_dy_mix(H(index), c_ops(index), sc_ops_w(index), sc_ops_c(index), self.nu_w, dY[index])
                rho_it = self.Kraus_mix_next_step_dif(M_dy, M_dy_dif, rho_unnorm[index], rho_it, c_ops(index), c_ops_dif(index, i), sc_ops_w(index), sc_ops_w_dif(index, i), sc_ops_c(index), sc_ops_c_dif(index, i), self.nu_w, self.nu_c, self.nu_0, self.shot_noise, dN[index])
                #rho_it = rho_it / (self.normfactor[index]) #* self.normfactor_counting[index])
                rho_it_tr_[index + 1] = rho_it.tr() / (self.normfactor[index] * self.normfactor_counting[index])
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

    def Kraus_mix_next_step_dif(self, M_dy, M_dy_dif, rho, rho_dif, c_ops, c_ops_dif, sc_ops_w, sc_ops_w_dif, sc_ops_c, sc_ops_c_dif, nu_w, nu_c, nu_0, shot_noise, dN):
        """Calculates the next values with the Kraus methode for mixed detection for the fisher estimation"""
        index = np.where(dN == 1)[0][0]
        
        
        if dN[-1] == 1:
            rho_tilde = rho
            rho_tilde_dif = rho_dif
        else:
            rho_tilde = self.detection_counting_rho_fisher(rho, sc_ops_c[index], nu_c[index], shot_noise[index])
            rho_tilde_dif = self.detection_counting_rho_dif(rho, rho_dif, sc_ops_c[index], sc_ops_c_dif[index], nu_c[index], shot_noise[index])


        
        rho_new = M_dy_dif * rho_tilde * M_dy.dag() + M_dy * rho_tilde_dif * M_dy.dag() + M_dy * rho_tilde * M_dy_dif.dag() +(
                  self.kraus_c_ops_dif(rho_tilde, rho_tilde_dif, nu_0, c_ops, c_ops_dif, self.c_ops_numbers)
                + self.kraus_c_ops_dif(rho_tilde, rho_tilde_dif, nu_w, sc_ops_w, sc_ops_w_dif, self.sc_ops_w_numbers)
                + self.kraus_c_ops_dif(rho_tilde, rho_tilde_dif, nu_c, sc_ops_c, sc_ops_c_dif, self.sc_ops_w_numbers)) * self.dt
        

        
        return rho_new
    

    def kraus_c_ops_dif(self, rho, rho_dif, nu, L, L_dif, L_number):
        """Calculates the differential kraus superoperator"""
        L_rho_dif = 0
        for i in range(L_number):
            L_rho_dif += (1 - nu[i]) * (L_dif[i] * rho * L[i].dag() + L[i] * rho_dif * L[i].dag() + L[i] * rho * L_dif[i].dag())
        
        return L_rho_dif
    
    def detection_counting_rho_dif(self, rho, rho_dif, sc_ops_c, sc_ops_c_dif, nu_c, shot_noise):
        rho_tilde_dif = shot_noise * rho_dif + nu_c * (sc_ops_c_dif * rho * sc_ops_c.dag() + sc_ops_c * rho_dif * sc_ops_c.dag() + sc_ops_c * rho * sc_ops_c_dif.dag())
        #norm_factor = shot_noise + nu_c * (sc_ops_c * rho * sc_ops_c.dag()).tr()
        #rho_tilde_dif = rho_tilde_dif / norm_factor

        return rho_tilde_dif
    
    def detection_counting_rho_fisher(self, rho, sc_ops_c, nu_c, shot_noise):
        """Calculates the density matrix for counting"""
        
        rho_tilde = shot_noise * rho + nu_c * sc_ops_c * rho * sc_ops_c.dag()
        rho_tilde = rho_tilde
        
        return rho_tilde
        



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
    def __init__(self, H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, N_states, markov_matrix, detector_effecency_w=None, detector_effecency_c=None, shot_noise=None):
        super().__init__(H, rho_0, times, c_ops, sc_ops_w, sc_ops_c, e_ops, detector_effecency_w, detector_effecency_c, shot_noise=shot_noise)
        self.N_states = N_states
        self.rho_0_size = rho_0[0]
        

        #self.index_list = index_list
        self.markov_matrix = markov_matrix
        self.P_n_rho = np.zeros((self.N_t, self.N_states), dtype=complex)
        self.P_n_E = np.zeros((self.N_t, self.N_states), dtype=complex)
        self.P_n_PQS = np.zeros((self.N_t, self.N_states), dtype=complex)
    
    def solve_mixed(self, dY, dN):
        """solves the stocastic master equation with a mix of counting and homodyne detection"""
        
        
        for index in range(self.N_t-1):
            rho_new = []
            if dN[index][-1] == 1:
                norm_factor = 1
            else:
                dN_index = np.where(dN[index] == 1)[0][0]
                norm_factor = self.calculated_norm_factor_counting(self.rho[index], self.sc_ops_c(index)[dN_index], self.nu_c[dN_index], self.shot_noise[dN_index])
            for i in range(self.N_states):
                M_dy = self.Kraus_M_dy_mix(self.H(i), self.c_ops(index), self.sc_ops_w(index), self.sc_ops_c(index), self.nu_w, dY[index])
                rho_new.append(self.Kraus_mix_next_step(M_dy, self.rho[index][i], self.c_ops(index), self.sc_ops_w(index), self.sc_ops_c(index), self.nu_w, self.nu_c, 
                                                        self.nu_0, self.shot_noise, dN[index], self.c_ops_numbers, self.sc_ops_w_numbers, self.sc_ops_c_numbers, norm_factor))
            rho_new = self.markov_oporator(self.rho[index], rho_new, self.markov_matrix)
            self.rho.append(self.PQS_normelization(rho_new))
            #self.calculate_expectation_values(self.rho[index + 1], self.e_ops(index + 1), index + 1)
        
            self.probability_rho(index)

    def probability_rho(self, index):
        """Calculates the probability of the HMM states given the density matrix"""
        for i in range(self.N_states):
            p_n = (self.extract_rho_n(self.rho[index], i)).tr()
            self.P_n_rho[index, i] = p_n

    def markov_oporator(self, rho, d_rho, r):
        """Calculates the markov oporator"""
        
        for i in range(self.N_states):
            for j in range(self.N_states):
                if i != j:
                    d_rho[i] += r[i, j] * rho[j] * self.dt
                    d_rho[j] += - r[i, j] * rho[j] * self.dt
            
        return d_rho
    
    def PQS_normelization(self, rho_new):
        """Normalizes the PQS states"""
        norm_factor = 0
        for i in range(self.N_states):
            norm_factor += rho_new[i].tr()

        for i in range(self.N_states):
            rho_new[i] = rho_new[i] / norm_factor
        
        return rho_new
    
    def extract_rho_n(self, rho, n):
        """Extracts the density matrix of the qubit from the density matrix of the qubit and HMM"""

        if self.N_states == 1:
            return rho

        rho_n = rho[n]
        return rho_n
    
    def detection_counting_rho(self, rho, sc_ops_c, nu_c, shot_noise, norm_factor):
        
        rho_tilde = shot_noise * rho + nu_c * sc_ops_c * rho * sc_ops_c.dag()
        #norm_factor = shot_noise + nu_c * (sc_ops_c * rho * sc_ops_c.dag()).tr()
        rho_tilde = rho_tilde / norm_factor
        
        return rho_tilde
    
    def calculated_norm_factor_counting(self, rho, sc_ops_c, nu_c, shot_noise):
        norm_factor = 0
        for i in range(self.N_states):
            norm_factor += shot_noise + nu_c * (sc_ops_c * rho[i] * sc_ops_c.dag()).tr()
        return norm_factor
    
    def calculated_norm_factor_counting_E(self, E, sc_ops_c, nu_c, shot_noise):
        norm_factor = 0
        for i in range(self.N_states):
            norm_factor += shot_noise + nu_c * (sc_ops_c.dag() * E[i] * sc_ops_c).tr()
        return norm_factor
    
    def Kraus_mix_next_step(self, M_dy, rho, c_ops, sc_ops_w, sc_ops_c, nu_w, nu_c, nu_0, shot_noise, dN, c_ops_numbers, sc_ops_w_numbers, sc_ops_c_numbers, norm_factor):
        """Calculates the next values with the Kraus methode"""
        index = np.where(dN == 1)[0][0]
        
        if dN[-1] == 1:
            rho_tilde = rho
        else:
            rho_tilde = self.detection_counting_rho(rho, sc_ops_c[index], nu_c[index], shot_noise[index], norm_factor)

        
        rho_new = M_dy * rho_tilde * M_dy.dag() + (self.kraus_c_ops(rho_tilde, nu_0, c_ops, c_ops_numbers) 
                                           + self.kraus_c_ops(rho_tilde, nu_w, sc_ops_w, sc_ops_w_numbers)
                                           + self.kraus_c_ops(rho_tilde, nu_c, sc_ops_c, sc_ops_c_numbers)) * self.dt

        return rho_new
    
    def Kraus_mix_next_step_E(self, M_dy, E, c_ops, sc_ops_w, sc_ops_c, nu_w, nu_c, nu_0, shot_noise, dN, c_ops_numbers, sc_ops_w_numbers, sc_ops_c_numbers, norm_factor):
        """Calculates the next values with the Kraus methode"""
       
        index = np.where(dN == 1)[0][0]
        
        if dN[-1] == 1:
            E_tilde = E
        else:
            E_tilde = self.detection_counting_E(E, sc_ops_c[index], nu_c[index], shot_noise[index], norm_factor)

        
        E_new = M_dy * E_tilde * M_dy.dag()    + (self.kraus_c_ops_E(E_tilde, nu_0, c_ops, c_ops_numbers) 
                                           + self.kraus_c_ops_E(E_tilde, nu_w, sc_ops_w, sc_ops_w_numbers)
                                           + self.kraus_c_ops_E(E_tilde, nu_c, sc_ops_c, sc_ops_c_numbers)) * self.dt

        return E_new


    def solve_mixed_E(self, dY, dN):
        """solves the stocastic master equation with a mix of counting and homodyne detection"""
        
        self.E = []
        E_0 = qt.qeye_like(self.rho_0[0])
        E_0 = E_0 / E_0.tr() / self.N_states
        
        
        E_HMM = []
        for i in range(self.N_states):
            E_HMM.append(E_0)
        self.E.append(E_HMM)
        



        for index in range(self.N_t-1):
            reversed_index = self.N_t - index - 1
            E_new = []
            if dN[reversed_index][-1] == 1:
                norm_factor = 1
            else:
                dN_index = np.where(dN[reversed_index] == 1)[0][0]
                
                norm_factor = self.calculated_norm_factor_counting_E(self.E[index], self.sc_ops_c(reversed_index)[dN_index], self.nu_c[dN_index], self.shot_noise[dN_index])
            for i in range(self.N_states):
                M_dy = self.Kraus_M_dy_mix_E(self.H(i), self.c_ops(reversed_index), self.sc_ops_w(reversed_index), self.sc_ops_c(reversed_index), self.nu_w, dY[reversed_index])
                E_new.append(self.Kraus_mix_next_step_E(M_dy, self.E[index][i], self.c_ops(reversed_index), self.sc_ops_w(reversed_index), self.sc_ops_c(reversed_index), 
                                                        self.nu_w, self.nu_c, self.nu_0, self.shot_noise, dN[reversed_index], self.c_ops_numbers, self.sc_ops_w_numbers, 
                                                        self.sc_ops_c_numbers, norm_factor))
            E_new = self.markov_oporator(self.E[index], E_new, self.markov_matrix)
            self.E.append(self.PQS_normelization(E_new))
            #self.calculate_expectation_values(self.rho[index + 1], self.e_ops(index + 1), index + 1)
        
        self.E = self.E[::-1]

        for index in range(self.N_t):
            self.probability_E(index)

    def Kraus_M_dy_mix_E(self, E, c_ops, sc_ops_w, sc_ops_c, nu_w, dY):
        """Calculates the Kraus operator for the case where there boath is a jump and a wieiner prosess"""
        M_dy = qt.qeye_like(self.rho_0_size) + 1j * E * self.dt
       

        for i in range(self.c_ops_numbers):
            M_dy += -1/2 * c_ops[i].dag() * c_ops[i] * self.dt
        

        for i in range(self.sc_ops_w_numbers):
            M_dy += -1/2 * sc_ops_w[i].dag() * sc_ops_w[i] * self.dt
        
        
        for i in range(self.sc_ops_c_numbers):
            M_dy += -1/2 * sc_ops_c[i].dag() * sc_ops_c[i] * self.dt
        
        
        for i in range(self.sc_ops_w_numbers):
            M_dy += np.sqrt(nu_w[i]) * dY[i] * sc_ops_w[i].dag()
        

        return M_dy
    
    def probability_E(self, index):
        """Calculates the probability of the HMM states given the effects matrix"""
        for i in range(self.N_states):
            p_n = (self.extract_E_n(self.E[index], i)).tr()
            self.P_n_E[index, i] = p_n 

    def kraus_c_ops_E(self, E, nu, L, L_number):
        """Calculates the kraus superoperator"""
        L_rho = 0
        for i in range(L_number):
            L_rho += (1 - nu[i]) * L[i].dag() * E * L[i]
        
        return L_rho
    
    def detection_counting_E(self, E, sc_ops_c, nu_c, shot_noise, norm_factor):
        
        rho_tilde = shot_noise * E + nu_c * sc_ops_c.dag() * E * sc_ops_c
        #norm_factor = shot_noise + nu_c * (sc_ops_c * rho * sc_ops_c.dag()).tr()
        rho_tilde = rho_tilde / norm_factor
        
        return rho_tilde
    
    def extract_E_n(self, E, n):
        """Extracts the density matrix of the qubit from the density matrix of the qubit and HMM"""

        if self.N_states == 1:
            return E

        E_n = E[n]
        return E_n
    
    def solve_mixed_PQS(self, dY, dN):
        """Solves the stocastic master equation for the density matrix and the effects matrix with detection record"""

        

        self.solve_mixed(dY, dN)
        self.solve_mixed_E(dY, dN)

        
        for index in range(self.N_t):
            self.probability_PQS(index)

    def probability_PQS(self, index):
        """Calculates the probability of the HMM states given both the density and effects matrix"""

        for i in range(self.N_states):
            p_n = (self.extract_rho_n(self.rho[index], i) * self.extract_E_n(self.E[index], i)).tr()
            self.P_n_PQS[index, i] = p_n
        norm_factor = np.sum(self.P_n_PQS[index])
        self.P_n_PQS[index] = self.P_n_PQS[index] / norm_factor


class Quantum_Fisher_Information(PQSSolver):
    def __init__(self, H, rho_0, times, c_ops):
        self.H = H
        self.rho_0_theta = rho_0
        self.times_theta = times
        self.c_ops = c_ops
        self.c_ops_numbers = len(c_ops(0))
        self.sc_ops_c_numbers = 0
        self.sc_ops_w_numbers = 0
        self.e_ops_numbers = 0
        self.sc_ops_c = 0
        self.sc_ops_w = 0
        self.e_ops = 0



        

        

    def calculate_quantum_fisher_information(self, theta_1, theta_2, theta_3):
        """Numericaly calculates the quantum fisher information"""


        H_theta_1 = self.H(theta_1)
        c_ops_theta_1 = self.c_ops(theta_1)
        super.__init__(H_theta_1, self.rho_0, self.times, c_ops_theta_1, self.sc_ops_w, self.sc_ops_c, self.e_ops)
        self.solve_mixed_2_sided()
        rho_1 = self.rho
        self.reset()

        H_theta_2 = self.H(theta_2)
        c_ops_theta_2 = self.c_ops(theta_2)
        super.__init__(H_theta_2, self.rho_0, self.times, c_ops_theta_2, self.sc_ops_w, self.sc_ops_c, self.e_ops)
        self.solve_mixed()
        rho_2 = self.rho
        self.reset()

        H_theta_3 = self.H(theta_3)
        c_ops_theta_3 = self.c_ops(theta_3)
        super.__init__(H_theta_3, self.rho_0, self.times, c_ops_theta_3, self.sc_ops_w, self.sc_ops_c, self.e_ops)
        self.solve_mixed()
        rho_3 = self.rho
        self.reset()

    def solve_mixed_2_sided(self):
        """solves the stocastic master equation with a mix of counting and homodyne detection"""
        
        for index in range(self.N_t-1):
            M_dy_left = self.Kraus_M_dy_mix_2_sided(self.H(index), self.c_ops(index), self.sc_ops_w(index), self.sc_ops_c(index))
            M_dy_rigth = self.Kraus_M_dy_mix_2_sided(self.H(index), self.c_ops(index), self.sc_ops_w(index), self.sc_ops_c(index))
            rho_new, normfactor_count = self.Kraus_mix_next_step(M_dy, self.rho[index], self.c_ops(index), self.sc_ops_w(index), self.sc_ops_c(index), self.nu_w, self.nu_c, self.nu_0, self.shot_noise, dN[index], self.c_ops_numbers, self.sc_ops_w_numbers, self.sc_ops_c_numbers)
            if index > 0:
                self.normfactor_counting[index] = normfactor_count * self.normfactor_counting[index-1]
            else:
                self.normfactor_counting[index] = normfactor_count
            self.normelization_Kraus_mixed(rho_new, index)
            self.calculate_expectation_values(self.rho[index + 1], self.e_ops(index + 1), index + 1)
        dN[self.N_t - 1] = self.calculate_dN(self.rho[self.N_t - 1], self.sc_ops_c(self.N_t - 1))
        dY[self.N_t - 1] = self.gennerate_outputsignal_mixed(dw[self.N_t - 1], self.rho[self.N_t - 1], self.sc_ops_w(self.N_t - 1), self.nu_w)

        


        
        

    def Kraus_M_dy_mix_2_sided(self, H, c_ops, sc_ops_w, sc_ops_c):
        """Calculates the Kraus operator for the case where there boath is a jump and a wieiner prosess"""
        M_dy = qt.qeye_like(self.rho_0_size) - 1j * H * self.dt
       

        for i in range(self.c_ops_numbers):
            M_dy += -1/2 * c_ops[i].dag() * c_ops[i] * self.dt
        

        for i in range(self.sc_ops_w_numbers):
            M_dy += -1/2 * sc_ops_w[i].dag() * sc_ops_w[i] * self.dt
        
        
        for i in range(self.sc_ops_c_numbers):
            M_dy += -1/2 * sc_ops_c[i].dag() * sc_ops_c[i] * self.dt
        
        


        




