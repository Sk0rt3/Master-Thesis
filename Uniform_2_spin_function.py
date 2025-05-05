
import numpy as np
import qutip as qt
import sys
sys.path.append('.')
import PQS_solver as PQS
import matplotlib.pyplot as plt
import os.path




class Two_spin_system:
    def __init__(self, name, N_m_strength=5, N_theta=1, t_gamma=100, gamma=2, dt=0.001, mu1=1, mu2=1, seed_signal=42, seed_simulation=8, 
                    t_signal_leadin=1, gamma_decay=2, gamma_phi=2, m_min=2, m_max=7, theta_min=0, theta_max=0, 
                    methode='Kraus', p=0.04, field_type='uniform', save=True, dephase_strength=0, entaglement=True, beta_1=np.array([0.1, 0, 0]), beta_2=np.array([0, 0.1, 0])):
        self.N_m_strength = N_m_strength
        self.N_theta = N_theta
        self.t_gamma = t_gamma
        self.gamma = gamma
        self.name = name
        self.dt = dt
        self.mu1 = mu1
        self.mu2 = mu2
        self.seed_signal = seed_signal
        self.seed_simulation = seed_simulation
        self.t_signal_leadin = t_signal_leadin
        self.gamma_decay = gamma_decay
        self.gamma_phi = gamma_phi
        self.m_min = m_min
        self.m_max = m_max
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.methode = methode
        self.p = p
        self.field_type = field_type
        self.save = save
        self.dephase_strength = dephase_strength
        self.entaglement = entaglement
        self.R_1 = np.array([0, 1, 0])
        self.R_2 = np.array([1, 0, 0])
        self.R_m = np.array([0, 0, 0])
        self.r_signal = np.random.default_rng(seed_signal)
        self.N_t_leadin = int(t_signal_leadin / dt)
        self.m_strength_posible = np.linspace(m_min, m_max, N_m_strength)
        self.theta_posible = np.linspace(theta_min, theta_max, N_theta)
        self.N_states = N_m_strength * N_theta
        self.t = t_gamma / gamma
        self.N_t = int(self.t / dt + 1)
        self.times = np.linspace(0, self.t, self.N_t, endpoint=True)
        self.times_gamma = self.times * gamma
        self.m_mesh, self.theta_mesh = np.meshgrid(self.m_strength_posible, self.theta_posible)
        self.m_mesh = self.m_mesh.flatten()
        self.theta_mesh = self.theta_mesh.flatten()
        self.B_1_posible = np.zeros((self.N_states, 3))
        self.B_2_posible = np.zeros((self.N_states, 3))

        self.unit_vector_x = np.array([1, 0, 0])
        self.unit_vector_y = np.array([0, 1, 0])
        self.unit_vector_z = np.array([0, 0, 1])

        self.N_dim = 2

        if self.field_type == 'dipole':
            for i in range(self.N_states):
                self.B_1_posible[i] = magnetic_field(self.R_1, rotation_matrix(self.theta_mesh[i], self.unit_vector_z) @ self.unit_vector_x * self.m_mesh[i])
                self.B_2_posible[i] = magnetic_field(self.R_2, rotation_matrix(self.theta_mesh[i], self.unit_vector_z) @ self.unit_vector_x * self.m_mesh[i])
            

        elif self.field_type == 'uniform':
            for i in range(self.N_states):
                self.B_1_posible[i] = rotation_matrix(self.theta_mesh[i], self.unit_vector_x) @ self.unit_vector_z * self.m_mesh[i] 
                self.B_2_posible[i] = rotation_matrix(self.theta_mesh[i], self.unit_vector_x) @ self.unit_vector_z * self.m_mesh[i]

        self.Delta_n_1_posible = self.B_1_posible * self.mu1
        self.Delta_n_2_posible = self.B_2_posible * self.mu2

        st = 0.0 # to regulate the transtion rate between HMM states

        self.r_HMM = gennerate_r_next_neighbour_jump(self.N_states, st)


        sigma_z = qt.jmat(self.N_dim/2 - 1/2, 'z')#qt.sig maz()# Pauli-Z operator for qubit
        sigma_x = qt.jmat(self.N_dim/2 - 1/2, 'x')#qt.sigmax() # Pauli-X operator for qubit
        sigma_y = qt.jmat(self.N_dim/2 - 1/2, 'y')#qt.sigmay() # Pauli-Y operator for qubit
        sigma_p = qt.jmat(self.N_dim/2 - 1/2, '+')#qt.sigmap() # raising operator for qubit
        sigma_m = qt.jmat(self.N_dim/2 - 1/2, '-')#qt.sigmam() # lowering operator for qubit



        self.sigma_z_1 = qt.tensor(sigma_z, qt.qeye(self.N_dim))
        self.sigma_x_1 = qt.tensor(sigma_x, qt.qeye(self.N_dim))
        self.sigma_y_1 = qt.tensor(sigma_y, qt.qeye(self.N_dim))
        #sigma_p_1 = qt.tensor(sigma_p, qt.qeye(self.N_dim))
        #sigma_m_1 = qt.tensor(sigma_m, qt.qeye(self.N_dim))

        self.sigma_z_2 = qt.tensor(qt.qeye(self.N_dim), sigma_z)
        self.sigma_x_2 = qt.tensor(qt.qeye(self.N_dim), sigma_x)
        self.sigma_y_2 = qt.tensor(qt.qeye(self.N_dim), sigma_y)
        #sigma_p_2 = qt.tensor(qt.qeye(self.N_dim), sigma_p)
        #sigma_m_2 = qt.tensor(qt.qeye(self.N_dim), sigma_m)

        state_0 = qt.tensor(qt.maximally_mixed_dm(self.N_dim), qt.maximally_mixed_dm(self.N_dim))#qt.tensor(qt.basis(N_dim, 0), qt.basis(N_dim, 0))
        self.rho_0 = state_0

        self.rho_HMM = [self.rho_0 / self.N_states for _ in range(self.N_states)]

        c_1 = lambda i : self.sigma_x_1 * dephase_strength + self.sigma_y_1 * dephase_strength + self.sigma_z_1 * dephase_strength
        c_2 = lambda i : self.sigma_x_2 * dephase_strength + self.sigma_y_2 * dephase_strength + self.sigma_z_2 * dephase_strength

        
        
        total_spin = lambda i : 1/4 * ((self.sigma_x_1 + self.sigma_y_1 + self.sigma_z_1) + (self.sigma_x_2 + self.sigma_y_2 + self.sigma_z_2)) * (self.sigma_x_1 + self.sigma_y_1 + self.sigma_z_1) + (self.sigma_x_2 + self.sigma_y_2 + self.sigma_z_2) #sigma_x_1 + sigma_x_2 + sigma_y_1 + sigma_y_2 + sigma_z_1 + sigma_z_2

        c_op = lambda i : [c_1(i), c_2(i)]
        if entaglement:
            sc_op = lambda i : [c_out_1_e(i), c_out_2_e(i)]
            e_op = lambda i : [c_out_1_e(i) + c_out_1_e_dag(i), c_out_2_e(i) + c_out_2_e_dag(i), total_spin(i)]
            c_out_1_e = lambda i : beta_1[0] * (self.sigma_x_1 + self.sigma_x_2) + beta_1[1] * (self.sigma_y_1 + self.sigma_y_2) + beta_1[2] * (self.sigma_z_1 + self.sigma_z_2)
            c_out_1_e_dag = lambda i : beta_1[0] * (self.sigma_x_1 + self.sigma_x_2).dag() + beta_1[1] * (self.sigma_y_1 + self.sigma_y_2).dag() + beta_1[2] * (self.sigma_z_1 + self.sigma_z_2).dag()
            c_out_2_e = lambda i : beta_2[0] * (self.sigma_x_1 + self.sigma_x_2) + beta_2[1] * (self.sigma_y_1 + self.sigma_y_2) + beta_2[2] * (self.sigma_z_1 + self.sigma_z_2)
            c_out_2_e_dag = lambda i : beta_2[0] * (self.sigma_x_1 + self.sigma_x_2).dag() + beta_2[1] * (self.sigma_y_1 + self.sigma_y_2).dag() + beta_2[2] * (self.sigma_z_1 + self.sigma_z_2).dag()
        else:
            c_out_1 = lambda i : beta_1[0] * self.sigma_x_1 + beta_1[1] * self.sigma_y_1 + beta_1[2] * self.sigma_z_1
            c_out_1_dag = lambda i : beta_1[0] * self.sigma_x_1.dag() + beta_1[1] * self.sigma_y_1.dag() + beta_1[2] * self.sigma_z_1.dag()
            c_out_2 = lambda i : beta_2[0] * self.sigma_x_2 + beta_2[1] * self.sigma_y_2 + beta_2[2] * self.sigma_z_2
            c_out_2_dag = lambda i : beta_2[0] * self.sigma_x_2.dag() + beta_2[1] * self.sigma_y_2.dag() + beta_2[2] * self.sigma_z_2.dag()
            sc_op = lambda i : [c_out_1(i), c_out_2(i)]
            e_op = lambda i : [c_out_1(i) + c_out_1_dag(i), c_out_2(i) + c_out_2_dag(i), total_spin(i)]


        self.c_ops = [c_op(i) for i in range(self.N_t)]
        self.c_ops_HMM = [c_op(i) for i in range(self.N_states)]

        self.sc_ops = [sc_op(i) for i in range(self.N_t)]
        self.sc_ops_HMM = [sc_op(i) for i in range(self.N_states)]

        self.e_ops = [e_op(i) for i in range(self.N_t)]
        self.e_ops_HMM = [e_op(i) for i in range(self.N_states)]

        J_index = []

        for i in range(self.N_states):
            for j in range(self.N_states):
                if i != j:
                    if self.r_HMM[i, j] != 0:
                        J_index.append([i, j])
                        
            
        self.J_index = np.array(J_index)

        self.meta_data_object = {'N_m_strength': N_m_strength, 'N_theta': N_theta, 't_gamma': t_gamma, 'gamma': gamma, 
                                   'dt': dt, 'mu1': mu1, 'mu2': mu2, 'seed_signal': seed_signal, 'seed_simulation': seed_simulation, 
                                    't_signal_leadin': t_signal_leadin, 'gamma_decay': gamma_decay, 'gamma_phi': gamma_phi, 
                                    'm_min': m_min, 'm_max': m_max, 'theta_min': theta_min, 'theta_max': theta_max, 
                                    'methode': methode, 'p': p, 'field_type': field_type, 'save': save, 'dephase_strength': 
                                    dephase_strength, 'entaglement': entaglement, 'beta_1': beta_1, 'beta_2': beta_2}


        


    def simulate_2_spin(self, input):
        Delta_n_1, Delta_n_2 = self.simulate_signal()
        dY = self.simulate_detection_record(Delta_n_1, Delta_n_2)
        Experiment_estimater = self.Estimate_signal(dY)
        RMS_error, mean_error = self.calculate_error(Experiment_estimater)
        self.save_function(RMS_error, mean_error, self.name)

        



    
    def simulate_signal(self):
        
        B_1 = np.zeros((self.N_t + self.N_t_leadin, 3))
        B_2 = np.zeros((self.N_t + self.N_t_leadin, 3))


        HMM_state_index = np.zeros(self.N_t + self.N_t_leadin, dtype=int) # the index of the HMM state at each time step
        HMM_state_index[0] = int(self.N_states / 2) #r_signal.integers(0, N_states) # the initial state of the HMM

            
        B_1[0] = self.B_1_posible[HMM_state_index[0]]
        B_2[0] = self.B_2_posible[HMM_state_index[0]]


        # generate the signal

        r_HMM_sim = self.r_HMM * self.dt - np.diag(np.diag(self.r_HMM)) * self.dt
        r_HMM_sim = r_HMM_sim + np.eye(self.N_states) * (1 - np.sum(r_HMM_sim, axis=1))

        q = np.random.choice(self.N_states, self.N_t_leadin + self.N_t, p=np.ones(self.N_states) / self.N_states)


        # generate the signal
        for i in range(1, self.N_t_leadin + self.N_t):
            q = np.random.choice(self.N_states, 1, p=r_HMM_sim[HMM_state_index[i - 1]])
            HMM_state_index[i] = q[0]



        self.HMM_state_index_new = HMM_state_index[self.N_t_leadin:] # remove the leadin of the signal

        B_1 = self.B_1_posible[self.HMM_state_index_new]
        B_2 = self.B_2_posible[self.HMM_state_index_new]
            
        #m_t = self.m_mesh[self.HMM_state_index_new]
        #theta_t = self.theta_mesh[self.HMM_state_index_new]

        Delta_n_1 = B_1 * self.mu1 # signal, rounded to the nearest integer as the ME simulationonly works for a descrete set of signal strengths
        Delta_n_2 = B_2 * self.mu2 # signal, rounded to the nearest integer as the ME simulationonly works for a descrete set of signal strengths

        

        return Delta_n_1, Delta_n_2
    
    def simulate_detection_record(self, Delta_n_1, Delta_n_2):
        H_1 = lambda i : 1/2 * (self.sigma_x_1 * Delta_n_1[i, 0] + self.sigma_y_1 * Delta_n_1[i, 1] + self.sigma_z_1 * Delta_n_1[i, 2])
        H_2 = lambda i : 1/2 * (self.sigma_x_2 * Delta_n_2[i, 0] + self.sigma_y_2 * Delta_n_2[i, 1] + self.sigma_z_2 * Delta_n_2[i, 2])

        H_pqs = [H_1(i) + H_2(i) for i in range(self.N_t)]
        Experiment = PQS.Experiment_simulation(H_pqs, self.rho_0, self.times, c_ops=self.c_ops, sc_ops=self.sc_ops, e_ops=self.e_ops, timedependent_H=True) # not very elegant that one need to write the 1 at N_states
        Experiment.solve(methode=self.methode)

        return Experiment.detection_record
    
    def Estimate_signal(self, dY):
        H_1_HMM = lambda i : 1/2 * (self.sigma_x_1 * self.Delta_n_1_posible[i, 0] + self.sigma_y_1 * self.Delta_n_1_posible[i, 1] + 
                                    self.sigma_z_1 * self.Delta_n_1_posible[i, 2])
        H_2_HMM = lambda i : 1/2 * (self.sigma_x_2 * self.Delta_n_2_posible[i, 0] + self.sigma_y_2 * self.Delta_n_2_posible[i, 1] + 
                                    self.sigma_z_2 * self.Delta_n_2_posible[i, 2])

        H_HMM = [H_1_HMM(i) + H_2_HMM(i) for i in range(self.N_states)]

        Experiment_estimater = PQS.Experiment_estimation(H_HMM, self.rho_HMM, self.times, self.c_ops_HMM, self.sc_ops_HMM, 
                                                         self.N_states, self.r_HMM, self.J_index, e_ops=self.e_ops_HMM, timedependent_H=False)
        Experiment_estimater.solve(dY, methode=self.methode)


        return Experiment_estimater
    
    def calculate_error(self, Estimater):
        simulation_results = Estimater.P_n_rho.real
        true_results = self.HMM_state_index_new
        RMS_error = self.RMS(true_results, simulation_results)
        mean_error = self.weighted_mean_error(self.B_1_posible[:, 2], simulation_results, true_results)
        return RMS_error, mean_error


    def RMS(self, True_x, estimated_x):
        for i in range(len(estimated_x)):
            estimated_x[i, True_x[i]] += - 1
        return np.sqrt(np.mean((estimated_x) ** 2, axis=1))
    
    def weighted_mean_error(self, x, w, x_True):
        x_estimate = np.sum(x * w, axis=1)
        error = (x_estimate - x_True) ** 2
        return error
    
    

    def save_function(self, RMS_error, mean_error, name):
        #check if the file exists and add meta data to the top pf the file if it does not exist
        
        if os.path.isfile(name+'_RMS_error'+".txt"):
            pass
        else:
            with open(name+'_RMS_error'+".txt", "a") as file:
                file.write('meta_data: ')
                file.write(str(self.meta_data_object))
                file.write('\n')

        #Appends the data to the file
        with open(name+'RMS_error'+".txt", "a") as file:
            np.savetxt(file, RMS_error, newline=' ')
            file.write('\n')


        
        if os.path.isfile(name+'_mean_error'+".txt"):
            pass
        else:
            with open(name+'mean_error'+".txt", "a") as file:
                file.write('meta_data: ')
                file.write(str(self.meta_data_object))
                file.write('\n')

        #Appends the data to the file
        with open(name+'mean_error'+".txt", "a") as file:
            np.savetxt(file, mean_error, newline=' ')
            file.write('\n')
            
            

        

        


def magnetic_field(r, m):
    """The magnetic field at position r due to a magnetic dipole m. The magnetic moment is in units of mu_0"""
    r_norm = np.sqrt(np.dot(r, r))

    
    
    B = 1 / (4 * np.pi) * (3 * np.dot(m, r) * r / (r_norm**5) - m / (r_norm**3))
    
    return B

def rotation_matrix(theta, axis):
    """The rotation matrix for a rotation of theta around the axis"""
    axis = axis / np.linalg.norm(axis)
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    return np.array([[a**2 + b**2 - c**2 - d**2, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                     [2 * (b * c + a * d), a**2 + c**2 - b**2 - d**2, 2 * (c * d - a * b)],
                     [2 * (b * d - a * c), 2 * (c * d + a * b), a**2 + d**2 - b**2 - c**2]])

def gennerate_r_next_neighbour_jump(N, st):
    r = np.eye(N, k=0) * (1 - 2 * st)
    r += np.eye(N, k=1) * st
    r += np.eye(N, k=-1) * st
    r[0, 0] = 1 - st
    r[-1, -1] = 1 - st
    return r




