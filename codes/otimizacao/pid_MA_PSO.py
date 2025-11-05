import numpy as np
from scipy.integrate import odeint
import random
from utils import simular_sistema_malha_aberta, visualizar_resultados


"""
Sistema de Controle PID com Otimização para k_p, k_i e k_d por Enxame de Partículas (PSO)

O código simula um modelo do motor, com controlador e simulações para avaliar a diferença 
entre a saída do sistema e a referência desejada.

Nota: Sistema de Controle PID em MALHA ABERTA com Otimização por PSO
O controle é baseado APENAS na referência desejada (feedforward).
Isso resulta em desempenho inferior, pois não corrige erros ou perturbações.
"""
save_dir = '/Users/luryand/Documents/PID2024-2/codes/otimizacao/plots/plots_malha_aberta'

a = 0.05
k = 2.0
a_model_error = 0.1
k_model_error = 0.3

tf = 2.0
ts_ms = 1
dt = ts_ms/1000.0

n_part = 30
max_iter = 30
lim = [(1.0, 50.0),      # kp: ganho proporcional
       (0.0, 20.0),      # ki: ganho integral  
       (0.1, 10.0)]      # kd: ganho derivativo

peso_inercia = 0.9
peso_local = 1.2
peso_global = 1.2
veloc_max = [(b[1]-b[0])*0.2 for b in lim]

def motor_model(x1_m, u, a, k):
    dx1_m = -a*k*x1_m + k*u
    return dx1_m

def motor_controller(tau, tau_ref, taup_ref, erro_acum, d_erro, k_p, k_i, k_d):
    """ REMOVIDO a saída com erro, aqui na malha aberta usamos apenas a referencia e
    não comparamos a referência com a saída real, SEM FEEDBACK"""
    # erro = tau_ref - tau
    v = k_p * tau_ref + k_i * ts_ms * erro_acum + k_d * d_erro / ts_ms

    if abs(v) > 12.0:
        v = np.sign(v) * 12.0
    return (a + a_model_error) * tau + v / (k + k_model_error)

def connected_systems_model(states, t, tau_ref, taup_ref, erro_acum, d_erro, k_p, k_i, k_d):
    """
    Na malha ABERTA o controlador não vai receber a saída atual (states[0]) como feedback
    removido x1_m do argumento da função motor_controller
    """
    x1_m = states[0]
    dc_volts = motor_controller(x1_m, tau_ref, taup_ref, erro_acum, d_erro, k_p, k_i, k_d)
    dx1_m = motor_model(x1_m, dc_volts, a, k)
    return [dx1_m]

def calcular_funcao_objetivo(kp, ki, kd):
    """
    Função Objetivo da MALHA ABERTA
    A diferença principal é que acumulamos só o tau_ref_i (referência) e não acrescentamos mais o erro da saída. 
    """
    n = int((1 / (ts_ms / 1000.0)) * tf + 1)
    time_vector = np.linspace(0, tf, n)
    torque_ref = np.sin(time_vector)
    torquep_ref = np.cos(time_vector)
    
    states = np.zeros((n, 1))
    states[0] = [0]
    erro_acum = 0
    control_effort = 0

    for i in range(n-1):
        t_span = [time_vector[i], time_vector[i+1]]
        # noise = np.random.normal(0, 0.01)
        tau = states[i, 0]
        tau_ref_i = torque_ref[i]
        taup_ref_i = torquep_ref[i]

        erro_acum += tau_ref_i * dt
        d_erro = taup_ref_i

        v = kp * tau_ref_i + ki * ts_ms *erro_acum + kd * d_erro / ts_ms
        control_effort += abs(v) * dt
        
        out_states = odeint(connected_systems_model, states[i], t_span,
                            args=(tau_ref_i, taup_ref_i, erro_acum, d_erro, kp, ki, kd))
        states[i+1] = out_states[-1]
    
    tau = states[:, 0]
    erro = torque_ref - tau

    time_weighted_error = np.abs(erro) * time_vector
    ita = np.trapz(time_weighted_error, time_vector)

    steady_state_start = int(0.9 * n)
    esa = np.mean(np.abs(erro[steady_state_start:]))

    erro_dinamico = np.mean(np.abs(erro[:steady_state_start]))
    
    # Penalizações da FObjetivo
    balance_penalty = 0
    # Penaliza kp muito baixo em relação ao ki
    if kp < 3.0 :
        balance_penalty += (3.0 - kp) ** 2 * 50.0

    # Penaliza ki muito baixo, winddown kk 
    if ki < 3.0:
        balance_penalty += (3.0 - ki) ** 2 * 30.0
    # Penaliza ki muito alto, é chamado de windup
    if ki > 10.0:
        balance_penalty += (ki - 10) * 2.0
    
    # Penaliza kd muito baixo (PID sem derivativo)
    if kd < 1.0:
        balance_penalty += (1.0 - kd) ** 2 * 20.0
    # Penaliza kp, ki, kd altos
    balance_penalty += max(0, kp - 80) * 1.0
    # balance_penalty += max(0, ki - 40) * 1.0
    balance_penalty += max(0, kd - 8) * 1.0

    effort_penalty = control_effort * 0.001
    
    # Modifica a função objetivo final
    j = 5.0*ita + 30.0*esa + 10.0*erro_dinamico

    return j + balance_penalty + effort_penalty

"""
Daqui por diante é o PSO (Particle Swarm Optimization), gera, testa e atualiza as combinações.

particles: cada uma é um conjunto de valores (k_p, k_i, k_d) que o algoritmo vai testar
velocity: como cada partícula se move pelo espaço de busca
pbest: melhor resultado que cada partícula já achou
gbest: melhor resultado geral de todas as partículas
 
A ideia é: cada partícula vai testando valores diferentes e se move em direção
ao que ela achou de melhor (pbest) e ao que o grupo todo achou de melhor (gbest)
"""
particles = []
velocity = []
pbest = []
pbest_fit = []

for i in range(n_part):
    p = [random.uniform(lim[j][0],lim[j][1]) for j in range(3)]
    particles.append(p)
    velocity.append([random.uniform(-veloc_max[j],veloc_max[j]) for j in range(3)])
    pbest.append(p[:])
    pbest_fit.append(calcular_funcao_objetivo(*p))

gbest = pbest[0][:]
gbest_fit = pbest_fit[0]
for i in range(1,n_part):
    if pbest_fit[i] < gbest_fit:
        gbest_fit = pbest_fit[i]
        gbest = pbest[i][:]

print("Busca Inicial:", gbest, gbest_fit)

for it in range(max_iter):
    for i in range(n_part):
        f = calcular_funcao_objetivo(*particles[i])
        if f < pbest_fit[i]:
            pbest_fit[i] = f
            pbest[i] = particles[i][:]
            if f < gbest_fit:
                gbest_fit = f
                gbest = particles[i][:]
                print(f"Nova melhor solução na iter {it}: kp={gbest[0]:.3f}, ki={gbest[1]:.3f}, kd={gbest[2]:.3f} | fit={gbest_fit:.10f}")

    for i in range(n_part):
        for d in range(3):
            r1 = random.random()
            r2 = random.random()
            velocity[i][d] = (peso_inercia*velocity[i][d] +
                         peso_local*r1*(pbest[i][d]-particles[i][d]) +
                         peso_global*r2*(gbest[d]-particles[i][d]))
            velocity[i][d] = max(-veloc_max[d], min(veloc_max[d], velocity[i][d]))
            particles[i][d] += velocity[i][d]
            particles[i][d] = max(lim[d][0], min(lim[d][1], particles[i][d]))

            if random.random() < 0.15:
                particles[i][d] = random.uniform(lim[d][0], lim[d][1])

    if it % 3 == 0:
        diversity = 0
        for d in range(3):
            values = [particles[i][d] for i in range(n_part)]
            diversity += np.std(values)

        if diversity < 5.0:
            print(f"Baixa diversidade na iter {it}, reinicializando...")
            for i in range(n_part//2):
                particles[i] = [random.uniform(lim[j][0],lim[j][1]) for j in range(3)]
                velocity[i] = [random.uniform(-veloc_max[j],veloc_max[j]) for j in range(3)]

print("Final:", gbest, gbest_fit)
print(f"Parâmetros do controlador PID: kp={gbest[0]:.3f}, ki={gbest[1]:.3f}, kd={gbest[2]:.3f}")

tf_simulacao = 5.0

# Sinal degrau
parametros_degrau = {'amplitude': 1.0}
tempo, ref, saida, controle = simular_sistema_malha_aberta(
    connected_systems_model, gbest[0], gbest[1], gbest[2], 'degrau', parametros_degrau, a, k, ts_ms, tf_simulacao, dt)
visualizar_resultados(tempo, ref, saida, controle, 'Degrau', save_dir)

# Sinal senoidal
parametros_senoidal = {'amplitude': 1.0, 'periodo': 1.0, 'offset': 0.0}
tempo, ref, saida, controle = simular_sistema_malha_aberta(
    connected_systems_model, gbest[0], gbest[1], gbest[2], 'senoidal', parametros_senoidal, a, k, ts_ms, tf_simulacao, dt)
visualizar_resultados(tempo, ref, saida, controle, 'Senoidal', save_dir)

# Sinal onda quadrada
parametros_quadrada = {'amplitude': 1.0, 'periodo': 1.0, 'offset': 0.0}
tempo, ref, saida, controle = simular_sistema_malha_aberta(
    connected_systems_model, gbest[0], gbest[1], gbest[2], 'quadrada', parametros_quadrada, a, k, ts_ms, tf_simulacao, dt)
visualizar_resultados(tempo, ref, saida, controle, 'Quadrada', save_dir)

# Sinal dente de serra
parametros_dente_serra = {'amplitude': 1.0, 'periodo': 1.0, 'offset': 0.0}
tempo, ref, saida, controle = simular_sistema_malha_aberta(
    connected_systems_model, gbest[0], gbest[1], gbest[2], 'dente_serra', parametros_dente_serra, a, k, ts_ms, tf_simulacao, dt)
visualizar_resultados(tempo, ref, saida, controle, 'Dente de Serra', save_dir)

# Sinal aleatório
parametros_aleatorio = {'amp_max': 1.0, 'amp_min': -1.0, 'periodo_max': 0.5, 'periodo_min': 0.1}
tempo, ref, saida, controle = simular_sistema_malha_aberta(
    connected_systems_model, gbest[0], gbest[1], gbest[2], 'aleatorio', parametros_aleatorio, a, k, ts_ms, tf_simulacao, dt)
visualizar_resultados(tempo, ref, saida, controle, 'Aleatório', save_dir)