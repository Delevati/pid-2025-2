import numpy as np
from scipy import signal
import random
from utils import simular_sistema_funcao_transferencia, visualizar_resultados
import warnings


save_dir = '/Users/luryand/Documents/PID2024-2/codes/otimizacao/plots/plots_funcao_transferencia'

tf = 2.0
ts_ms = 1
dt = ts_ms/1000.0

n_part = 30
max_iter = 30
lim = [(0.01, 50.0),    # kp
       (0.0, 20.0),     # ki
       (0.0, 10.0)]      # kd

peso_inercia = 0.9
peso_local = 1.2
peso_global = 1.2
veloc_max = [(b[1]-b[0])*0.2 for b in lim]

def calcular_funcao_objetivo(kp, ki, kd):
    try:
        _, _, y, _ = simular_sistema_funcao_transferencia(
            kp, ki, kd, 
            'senoidal',
            {'amplitude': 1.0, 'periodo': 1.0, 'offset': 0.0},
            ts_ms, tf, dt
        )
        
        n = len(y)
        time_vector = np.linspace(0, tf, n)
        torque_ref = np.sin(time_vector)
        
        erro = torque_ref - y
        erro_quad = np.mean(erro**2)
        
        penalizacao_kd = 10.0 if kd < 0.1 else 0.0
        
        return erro_quad + penalizacao_kd
    except:
        return float('inf')

particles = []
velocity = []
pbest = []
pbest_fit = []

for i in range(n_part):
    p = [random.uniform(lim[j][0], lim[j][1]) for j in range(3)]
    particles.append(p)
    velocity.append([random.uniform(-veloc_max[j], veloc_max[j]) for j in range(3)])
    pbest.append(p[:])
    pbest_fit.append(calcular_funcao_objetivo(*p))

gbest = pbest[0][:]
gbest_fit = pbest_fit[0]
for i in range(1, n_part):
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
                particles[i] = [random.uniform(lim[j][0], lim[j][1]) for j in range(3)]
                velocity[i] = [random.uniform(-veloc_max[j], veloc_max[j]) for j in range(3)]

print("Final:", gbest, gbest_fit)
print(f"Parâmetros do controlador PID: kp={gbest[0]:.3f}, ki={gbest[1]:.3f}, kd={gbest[2]:.3f}")

tf_simulacao = 5.0

parametros_degrau = {'amplitude': 1.0}
tempo, ref, saida, controle = simular_sistema_funcao_transferencia(
    gbest[0], gbest[1], gbest[2], 'degrau', parametros_degrau, ts_ms, tf_simulacao, dt)
visualizar_resultados(tempo, ref, saida, controle, 'Degrau', save_dir)

parametros_senoidal = {'amplitude': 1.0, 'periodo': 1.0, 'offset': 0.0}
tempo, ref, saida, controle = simular_sistema_funcao_transferencia(
    gbest[0], gbest[1], gbest[2], 'senoidal', parametros_senoidal, ts_ms, tf_simulacao, dt)
visualizar_resultados(tempo, ref, saida, controle, 'Senoidal', save_dir)

parametros_quadrada = {'amplitude': 1.0, 'periodo': 1.0, 'offset': 0.0}
tempo, ref, saida, controle = simular_sistema_funcao_transferencia(
    gbest[0], gbest[1], gbest[2], 'quadrada', parametros_quadrada, ts_ms, tf_simulacao, dt)
visualizar_resultados(tempo, ref, saida, controle, 'Quadrada', save_dir)

parametros_dente_serra = {'amplitude': 1.0, 'periodo': 1.0, 'offset': 0.0}
tempo, ref, saida, controle = simular_sistema_funcao_transferencia(
    gbest[0], gbest[1], gbest[2], 'dente_serra', parametros_dente_serra, ts_ms, tf_simulacao, dt)
visualizar_resultados(tempo, ref, saida, controle, 'Dente de Serra', save_dir)

parametros_aleatorio = {'amp_max': 1.0, 'amp_min': -1.0, 'periodo_max': 0.5, 'periodo_min': 0.1}
tempo, ref, saida, controle = simular_sistema_funcao_transferencia(
    gbest[0], gbest[1], gbest[2], 'aleatorio', parametros_aleatorio, ts_ms, tf_simulacao, dt)
visualizar_resultados(tempo, ref, saida, controle, 'Aleatório', save_dir)