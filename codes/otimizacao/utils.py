import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.integrate import odeint

def gerar_sinal_referencia(tipo_sinal, tempo, parametros):
    if tipo_sinal == 'degrau':
        amplitude = parametros.get('amplitude', 1.0)
        return amplitude * np.ones_like(tempo)
    
    elif tipo_sinal == 'senoidal':
        amplitude = parametros.get('amplitude', 1.0)
        periodo = parametros.get('periodo', 1.0)
        offset = parametros.get('offset', 0.0)
        frequencia = 1 / periodo
        return amplitude * np.sin(2 * np.pi * frequencia * tempo) + offset
    
    elif tipo_sinal == 'quadrada':
        amplitude = parametros.get('amplitude', 1.0)
        periodo = parametros.get('periodo', 1.0)
        offset = parametros.get('offset', 0.0)
        frequencia = 1 / periodo
        return amplitude * np.sign(np.sin(2 * np.pi * frequencia * tempo)) + offset
    
    elif tipo_sinal == 'dente_serra':
        amplitude = parametros.get('amplitude', 1.0)
        periodo = parametros.get('periodo', 1.0)
        offset = parametros.get('offset', 0.0)
        
        signal = np.zeros_like(tempo)
        for i in range(len(tempo)):
            signal[i] = amplitude * (2 * ((tempo[i] / periodo) % 1.0) - 1) + offset
        return signal
    
    elif tipo_sinal == 'aleatorio':
        amp_max = parametros.get('amp_max', 1.0)
        amp_min = parametros.get('amp_min', -1.0)
        periodo_max = parametros.get('periodo_max', 0.5)
        periodo_min = parametros.get('periodo_min', 0.1)
        
        signal = np.zeros_like(tempo)
        current_time = 0
        current_value = np.random.uniform(amp_min, amp_max)
        signal[0] = current_value
        
        for i in range(1, len(tempo)):
            if tempo[i] >= current_time:
                current_time += np.random.uniform(periodo_min, periodo_max)
                current_value = np.random.uniform(amp_min, amp_max)
            signal[i] = current_value
        
        return signal
    
    else:
        raise ValueError(f"Tipo de sinal '{tipo_sinal}' não implementado")

def simular_sistema_malha_fechada(connected_systems_model, kp, ki, kd, tipo_sinal, parametros_sinal, a, k, ts_ms, tf, dt):
    """
    Simula o sistema em malha fechada com os parâmetros otimizados do controlador PID
    """
    n = int((1 / (ts_ms / 1000.0)) * tf + 1)
    time_vector = np.linspace(0, tf, n)
    
    torque_ref = gerar_sinal_referencia(tipo_sinal, time_vector, parametros_sinal)
    
    states = np.zeros((n, 1))
    states[0] = [0]
    erro_anterior = 0
    erro_acum = 0
    control_signals = np.zeros(n)
    
    for i in range(n-1):
        t_span = [time_vector[i], time_vector[i+1]]
        tau = states[i, 0]
        tau_ref_i = torque_ref[i]
        
        if i < n-2:
            taup_ref_i = (torque_ref[i+1] - torque_ref[i]) / (time_vector[i+1] - time_vector[i])
        else:
            taup_ref_i = 0
        
        erro_atual = tau_ref_i - tau
        erro_acum += erro_atual * dt
        d_erro = (erro_atual - erro_anterior) / dt
        
        v = kp * erro_atual + ki * ts_ms * erro_acum + kd * d_erro / ts_ms
        control_signals[i] = v
        
        if abs(v) > 12.0:
            v = np.sign(v) * 12.0
            erro_acum = erro_acum - erro_atual * dt
            control_signals[i] = v
        
        out_states = odeint(connected_systems_model, states[i], t_span,
                          args=(tau_ref_i, taup_ref_i, erro_acum, d_erro, kp, ki, kd))
        states[i+1] = out_states[-1]
        erro_anterior = erro_atual
    
    erro_atual = torque_ref[-1] - states[-1, 0]
    v = kp * erro_atual + ki * ts_ms * erro_acum + kd * d_erro / ts_ms
    control_signals[-1] = v if abs(v) <= 12.0 else np.sign(v) * 12.0
    
    return time_vector, torque_ref, states[:, 0], control_signals

def visualizar_resultados(time_vector, reference, output, control_signal, tipo_sinal, save_dir=None):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(time_vector, reference, 'r--', linewidth=2, label='Referência')
    plt.plot(time_vector, output, 'b-', linewidth=2, label='Saída')
    plt.grid(True)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Resposta do Sistema - Sinal {tipo_sinal}')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(time_vector, control_signal, 'g-', linewidth=2)
    plt.grid(True)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Tensão (V)')
    plt.title('Sinal de Controle')
    
    plt.axhline(y=12.0, color='r', linestyle='--', alpha=0.7)
    plt.axhline(y=-12.0, color='r', linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{tipo_sinal.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot salvo em {filename}")
    
    plt.show()