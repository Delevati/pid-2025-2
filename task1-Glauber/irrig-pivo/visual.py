import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from modelo import ler_sensor_umidade

def criar_figura(setores, obstaculos):
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(right=0.7)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title("Simulação do Pivô de Irrigação com Dinâmica Física")

    setor_colors = ['#c2b280', '#8c9e5e', '#a48464', '#d2b48c']
    setor_labels = ['A', 'B', 'C', 'D']
    setor_wedges = []
    for i in range(4):
        wedge = Wedge((0,0), 1.2, i*90, (i+1)*90, facecolor=setor_colors[i], alpha=0.5, edgecolor='gray')
        ax.add_patch(wedge)
        setor_wedges.append(wedge)
        ang_label = np.deg2rad(i*90 + 45)
        ax.text(0.8*np.cos(ang_label), 0.8*np.sin(ang_label), setor_labels[i], fontsize=14, weight='bold', ha='center', va='center')

    for obs in obstaculos:
        ang = np.deg2rad(obs)
        ax.plot([1.1*np.cos(ang)], [1.1*np.sin(ang)], 'ro', markersize=10, zorder=10)
    
    pivo_centro = Circle((0,0), 0.07, color='black', zorder=10)
    ax.add_patch(pivo_centro)

    pivo_arm, = ax.plot([], [], 'k-', linewidth=4, zorder=8)
    aspersores, = ax.plot([], [], 'ko', markersize=4, zorder=9)
    water, = ax.plot([], [], 'o', color='deepskyblue', markersize=3, alpha=0.7)

    motores_plot = []
    for _ in range(10):
        motor, = ax.plot([], [], 'o', color='gray', markersize=6, zorder=9)
        motores_plot.append(motor)

    info_text = fig.text(
        0.72, 0.97, '', fontsize=9, va='top', ha='left', family='monospace',
        bbox=dict(facecolor='whitesmoke', edgecolor='lightgray', boxstyle='round,pad=0.5')
    )
    
    elementos = {
        "setor_wedges": setor_wedges,
        "pivo_arm": pivo_arm,
        "aspersores": aspersores,
        "water": water,
        "info_text": info_text,
        "motores_plot": motores_plot
    }
    return fig, ax, elementos

def atualizar_visual(elementos, setores, estado, parametros):
    setor_wedges = elementos["setor_wedges"]
    pivo_arm = elementos["pivo_arm"]
    aspersores = elementos["aspersores"]
    water = elementos["water"]
    info_text = elementos["info_text"]
    motores_plot = elementos["motores_plot"]

    comprimento_braco = parametros["comprimento_braco"]
    ang_atual = estado["ang_atual"]
    ang_rad = np.deg2rad(ang_atual)
    
    x_arm = [0, 1.2 * np.cos(ang_rad)]
    y_arm = [0, 1.2 * np.sin(ang_rad)]
    pivo_arm.set_data(x_arm, y_arm)

    posicoes_aspersores = estado["posicoes_aspersores"]
    aspersores_norm = posicoes_aspersores / comprimento_braco * 1.2
    x_aspersores = aspersores_norm * np.cos(ang_rad)
    y_aspersores = aspersores_norm * np.sin(ang_rad)
    aspersores.set_data(x_aspersores, y_aspersores)
    
    vazao_total = estado["vazao_total"]
    if vazao_total > 10:
        x_water, y_water = [], []
        for i, pos_norm in enumerate(aspersores_norm):
            if estado["vazoes_aspersores"][i] > 5.0:
                x_asp, y_asp = pos_norm * np.cos(ang_rad), pos_norm * np.sin(ang_rad)
                raio_aspersao = min(0.06, estado["vazoes_aspersores"][i] / 500)
                n_drops = max(3, int(estado["vazoes_aspersores"][i] / 15))
                for _ in range(n_drops):
                    angle_drop, radius_drop = np.random.uniform(0, 2*np.pi), np.random.uniform(0, raio_aspersao)
                    x_water.append(x_asp + radius_drop * np.cos(angle_drop))
                    y_water.append(y_asp + radius_drop * np.sin(angle_drop))
        water.set_data(x_water, y_water)
    else:
        water.set_data([], [])

    for i, wedge in enumerate(setor_wedges):
        umidade = setores[i]['umidade']
        capacidade = setores[i]['capacidade']
        cor_alpha = np.interp(umidade, [0, capacidade * 1.1], [0.3, 0.9])
        wedge.set_alpha(cor_alpha)
        
    estados_motores = estado.get("estados_motores", [])
    for i, motor_plot in enumerate(motores_plot):
        if i < len(estados_motores):
            m = estados_motores[i]
            cor = 'red' if m["ligado"] else 'gray'
            x = (m["pos"] / comprimento_braco) * 1.2 * np.cos(ang_rad)
            y = (m["pos"] / comprimento_braco) * 1.2 * np.sin(ang_rad)
            motor_plot.set_data([x], [y])
            motor_plot.set_color(cor)

    info_texto = (
        f"═══ SIMULAÇÃO ═══\n"
        f"Dia: {estado['dias_completos']:3d} | Hora: {estado['hora_do_dia']:5.1f}h\n"
        f"Ângulo:       {estado['ang_atual']:6.1f}°\n"
        f"Velocidade:   {estado['vel_angular']:6.2f}°/min\n"
        f"Torque motor: {estado['torque_motor']:7.0f} N⋅m\n"
        f"Declive:      {estado['declive']:+5.1f}°\n"
        f"\n═══ IRRIGAÇÃO (Setor {estado['setor_atual']['nome']}) ═══\n"
        f"Pressão Atual: {estado['pressao_atual']:5.1f} bar\n"
        f"Pressão Desej: {estado['pressao_desejada']:5.1f} bar\n"
        f"Vazão Total:  {estado['vazao_total']:6.1f} L/s\n"
        f"Umidade real: {estado['setor_atual']['umidade']:6.1%}\n"
        f"Umidade sens: {ler_sensor_umidade(estado['setor_atual']['umidade']):6.1%}\n"
        f"Capacidade:   {estado['setor_atual']['capacidade']:6.1%}\n"
        f"\n═══ AMBIENTE E STATUS ═══\n"
        f"Temperatura:  {estado['temperatura_ambiente']:5.1f}°C\n"
        f"Consumo total:{estado['consumo_agua_total']/1000:8.1f} m³\n"
        f"Umidade média:{estado['umidade_media']:6.1%}\n"
        f"MODO: {'EMERGÊNCIA' if estado['modo_emergencia'] else 'NORMAL'}\n"
    )
    info_text.set_text(info_texto)

    todos_elementos = [pivo_arm, aspersores, water, info_text] + setor_wedges + motores_plot
    return todos_elementos
