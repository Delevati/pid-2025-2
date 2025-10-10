from modelo import setores, parametros, estado_inicial, atualizar_estado
from visual import criar_figura, atualizar_visual
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import copy
import shutil
import os

fig, ax, elementos = criar_figura(setores)
estado = copy.deepcopy(estado_inicial)

def frame_func(frame):
    global estado
    estado = atualizar_estado(estado, setores, parametros)
    return atualizar_visual(elementos, setores, estado, parametros)

ani = FuncAnimation(fig, frame_func, frames=3600, interval=100, blit=False)
plt.tight_layout()
plt.show()

for root, dirs, files in os.walk('.', topdown=False):
    for name in dirs:
        if name == '__pycache__':
            shutil.rmtree(os.path.join(root, name))