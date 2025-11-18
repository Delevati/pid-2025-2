from PIL import Image
import os

# Diretórios
dir_mf = '/Users/luryand/Documents/PID2024-2/codes/otimizacao/plots/plots_malha_fechada'
dir_ma = '/Users/luryand/Documents/PID2024-2/codes/otimizacao/plots/plots_malha_aberta'
dir_output = '/Users/luryand/Documents/PID2024-2/codes/otimizacao/plots/plots_combinados'

# Criar diretório de saída se não existir
os.makedirs(dir_output, exist_ok=True)

# Mapear os arquivos correspondentes
pares = [
    ('degrau.png', 'degrau_combinado.png'),
    ('senoidal.png', 'senoidal_combinado.png'),
    ('quadrada.png', 'quadrada_combinado.png'),
    ('dente_de_serra.png', 'dente_serra_combinado.png'),
    ('aleatório.png', 'aleatorio_combinado.png')
]

for arquivo_original, arquivo_saida in pares:
    try:
        # Carregar imagens
        img_mf = Image.open(os.path.join(dir_mf, arquivo_original))
        img_ma = Image.open(os.path.join(dir_ma, arquivo_original))
        
        # Redimensionar para mesma altura se necessário
        if img_mf.size[1] != img_ma.size[1]:
            nova_altura = max(img_mf.size[1], img_ma.size[1])
            img_mf = img_mf.resize((int(img_mf.size[0] * nova_altura / img_mf.size[1]), nova_altura))
            img_ma = img_ma.resize((int(img_ma.size[0] * nova_altura / img_ma.size[1]), nova_altura))
        
        # Criar imagem combinada (lado a lado)
        largura_total = img_mf.size[0] + img_ma.size[0]
        altura = img_mf.size[1]
        img_combinada = Image.new('RGB', (largura_total, altura), 'white')
        
        # Colar imagens (MF esquerda, MA direita)
        img_combinada.paste(img_mf, (0, 0))
        img_combinada.paste(img_ma, (img_mf.size[0], 0))
        
        # Salvar
        caminho_saida = os.path.join(dir_output, arquivo_saida)
        img_combinada.save(caminho_saida)
        print(f'✅ Criado: {arquivo_saida}')
        
    except FileNotFoundError as e:
        print(f'❌ Erro: Arquivo não encontrado - {arquivo_original}')
    except Exception as e:
        print(f'❌ Erro ao processar {arquivo_original}: {e}')

print(f'\n📁 Imagens combinadas salvas em: {dir_output}')