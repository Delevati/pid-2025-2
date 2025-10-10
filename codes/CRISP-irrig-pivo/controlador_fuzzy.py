import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl



erro_umidade = ctrl.Antecedent(np.arange(-0.5, 0.51, 0.01), 'erro_umidade')


fator_clima = ctrl.Antecedent(np.arange(0, 10.1, 0.1), 'fator_clima')


vazao_agua = ctrl.Consequent(np.arange(0, 101, 1), 'vazao_agua')



erro_umidade['MUITO_SECO'] = fuzz.trapmf(erro_umidade.universe, [0.15, 0.3, 0.5, 0.5])
erro_umidade['SECO'] = fuzz.trimf(erro_umidade.universe, [0.05, 0.15, 0.25])
erro_umidade['IDEAL'] = fuzz.trimf(erro_umidade.universe, [-0.05, 0, 0.05])
erro_umidade['UMIDO'] = fuzz.trimf(erro_umidade.universe, [-0.2, -0.1, 0])
erro_umidade['MUITO_UMIDO'] = fuzz.trapmf(erro_umidade.universe, [-0.5, -0.5, -0.2, -0.1])


fator_clima['AMENO'] = fuzz.trapmf(fator_clima.universe, [0, 0, 2, 4])
fator_clima['MODERADO'] = fuzz.trimf(fator_clima.universe, [3, 5, 7])
fator_clima['AGRESSIVO'] = fuzz.trapmf(fator_clima.universe, [6, 8, 10, 10])


vazao_agua['ZERO'] = fuzz.trimf(vazao_agua.universe, [0, 0, 10])
vazao_agua['BAIXA'] = fuzz.trimf(vazao_agua.universe, [5, 25, 45])
vazao_agua['MEDIA'] = fuzz.trimf(vazao_agua.universe, [35, 50, 65])
vazao_agua['ALTA'] = fuzz.trimf(vazao_agua.universe, [55, 75, 95])
vazao_agua['MAXIMA'] = fuzz.trapmf(vazao_agua.universe, [90, 100, 100, 100])



rule1 = ctrl.Rule(erro_umidade['MUITO_UMIDO'], vazao_agua['ZERO'])
rule2 = ctrl.Rule(erro_umidade['UMIDO'], vazao_agua['ZERO'])
rule3 = ctrl.Rule(erro_umidade['IDEAL'] & fator_clima['AMENO'], vazao_agua['ZERO'])
rule4 = ctrl.Rule(erro_umidade['IDEAL'] & fator_clima['MODERADO'], vazao_agua['BAIXA'])
rule5 = ctrl.Rule(erro_umidade['IDEAL'] & fator_clima['AGRESSIVO'], vazao_agua['MEDIA'])

rule6 = ctrl.Rule(erro_umidade['SECO'] & fator_clima['AMENO'], vazao_agua['MEDIA'])
rule7 = ctrl.Rule(erro_umidade['SECO'] & fator_clima['MODERADO'], vazao_agua['ALTA'])
rule8 = ctrl.Rule(erro_umidade['SECO'] & fator_clima['AGRESSIVO'], vazao_agua['ALTA'])

rule9 = ctrl.Rule(erro_umidade['MUITO_SECO'] & fator_clima['AMENO'], vazao_agua['ALTA'])
rule10 = ctrl.Rule(erro_umidade['MUITO_SECO'] & fator_clima['MODERADO'], vazao_agua['MAXIMA'])
rule11 = ctrl.Rule(erro_umidade['MUITO_SECO'] & fator_clima['AGRESSIVO'], vazao_agua['MAXIMA'])



sistema_controle = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11])


simulador_pivo = ctrl.ControlSystemSimulation(sistema_controle)


def get_controle_fuzzy(valor_erro_umidade, valor_fator_clima):
    """
    Calcula a ação de controle Fuzzy com base nas entradas do sistema.

    Args:
        valor_erro_umidade (float): O erro atual (umidade_ideal - umidade_medida).
        valor_fator_clima (float): O fator climático calculado (0 a 10).

    Returns:
        float: A porcentagem de vazão de água (0 a 100) a ser aplicada.
    """
    
    simulador_pivo.input['erro_umidade'] = valor_erro_umidade
    simulador_pivo.input['fator_clima'] = valor_fator_clima

    simulador_pivo.compute()

    return simulador_pivo.output['vazao_agua']