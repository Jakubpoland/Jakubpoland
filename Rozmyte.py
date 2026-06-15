import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# Definicja zmiennych stanu
x1 = ctrl.Antecedent(np.arange(-1,1.01,0.01), 'x1')
x2 = ctrl.Antecedent(np.arange(-1,1.01,0.01), 'x2')
u =ctrl.Consequent(np.arange(-2,2.01,0.01), 'u')
# Zbiory rozmyte dla poprzednika x1
x1['A1'] = fuzz.trimf(x1.universe, [-1,-1, 0])
x1['A2'] = fuzz.trimf(x1.universe, [-1, 0, 1])
x1['A3'] = fuzz.trimf(x1.universe, [0, 1, 1])
# Zbiory rozmyte dla poprzednika x2
x2['B1'] = fuzz.trimf(x2.universe, [-1,-1, 0])
x2['B2'] = fuzz.trimf(x2.universe, [-1, 0, 1])
x2['B3'] = fuzz.trimf(x2.universe, [ 0, 1, 1])
# Zbiory rozmyte dla następnika u
u['C1'] = fuzz.trimf(u.universe, [-2, -2, -1])
u['C2'] = fuzz.trimf(u.universe, [-1, 0, 1])
u['C3'] = fuzz.trimf(u.universe, [1, 2, 2])
# Definicje reguł
regula1 = ctrl.Rule(x1['A1'] & x2['B2'], u['C1'])
regula2 = ctrl.Rule(x1['A1'] & x2['B3'], u['C2'])
regula3 = ctrl.Rule(x1['A2'] & x2['B2'], u['C2'])
regula4 = ctrl.Rule(x1['A2'] & x2['B3'], u['C3'])
# Dodanie zdefiniowanych reguł do zbioru rozmytego
u_ctr = ctrl.ControlSystem([regula1,regula2,regula3,regula4])
u_sym = ctrl.ControlSystemSimulation(u_ctr)
# Obliczenie wyniku dla wartości x1 =-1.7, x2 = 0.9
u_sym.input['x1'] = 0.7
u_sym.input['x2'] = -0.4
u_sym.compute()
print('Wynik',u_sym.output['u'])
u.view(sim=u_sym)
plt.show()