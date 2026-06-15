import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

#Definicja zmiennych stanu
left = ctrl.Antecedent(np.arange(0, 100.01, 0.01), 'left')
right = ctrl.Antecedent(np.arange(0, 100.01, 0.01), 'right')
front = ctrl.Antecedent(np.arange(0, 100.01, 0.01), 'front')

Vl = ctrl.Consequent(np.arange(-50, 50.01, 0.01), 'Vl')
Vr = ctrl.Consequent(np.arange(-50, 50.01, 0.01), 'Vr')

left['S'] = fuzz.trimf(left.universe, [0, 0, 100])
left['B'] = fuzz.trimf(left.universe, [0, 100, 100])

right['S'] = fuzz.trimf(right.universe, [0, 0, 100])
right['B'] = fuzz.trimf(right.universe, [0, 100, 100])

front['S'] = fuzz.trimf(front.universe, [0, 0, 100])
front['B'] = fuzz.trimf(front.universe, [0, 100, 100])

Vl['front'] = fuzz.trimf(Vl.universe, [0, 50, 50])
Vl['back'] = fuzz.trimf(Vl.universe, [-50, -50, 0])
Vr['front'] = fuzz.trimf(Vr.universe, [0, 50, 50])
Vr['back'] = fuzz.trimf(Vr.universe, [-50, -50, 0])

regula1 = ctrl.Rule(antecedent=left['S'] & right['S'] & front['S'], 
                    consequent=(Vl['front'], Vr['front']))
regula2 = ctrl.Rule(antecedent=left['S'] & right['S'] & front['B'], 
                    consequent=(Vl['back'], Vr['front']))
regula3 = ctrl.Rule(antecedent=left['S'] & right['B'] & front['S'], 
                    consequent=(Vl['back'], Vr['front']))
regula4 = ctrl.Rule(antecedent=left['S'] & right['B'] & front['B'], 
                    consequent=(Vl['back'], Vr['front']))
regula5 = ctrl.Rule(antecedent=left['B'] & right['S'] & front['S'], 
                    consequent=(Vl['front'], Vr['back']))
regula6 = ctrl.Rule(antecedent=left['B'] & right['S'] & front['B'], 
                    consequent=(Vl['front'], Vr['back']))
regula7 = ctrl.Rule(antecedent=left['B'] & right['B'] & front['S'], 
                    consequent=(Vl['front'], Vr['front']))
regula8 = ctrl.Rule(antecedent=left['B'] & right['B'] & front['B'], 
                    consequent=(Vl['back'], Vr['front']))

move_ctrl = ctrl.ControlSystem([regula1, regula2, regula3, regula4, regula5, regula6, regula7, regula8])
move_sim = ctrl.ControlSystemSimulation(move_ctrl)

move_sim.input['left'] = 70
move_sim.input['right'] = 90
move_sim.input['front'] = 0
move_sim.compute()

print("Wynik", move_sim.output['Vl'], move_sim.output['Vr'])
Vl.view(sim=move_sim)
Vr.view(sim=move_sim)
plt.show();