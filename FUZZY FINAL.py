import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

temperature = ctrl.Antecedent(np.arange(10,95,5), 'temperature')
pressure = ctrl.Antecedent(np.arange(1,6,0.05), 'pressure')
h_power = ctrl.Consequent(np.arange(1,6,0.05), 'h_power')
pressure['low'] = fuzz.trimf(pressure.universe,[1,1,1.75])
pressure['ba'] = fuzz.trimf(pressure.universe,[1.25,2,2.75])
pressure['avg'] = fuzz.trimf(pressure.universe,[2,3,4])
pressure['aa'] = fuzz.trimf(pressure.universe,[3.25,4,4.75])
pressure['high'] = fuzz.trimf(pressure.universe,[4.25,5,5])
temperature['low'] = fuzz.trimf(temperature.universe,[10,10,25])
temperature['ba'] = fuzz.trimf(temperature.universe,[15,30,45])
temperature['avg'] = fuzz.trimf(temperature.universe,[40,50,60])
temperature['aa'] = fuzz.trimf(temperature.universe,[55,70,85])
temperature['high'] = fuzz.trimf(temperature.universe,[75,90,90])
h_power['low'] = fuzz.trimf(h_power.universe,[1,1,1.5])
h_power['ml'] = fuzz.trimf(h_power.universe,[1.25,2,2.75])
h_power['med'] = fuzz.trimf(h_power.universe,[2.5,3,3.75])
h_power['mh'] = fuzz.trimf(h_power.universe,[3.5,4,4.5])
h_power['high'] = fuzz.trimf(h_power.universe,[4.25,5,5])
h_power.view()
temperature.view()
pressure.view()
rule1 = ctrl.Rule(temperature['aa'] & pressure['aa'], h_power['ml'])
rule2 = ctrl.Rule(temperature['high'] & pressure['high'], h_power['low'])
rule3 = ctrl.Rule(temperature['high'] & pressure['low'], h_power['med'])

hp_ctrl = ctrl.ControlSystem([rule1,rule2,rule3])
heating_power = ctrl.ControlSystemSimulation(hp_ctrl)

heating_power.input['temperature'] = 80 
heating_power.input['pressure'] = 4.5

heating_power.compute()

print('Crisp Output:',heating_power.output['h_power'])