#!/usr/bin/python3.6
import nasa42
import time

simulation = nasa42.Simulation()
eci = nasa42.WORLD_INERTIAL_FRAME
now = 1474416005

def stepCallback(simp):
    global now
    global eci
    #print('Py callback ' + str(sim))
    if now > 1474419100:
        print('Ending sim at ' + str(now))
        return 1

    now = now + 100
    simp.propagate(now)
    print('Time: ' + str(simp.epoch))
    print('Position: ' + str(simp.spacecraft[0].position(eci)))
    return 0

def ssCallback(sim):
    return sim.SimStep()

simulation.startGUI(stepCallback)
#for t in range(1474416005, 1474419100, 100):
#   simulation.propagate(t)
#   print('Position: ' + str(simulation.spacecraft[0].position(eci)))
