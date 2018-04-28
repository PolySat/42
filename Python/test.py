#!/usr/bin/python
import nasa42

sim = nasa42.Simulation()
eci = nasa42.WORLD_INERTIAL_FRAME

for t in range(1474416005, 1474419100, 100):
   sim.propagate(t)
   print('Position: ' + str(sim.spacecraft[0].position(eci)))
