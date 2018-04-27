#!/usr/bin/python
import polysat

t = polysat.TemperatureSensor(name='threeV_plTmpSensor', location='mb')
print "temperature sensor: " + str(t.read_temp())
sa = polysat.SolarAngleSensor(name='S_ANG', location='test')
print "solar angle sensor: " + str(sa.read_data())
acc = polysat.AccelerometerSensor(name='mb_accel', location='mb')
print "accelerometer: " + str(acc.read_data());
