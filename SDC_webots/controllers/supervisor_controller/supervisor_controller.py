"""supervisor_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor

TIME_STEP = 32

# create the Robot instance.
robot = Supervisor()

#get pedestrian's translation
#pedestrian_node = robot.getFromDef('PEDESTRIAN')
#translation_pedestrian = pedestrian_node.getField('translation')

#get vehicle's position
vehicle_node = robot.getFromDef('VEHICLE')
#translation_vehicle = vehicle_node.getField('translation')
#xveh, yveh, zveh = translation_vehicle.getSFVec3f()
xveh, yveh, zveh = vehicle_node.getPosition()

#get oil barrel's translation
obarrel_node = robot.getFromDef('BARREL')
translation_barrel = obarrel_node.getField('translation')


occurrence = 0
duration = 0
barrel = False
barrel_counter = 0
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(TIME_STEP) != -1:
    if occurrence%60 == 0 and barrel == False:
        #throw a barrel
        barrel = True
        xveh, yveh, zveh = vehicle_node.getPosition()
        barrel_value = [xveh - 10, yveh, zveh + 0.25]
        translation_barrel.setSFVec3f(barrel_value)
        #start counting duration
        duration += 1
        barrel_counter += 1
        #print("Barrel ", barrel_counter, barrel_value)
        #print
    elif barrel == True and duration == 10:
        #remove barrel
        barrel = False
        duration = 0
        barrel_value = [0.0, 0.0, 0.0]
        translation_barrel.setSFVec3f(barrel_value)
    elif barrel == True:
         #increase duration only
         duration += 1
    #out of the if-statement, increase occurrence    
    occurrence += 1

# Enter here exit cleanup code.
