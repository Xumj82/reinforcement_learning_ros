import imp
import roslibpy
from gazebo_env.gazebo_env import RobotGazeboEnv
from gazebo_env.gazebo_connection import GazeboConnection
from gazebo_env.controllers_connection import ControllersConnection

client = roslibpy.Ros(host='localhost', port=9090)
client.run()
env = ControllersConnection(client,namespace='hockey_robot',controllers_list= ['joint3_position_controller','joint4_position_controller'])
env.reset_controllers()
client.terminate()