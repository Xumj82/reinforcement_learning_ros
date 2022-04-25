#!/usr/bin/env python
import logging
import logging.handlers
import datetime
import roslibpy
from std_srvs.srv import Empty
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest,GetModelState
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

class GazeboConnection():

    def __init__(self, client , reset_world_or_sim, logger = None , max_retry = 20,):

        self.client = client
        self._max_retry = max_retry

        if logger is None:
            self.logger = logging.getLogger('gazebo')
            self.logger.setLevel(logging.DEBUG)

            rf_handler = logging.handlers.TimedRotatingFileHandler('log/gazebo.log', when='midnight', interval=1, backupCount=7, atTime=datetime.time(0, 0, 0, 0))
            rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

            f_handler = logging.FileHandler('log/gazebo_error.log')
            f_handler.setLevel(logging.ERROR)
            f_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))
        else:
            self.logger = logger

        self.reset_world_or_sim = reset_world_or_sim
        self.unpause = roslibpy.Service(client,'/gazebo/unpause_physics', 'std_srvs/Empty')
        self.pause = roslibpy.Service(client,'/gazebo/pause_physics', 'std_srvs/Empty')
        self.reset_simulation_proxy = roslibpy.Service(client,'/gazebo/reset_simulation', 'std_srvs/Empty')
        self.reset_world_proxy = roslibpy.Service(client,'/gazebo/reset_world', 'std_srvs/Empty')
        self.set_model_state = roslibpy.Service(self.client, '/gazebo/set_model_state', 'gazebo_msgs/SetModelState')
        self.link_state_serv = roslibpy.Service(self.client, '/gazebo/get_link_state', 'gazebo_msgs/GetLinkState')
        self.pauseSim()

    def pauseSim(self):
        self.logger.debug("PAUSING service found...") 
        paused_done = False
        counter = 0
        while not paused_done and self.client.is_connected:
            if counter < self._max_retry:
                try:
                    self.logger.debug("PAUSING service calling...")
                    self.pause.call(roslibpy.ServiceRequest())
                    paused_done = True
                    self.logger.debug("PAUSING service calling...DONE")
                except Exception as e:
                    counter += 1
                    self.logger.warning("/gazebo/pause_physics service call failed")
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo pause service"
                self.logger.error(error_message)
                assert False, error_message

        self.logger.debug("PAUSING FINISH")

    def unpauseSim(self):
        self.logger.debug("UNPAUSING service found...")
        unpaused_done = False
        counter = 0
        while not unpaused_done and self.client.is_connected:
            if counter < self._max_retry:
                try:
                    self.logger.debug("UNPAUSING service calling...")
                    self.unpause.call(roslibpy.ServiceRequest())
                    unpaused_done = True
                    self.logger.debug("UNPAUSING service calling...DONE")
                except Exception as e:
                    counter += 1
                    self.logger.warning("/gazebo/unpause_physics service call failed...Retrying "+str(counter))
            else:
                error_message = "Maximum retries done"+str(self._max_retry)+", please check Gazebo unpause service"
                self.logger.error(error_message)
                assert False, error_message

        self.logger.debug("UNPAUSING FiNISH")


    def resetSim(self):
        """
        This was implemented because some simulations, when reseted the simulation
        the systems that work with TF break, and because sometime we wont be able to change them
        we need to reset world that ONLY resets the object position, not the entire simulation
        systems.
        """
        if self.reset_world_or_sim == "SIMULATION":
            self.logger.debug("SIMULATION RESET")
            self.resetSimulation()
        elif self.reset_world_or_sim == "WORLD":
            self.logger.debug("WORLD RESET")
            self.resetWorld()
        elif self.reset_world_or_sim == "NO_RESET_SIM":
            self.logger.debug("NO RESET SIMULATION SELECTED")
        else:
            self.logger.debug("WRONG Reset Option:"+str(self.reset_world_or_sim))

    def resetSimulation(self):
        try:
            self.reset_simulation_proxy.call(roslibpy.ServiceRequest())
        except Exception as e:
            self.logger.warning("/gazebo/reset_simulation service call failed")

    def resetWorld(self):
        try:
            self.reset_world_proxy.call(roslibpy.ServiceRequest())
        except Exception as e:
            self.logger.warning("/gazebo/reset_world service call failed")

    def init_values(self):

        self.resetSim()

        if self.start_init_physics_parameters:
            self.logger.debug("Initialising Simulation Physics Parameters")
            self.init_physics_parameters()
        else:
            self.logger.debug("NOT Initialising Simulation Physics Parameters")

    def get_link_position(self,link_name, model_name):
        val ={
            # 'model_name':model_name,
            'link_name': model_name+'::'+link_name,
            'reference_frame': 'world'
        }
        request = roslibpy.ServiceRequest(values=val)
        response = self.link_state_serv.call(request)
        link_pos = response['link_state']['pose']['position']
        return link_pos
    
    def set_model_position(self,model_name,position):
        state_msg ={
            'model_state':{
                'model_name': model_name,
                'pose':{'position':position,'orientation':{'x':0,'y':0,'z':0,'w':0}},
                'twist':{ 'linear': {'x': 0.0 , 'y': 0 ,'z': 0 },'angular': { 'x': 0.0 , 'y': 0 , 'z': 0.0 }},
                'reference_frame': 'world'
            }
        }
        request = roslibpy.ServiceRequest(values=state_msg)
        self.set_model_state.call(request)