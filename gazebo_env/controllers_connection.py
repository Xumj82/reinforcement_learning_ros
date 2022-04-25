import logging
import roslibpy
import time
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest, SwitchControllerResponse

class ControllersConnection():
    
    def __init__(self,client,logger:logging.Logger,namespace, controllers_list):

        self.logger = logger
        self.logger.debug("Start Init ControllersConnection")
        self.controllers_list = controllers_list
        self.switch_service_name = '/'+namespace+'/controller_manager/switch_controller'
        self.switch_service = roslibpy.Service(client,self.switch_service_name, 'controller_manager_msgs/SwitchController')
        self.logger.debug("END Init ControllersConnection")

    def switch_controllers(self, controllers_on, controllers_off, strictness=1):
        """
        Give the controllers you want to switch on or off.
        :param controllers_on: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :param controllers_off: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :return:
        """

        try:
            switch_request = roslibpy.ServiceRequest(
                values={'start_controllers':controllers_on,
                        'stop_controllers':controllers_off, 
                        'strictness':strictness,
                        'start_asap':False,
                        'timeout':0
                        })
            # switch_request_object.start_controllers = controllers_on
            # switch_request_object.start_controllers = controllers_off
            # switch_request_object.strictness = strictness

            switch_result = self.switch_service.call(switch_request)
            """
            [controller_manager_msgs/SwitchController]
            int32 BEST_EFFORT=1
            int32 STRICT=2
            string[] start_controllers
            string[] stop_controllers
            int32 strictness
            ---
            bool ok
            """
            print("Switch Result==>"+str(switch_result['ok']))

            return switch_result['ok']

        except Exception as e:
            print (self.switch_service_name+" service call failed")

            return None

    def reset_controllers(self):
        """
        We turn on and off the given controllers
        :param controllers_reset: ["name_controler_1", "name_controller2",...,"name_controller_n"]
        :return:
        """
        reset_result = False

        result_off_ok = self.switch_controllers(controllers_on = [],
                                controllers_off = self.controllers_list)

        print("Deactivated Controlers")

        if result_off_ok:
            print("Activating Controlers")
            result_on_ok = self.switch_controllers(controllers_on=self.controllers_list,
                                                    controllers_off=[])
            if result_on_ok:
                print("Controllers Reseted==>"+str(self.controllers_list))
                reset_result = True
            else:
                print("result_on_ok==>" + str(result_on_ok))
        else:
            print("result_off_ok==>" + str(result_off_ok))

        return reset_result

    def update_controllers_list(self, new_controllers_list):

        self.controllers_list = new_controllers_list