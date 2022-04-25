import roslibpy
import json
import time
import base64
import time
import cv2
import numpy as np
import roslibpy.actionlib
from .robot_gazebo_env import RobotGazeboEnv

class HockeyBotEnv(RobotGazeboEnv):
    """Superclass for all CubeSingleDisk environments.
    """

    def __init__(self, robot_name_space,controllers_list, logger=None):
        
        # init logger
        # self.viewer = rendering.Viewer(600, 400)
        self.logger = logger

        self.robot_name_space = robot_name_space
        self.controllers_list = controllers_list

        self.logger.debug("Start HockeyBotEnv INIT...")


        # We launch the init function of the Parent Class robot_gazebo_env.RobotGazeboEnv
        super(HockeyBotEnv, self).__init__( logger = self.logger,
                                            controllers_list=self.controllers_list,
                                            robot_name_space=self.robot_name_space,
                                            reset_controls=False)
    
        self.gazebo.unpauseSim()

        self.compress_image = None
        self.joint_states = None
        self.puck_pos = None
        self.goal = None

        # defin service
        self._link_state_serv = roslibpy.Service(self.client, '/gazebo/get_link_state', 'gazebo_msgs/GetLinkState')


        # define subscriber here
        roslibpy.Topic(self.client, '/hockey_robot/camera1/image_raw/compressed', 'sensor_msgs/CompressedImage').subscribe(
            self._compress_image_callback)

        # define publisher here
        self._track_pub = roslibpy.Topic(self.client, 'hockey_robot/joint3_position_controller/command', 'std_msgs/Float64')
        self._racket_pub = roslibpy.Topic(self.client, 'hockey_robot/joint4_position_controller/command', 'std_msgs/Float64')
        self._reset_puck_pub = roslibpy.Topic(self.client,'/hockey_robot/puck/reset_pose', 'std_msgs/String')

        self.gazebo.pauseSim()
        self.logger.debug("Finished HockeyBotEnv INIT...")

    # def render(self,mode='human', close=False):
    #             # 下面就可以定义你要绘画的元素了
    #     line1 = rendering.Line((100, 300), (500, 300))
    #     line2 = rendering.Line((100, 200), (500, 200))
    #     # 给元素添加颜色
    #     line1.set_color(0, 0, 0)
    #     line2.set_color(0, 0, 0)
    #     # 把图形元素添加到画板中
    #     self.viewer.add_geom(line1)
    #     self.viewer.add_geom(line2)

    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _compress_image_callback(self,data):
        base64_bytes = data['data'].encode('ascii')
        image_bytes = base64.b64decode(base64_bytes)
        jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
        self.compress_image = cv2.imdecode(jpg_as_np, flags=1)


    def _puck_pos_callback(self, data):
        pos_string = data['data']
        puck_pos = json.loads(pos_string)
        self.puck_pos = puck_pos

    # Methods that the TrainingEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TrainingEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()
        

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TrainingEnvironment will need.
    # ----------------------------
    def move_racket(self, x, y , epsilon=0.05, update_rate=10):
        """
        It will move the base based on the linear and angular speeds given.
        It will wait untill those twists are achived reading from the odometry topic.
        :param linear_pos: Speed in the X axis of the robot base frame
        :param angular_pos: Speed of the angular turning of the robot base frame
        :param epsilon: Acceptable difference between the speed asked and the odometry readings
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        self.logger.debug("HockeyBot Move position>> ({},{})".format(x,y))

        self._track_pub.publish(roslibpy.Message({'data': x}))
        self._racket_pub.publish(roslibpy.Message({'data': y}))
        self.wait_until_pos_achieved(x,y)
        #self.wait_until_twist_achieved(cmd_vel_value,epsilon,update_rate)
        # Weplace a waitof certain amiunt of time, because this twist achived doesnt work properly
        # time.sleep(0.2)
    def get_compress_image(self):
        return self.compress_image

    def get_actual_racket_pose(self,racket_name):
        return self.gazebo.get_link_position(link_name=racket_name,model_name=self.robot_name_space)
    
    def wait_until_pos_achieved(self,x,y,epsilon= 0.05, update_rate= 0.1):
        """
        We wait for the cmd_vel twist given to be reached by the robot reading
        from the odometry.
        :param cmd_vel_value: Twist we want to wait to reach.
        :param epsilon: Error acceptable in odometry readings.
        :param update_rate: Rate at which we check the odometry.
        :return:
        """
        self.logger.debug("START wait_until_pos_achieved...")

        # start_wait_time = roslibpy.Time().to_sec()
        end_wait_time = 0.0
        epsilon = 0.05

        # self.logger.debug("Desired Twist Cmd>>" + str(cmd_vel_value))
        self.logger.debug("epsilon>>" + str(epsilon))


        hori_pos_plus = x + epsilon
        hori_pos_minus = x - epsilon
        vert_pos_plus = y + epsilon
        vert_pos_minus = y - epsilon

        while self.client.is_connected and self.joint_states is not None:
            x_pos = self.joint_states['position'][1]
            y_pos = self.joint_states['position'][3]

            self.logger.debug("Track pos =" + str(x_pos) + ", ?RANGE=[" + str(hori_pos_plus) + ","+str(hori_pos_minus)+"]")
            self.logger.debug("Racket pos =" + str(y_pos) + ", ?RANGE=[" + str(vert_pos_plus) + ","+str(vert_pos_minus)+"]")

            hori_pos_close = hori_pos_minus < x_pos < hori_pos_plus
            vert_pos_close = vert_pos_minus < y_pos < vert_pos_plus

            if hori_pos_close and vert_pos_close:
                self.logger.debug("Reached position!")
                # end_wait_time = roslibpy.Time().to_sec()
                break
            self.logger.debug("Not there yet, keep waiting...")
            time.sleep(update_rate)

        # delta_time = end_wait_time- start_wait_time
        # self.logger.debug("[Wait Time=" + str(delta_time)+"]")

        self.logger.debug("END wait_until_pos_achieved...")