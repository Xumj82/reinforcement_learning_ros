import random
from turtle import position
import roslibpy
import roslibpy.actionlib

import cv2
import time
import numpy

from gym import spaces
from .hockey_bot_env import HockeyBotEnv

class HockeyTaskEnv(HockeyBotEnv):
    def __init__(self,action_names, number_actions,image_shape,logger = None):
        """
        This Task Env is designed for having the hockeybot in some kind of maze.
        It will learn how to move around the maze without crashing.
        """

        # ROSLauncher(rospackage_name="hockey_robot_gazebo",
        #             launch_file_name="air_hockey.launch",
        #             ros_ws_abspath=ros_ws_abspath)


        # Here we will add any init functions prior to starting the MyRobotEnv
        super(HockeyTaskEnv, self).__init__(robot_name_space='hockey_robot',
                                            controllers_list=['joint3_position_controller','joint4_position_controller'],
                                            logger = logger
                                            )

        self.action_names = action_names
        self.image_shape = image_shape
        # Only variable needed to be set here
        self.action_space = spaces.Discrete(number_actions)


        # We set the reward range, which is not compulsory but here we do it.
        self.reward_range = (-numpy.inf, numpy.inf)
        # Actions and Observations
        # self.dec_obs = rospy.get_param(
        #     "/hockeybot/number_decimals_precision_obs", 1)

        self.init_puck_position =  {'x':0, 'y':0, 'z':0.8051}
        self.puck_pose_queue = []
        self.puck_pose_queue_size = 2

        self.move_step = 0.01

        self.init_x_pos = 0
        self.init_y_pos = 0
        self.plan_x_pos = 0
        self.plan_y_pos = 0
        # self.n_observations = number_observations
        self.max_goal_y_value = 0.9 #blue
        self.min_goal_y_value = -0.9 #red

        self.max_goal_x_value = 0.2
        self.min_goal_x_value = -0.2

        self.max_x_value = 0.86
        self.min_x_value = 0
        self.max_y_value = 0.48
        self.min_y_value = -0.48

        self.move_stride = 0.1
        # self.max_orient_value = rospy.get_param('/hockeybot/max_orient_value')
        # self.min_orient_value = rospy.get_param('/hockeybot/min_orient_value')
        # self.max_pos_value = rospy.get_param('/hockeybot/max_pos_value')
        # self.min_pos_value = rospy.get_param('/hockeybot/min_pos_value')

        # self.max_goal_value = rospy.get_param('/hockeybot/max_goal_value')
        # self.min_goal_value = rospy.get_param('/hockeybot/min_goal_value')

        # self.goal_thr = rospy.get_param('/hockeybot/goal_th')

        self.logger.debug("n_observations===>"+str(image_shape))

        # high = numpy.array([self.max_x_value,self.max_y_value,self.max_orient_value, self.max_pos_value])
        # low = numpy.array([self.min_x_value,self.min_y_value,self.min_orient_value, self.min_pos_value])

        # We only use two integers
        self.observation_space = spaces.Box(0,255,[image_shape[0],image_shape[1],3])

        self.logger.debug("ACTION SPACES TYPE===>"+str(self.action_space))
        # print("OBSERVATION SPACES TYPE===>" + str(self.observation_space))

        # Rewards
        # self.goal_reward = 10
        # self.loss_reward = 10
        self.hit_reward = 1
        self.area_reward = 1
        self.win_reward = 200
        self.lose_reward = -200
        self.end_episode_points = 0

        self.cumulated_steps = 0.0

    

    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        self.move_racket(x=0,y=0)

        # self._reset_puck_pub.publish(roslibpy.Message({'data':'reset'}))
        return True

    def _set_model_pose(self):
        self.init_puck_position['y'] = random.choice([0.5,-0.5])
        self.gazebo.set_model_position(model_name='puck',position=self.init_puck_position)
        self.puck_pose_queue.clear()
        self.puck_pose_queue.append(self.init_puck_position)
        return True

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # self._set_init_pose()
        self.goal = 0
        # For Info Purposes
        self.cumulated_reward = 0.0
        # Set to false Done, because its calculated asyncronously
        self._episode_done = False

        # We wait a small ammount of time to start everything because in very fast resets, laser scan values are sluggish
        # and sometimes still have values from the prior position that triguered the done.
        time.sleep(1.0)

    def _set_action(self, action):
        """
        This set action will Set the linear and angular speed of the turtlebot2
        based on the action number given.
        :param action: The action integer that set s what movement to do next.
        """
        stride = self.move_stride
        self.logger.debug("Start Set Action ==>"+str(action))
        if action == 3 and self.plan_y_pos + stride < self.max_y_value: ## right
            self.plan_y_pos += stride
            self.move_racket(x=self.plan_x_pos, y=self.plan_y_pos)
        elif action == 2 and self.plan_y_pos - stride > self.min_y_value: ## left
            self.plan_y_pos -= stride
            self.move_racket(x=self.plan_x_pos, y=self.plan_y_pos)
        elif action == 1 and self.plan_x_pos - stride > self.min_x_value: ## backward
            self.plan_x_pos -= stride
            self.move_racket(x=self.plan_x_pos, y=self.plan_y_pos)
        elif action == 0 and self.plan_x_pos + stride < self.max_x_value: ## forward
            self.plan_x_pos += stride
            self.move_racket(x=self.plan_x_pos, y=self.plan_y_pos)
        else:
            action = 4
        
        self.logger.debug("END Set Action ==>"+str(action) +
                       ", NAME="+str(self.action_names[action]))

    def _get_obs(self):
        """
        Here we define what sensor data defines our robots observations
        To know which Variables we have acces to, we need to read the
        :return:
        """
        self.logger.debug("Start Get Observation ==>")
        # We get the laser scan data
        puck_pos = self.get_actual_puck_pose()
        if len(self.puck_pose_queue)<self.puck_pose_queue_size:
            self.puck_pose_queue.append(puck_pos)
        else:
            self.puck_pose_queue.pop(0)
            self.puck_pose_queue.append(puck_pos)

        img = self.get_compress_image()
        img = cv2.resize(img, self.image_shape)
        # self.logger.debug("img_size ==> "+ str(img.shape))
        return img

    def _is_done(self, observations):

        puck_pos = self.puck_pose_queue[-1]
        if self.min_goal_x_value<puck_pos['x'] <self.max_goal_x_value:
            if puck_pos['y'] > self.max_goal_y_value:
                self._episode_done = True
                self.end_episode_points = self.lose_reward
            if puck_pos['y'] < self.min_goal_y_value:
                self._episode_done = True
                self.end_episode_points = self.win_reward

        return self._episode_done

    def _compute_reward(self, observations, done):
        reward = 0
        puck_pos = self.puck_pose_queue[-1]
        puck_speed = self.puck_pose_queue[-1]['y']-self.puck_pose_queue[0]['y']
        racket_pos = self.get_actual_racket_pose(racket_name='pusher_2')
        puck_racket_distance = self.get_distance(puck_pos['x'],racket_pos['x'],puck_pos['y'],racket_pos['y'])
        if not done:
            # reward -= self.area_reward * self.get_distance(puck_pos['y'],puck_pos['x'],self.min_goal_y_value,0) # more close to other side goal, less punishment
            # reward += self.area_reward * self.get_distance(puck_pos['y'],puck_pos['x'],self.max_goal_y_value,0) # more close to self goal, less reward

            
            # puck_blue_goal_distance = self.get_distance(puck_pos['y'],puck_pos['x'],self.max_goal_y_value,0)
            # puck_red_goal_distance = self.get_distance(puck_pos['y'],puck_pos['x'],self.min_goal_y_value,0)

            # reward += (puck_blue_goal_distance/puck_red_goal_distance) * self.area_reward

            if  puck_pos['y'] >0 and puck_pos['y'] < racket_pos['y'] and puck_speed >= -0.01:
                reward -= self.hit_reward * abs(puck_racket_distance-0.095) # racket far away punish

            if  puck_pos['y'] > racket_pos['y']:
                reward -= self.hit_reward * 2
            
            # if abs(self.get_distance(puck_pos['y'],puck_pos['x'],racket_pos['y'],racket_pos['x'])) 
            # else:
            #     reward = self.turn_reward
        else:
            reward = self.end_episode_points
        
        self.logger.debug("Done:{} puck_racket_distance==>{} puck_speed==>{} reward==> {}".format(done, puck_racket_distance,puck_speed,reward))
        self.logger.debug("reward=" + str(reward))
        self.cumulated_reward += reward
        self.logger.debug("Cumulated_reward=" + str(self.cumulated_reward))
        self.cumulated_steps += 1
        self.logger.debug("Cumulated_steps=" + str(self.cumulated_steps))

        return reward

    def get_actual_puck_pose(self):
        return self.gazebo.get_link_position(link_name='link',model_name='puck')

    def get_distance(self,x1,x2,y1,y2):
        return ((((x2 - x1 )**2) + ((y2-y1)**2) )**0.5)