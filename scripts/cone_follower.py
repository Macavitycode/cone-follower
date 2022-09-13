#!/bin/env python3

import cv2
import time
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image

# install scikit-spatial
from skspatial.objects import Plane, Line, Points
from skspatial.plotting import plot_3d

import numpy as np

class Controller:

    def __init__(self):

        self.speed_scaler = 0.1

        self.CamSub = rospy.Subscriber("/ackermann_vehicle/cam/image_raw", Image, self.CamCB)
        self.CmdPub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.CamPub = rospy.Publisher("/ackermann_vehicle/cam/proc", Image, queue_size=1)

        self.lower_orange = np.array([0,30,30])
        self.upper_orange = np.array([45,255,255])

        self.cont_min_size = 60
        self.cont_max_x = 60
        self.cont_max_y = 30

        self.img_min_x = 1280
        self.img_min_y = 720
        self.img_max_x = 0
        self.img_max_y = 0

        self.t_padding = 500
        self.z_min = 10
        self.z_max = 10000

        self.lht = [0 + self.t_padding, 0, self.z_min]
        self.llt = [0 + self.t_padding, 0, self.z_max]

        self.rht = [self.img_min_x - self.t_padding, 0, self.z_min]
        self.rlt = [self.img_min_x - self.t_padding, 0, self.z_max]

        self.base_plane = Plane([0, 0, 0], normal=[0, 0, 1])

        self.hw = self.img_min_x/2
        self.initial_left =     [[0 - self.t_padding, 360, 0]]
        self.initial_right =    [[1280 + self.t_padding, 360, 0]]
        self.z_scale = 10e-3

        self.rect_min_size = 20
        self.rect_max_size = int(pow(720 * 1280/3, 1) - 100)
        self.count = 0

        self.history_size = 50
        self.integral_control_scale_x = 0.4
        self.derivative_control_scale_x = 0.4
        self.center_queue = [640] * self.history_size
        self.c_mean_queue = self.center_queue
        self.dir = self.hw
        self.rad_scale = 1.57

    def CamCB(self, msg):

        img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")

        cmdmsg = self.img_proc(img)

        print("Publishing message: ", cmdmsg)

        self.CmdPub.publish(cmdmsg)


    def img_proc(self, img):

        #  print("showing img ", self.count)

        self.count = (self.count + 1) % 10

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_orange, self.upper_orange)

        contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        contours = np.asarray(contours)
        
        points_3d = []

        for contour in contours:
            (x,y,w,h) = cv2.boundingRect(contour)
            self.img_min_x, self.img_max_x = min(x, self.img_min_x), max(x+w, self.img_max_x)
            self.img_min_y, self.img_max_y = min(y, self.img_min_y), max(y+h, self.img_max_y)
            if w > self.cont_min_size:
                if w < h:
                    points_3d.append([x+w/2, y+h/2, self.z_scale * (self.rect_max_size - (w*h))])
                    cv2.rectangle(img, (x,y), (x+w,y+h), (0, 255, 0), 2)
                else:
                    hw = w/2
                    points_3d.append([x+(w*1/4), y+h/2, self.z_scale * (self.rect_max_size - (w*h))])
                    points_3d.append([x+(w*3/4), y+h/2, self.z_scale * (self.rect_max_size - (w*h))])
                    cv2.rectangle(img, (x,y), (int(x+hw),y+h), (0, 255, 0), 2)
                    cv2.rectangle(img, (int(x+hw),y), (x+w,y+h), (0, 255, 0), 2)

        try:
            left_points =  Points(points_3d + self.initial_left)
            right_points = Points(points_3d + self.initial_right)

            left_line_fit = Line.best_fit(left_points)
            right_line_fit = Line.best_fit(right_points)

            plane_left_line = self.base_plane.project_line(left_line_fit)
            plane_right_line = self.base_plane.project_line(right_line_fit)
            point_intersect = plane_right_line.intersect_line(plane_left_line)

            c = [int(i) for i in point_intersect[0:2]]
            #  print(c)

            [projected_left_x_lt, projected_left_y_lt,    _] = left_line_fit.project_point(self.llt)
            [projected_right_x_lt, projected_right_y_lt,  _] = right_line_fit.project_point(self.rlt)

            [projected_left_x_ht, projected_left_y_ht,  _] = left_line_fit.project_point(self.lht)
            [projected_right_x_ht, projected_right_y_ht,_] = right_line_fit.project_point(self.rht)

            #  print([projected_left_x_0, projected_left_y_0])
            #  print([projected_left_x_ht, projected_left_y_ht])
            #  print([projected_right_x_0, projected_right_y_0])
            #  print([projected_right_x_ht, projected_right_y_ht])

            pl1 = [int(i) for i in [projected_left_x_lt, projected_left_y_lt]]
            pl2 = [int(i) for i in [projected_left_x_ht, projected_left_y_ht]]
            pr1 = [int(i) for i in [projected_right_x_lt, projected_right_y_lt]]
            pr2 = [int(i) for i in [projected_right_x_ht, projected_right_y_ht]]

            #  print(pl1, pl2, pr1, pr2)

            cv2.line(img, pl1, pl2, (255, 0, 0), 2)
            cv2.line(img, pr1, pr2, (255, 0, 0), 2)
            cv2.line(img, (640, 0), (640, 720), (255, 255, 0), 2)
            cv2.circle(img, c, 10, (0, 0, 255), 2)
            
            self.center_queue.append(c[0])
            self.center_queue.pop(0)

            self.c_mean_queue.append(sum(self.center_queue)/self.history_size)
            self.c_mean_queue.pop(0)
            
            icx = (self.c_mean_queue[0] - self.c_mean_queue[self.history_size - 1]) * self.integral_control_scale_x
            dcx = (self.dir - c[0]) * self.derivative_control_scale_x
            self.dir = c[0] + dcx + icx

            d1 = [int(self.dir), 450]
            d2 = [640, 700]
            cv2.line(img, d1, d2, (0, 255, 255), 2)

            angle = ((self.dir - self.hw)/self.hw) * self.rad_scale
            cmdmsg = Twist()
            cmdmsg.linear.x = 0.1
            cmdmsg.angular.z = angle

        except:
            cmdmsg = Twist()


        imgmsg = CvBridge().cv2_to_imgmsg(img, encoding="passthrough")
        self.CamPub.publish(imgmsg)

        return cmdmsg

if __name__ == "__main__":
    rospy.init_node("cone_follower_node")
    print("Started node")
    c = Controller()
    rospy.spin()
