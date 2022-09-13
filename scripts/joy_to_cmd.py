#!/usr/bin/env python3

import rospy, math

from std_msgs.msg import Int16, Float32
from sensor_msgs.msg import Joy

class Controller:

    def __init__(self):


        self.hubpub = rospy.Publisher("/hub_speed", Int16, queue_size=10)
        self.steerpub = rospy.Publisher("/steer_speed", Int16, queue_size=10)
        self.hub_msg = Int16()
        self.steer_msg = Int16()

        self.joysub = rospy.Subscriber("/joy", Joy, self.subCB)

    def subCB(self, msg):

        ax = msg.axes
        self.hub_msg.data = int(ax[1] * 256)
        self.steer_msg.data = int(ax[3] * 256)

        self.hubpub.publish(self.hub_msg)
        self.steerpub.publish(self.steer_msg)

        print(self.steer_msg, " | ", self.hub_msg)


if __name__ == '__main__': 

    rospy.init_node("joy_to_cmd")

    c = Controller()
    rospy.spin()
