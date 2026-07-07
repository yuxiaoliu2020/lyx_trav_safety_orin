##! /home/liu/miniforge3/envs/lyx/bin/python
# filepath: /home/lyx/lyx_trav_risk_nav/src/nav_realcar/scripts/tf_broadcaster.py

import rospy
import tf2_ros
import geometry_msgs.msg
from nav_msgs.msg import Odometry

class FixpositionTfBroadcaster:
    def __init__(self):
        rospy.init_node('fp_tf_broadcaster')
        
        # 获取参数
        self.pose_topic = rospy.get_param('~pose_topic', '/fixposition/odometry_enu')
        
        # 创建TF广播器
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        # 订阅位姿话题
        rospy.Subscriber(self.pose_topic, Odometry, self.pose_callback)
        
        rospy.loginfo(f"TF广播器已初始化，监听话题: {self.pose_topic}")

    def pose_callback(self, msg):
        """处理Fixposition位姿消息，发布FP_ENU0到FP_POI的变换"""
        try:
            # 创建变换消息
            transform = geometry_msgs.msg.TransformStamped()
            transform.header = msg.header  # 使用相同的时间戳和frame_id
            transform.child_frame_id = msg.child_frame_id  # 应该是"FP_POI"
            
            # 复制位置和方向
            transform.transform.translation.x = msg.pose.pose.position.x
            transform.transform.translation.y = msg.pose.pose.position.y
            transform.transform.translation.z = msg.pose.pose.position.z
            transform.transform.rotation = msg.pose.pose.orientation
            
            # 发布变换
            self.tf_broadcaster.sendTransform(transform)
            
        except Exception as e:
            rospy.logerr(f"发布TF变换时出错: {e}")

if __name__ == '__main__':
    try:
        broadcaster = FixpositionTfBroadcaster()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass