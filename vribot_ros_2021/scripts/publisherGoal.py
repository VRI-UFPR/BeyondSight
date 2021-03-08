# import rospy
# from geometry_msgs.msg import PoseStamped
#
# rospy.init_node("mynode")
#
# goal_publisher = rospy.Publisher("move_base_simple/goal", PoseStamped, queue_size=5)
#
# goal = PoseStamped()
#
# goal.header.seq = 1
# goal.header.stamp = rospy.Time.now()
# goal.header.frame_id = "map"
#
# goal.pose.position.x = 10.0
# goal.pose.position.y = 0.0
# goal.pose.position.z = 0.0
#
# goal.pose.orientation.x = 0.0
# goal.pose.orientation.y = 0.0
# goal.pose.orientation.z = 0.0
# goal.pose.orientation.w = 1.0
#
# goal_publisher.publish(goal)
#
# rospy.spin()


import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


# poses = [[1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0],
#          [1.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0],
#          [0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0],
#          [0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0]]


def movebase_client(currentPose):

    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    # goal.target_pose.pose.position.x = 1.0
    # goal.target_pose.pose.position.y = 0.0
    # goal.target_pose.pose.position.z = 0.0
    #
    # goal.target_pose.pose.orientation.x = 0.0
    # goal.target_pose.pose.orientation.y = 0.0
    # goal.target_pose.pose.orientation.z = 0.0
    # goal.target_pose.pose.orientation.w = 1.0

    goal.target_pose.pose.position.x = currentPose[0]
    goal.target_pose.pose.position.y = currentPose[1]
    goal.target_pose.pose.position.z = currentPose[2]

    goal.target_pose.pose.orientation.x = currentPose[3]
    goal.target_pose.pose.orientation.y = currentPose[4]
    goal.target_pose.pose.orientation.z = currentPose[5]
    goal.target_pose.pose.orientation.w = currentPose[6]

    client.send_goal(goal)
    wait = client.wait_for_result()
    if not wait:
        rospy.logerr("Action server not available!")
        rospy.signal_shutdown("Action server not available!")
    else:
        return client.get_result()


if __name__ == '__main__':

    # 1x1 square
    # poses = [[1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0],
    #          [1.0, 1.0, 0.0,  0.0, 0.0, -0.707, -0.707],
    #          [0.0, 1.0, 0.0,  0.0, 0.0, -1.0, 0.0],
    #          [0.0, 0.0, 0.0,  0.0, 0.0, -0.707, 0.707]]

    # poses = [[2.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0],
    #          [2.0, 2.0, 0.0,  0.0, 0.0, -0.707, -0.707],
    #          [0.0, 2.0, 0.0,  0.0, 0.0, -1.0, 0.0],
    #          [0.0, 0.0, 0.0,  0.0, 0.0, -0.707, 0.707]]

    # poses = [[2.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0],
    #          [0.0, -2.0, 0.0,  0.0, 0.0, -1.0, 0.0]]

    # poses = [[0.5, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0],
    #          [1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0],
    #          [1.5, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0],
    #          [2.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0],
    #          [0.0, -2.0, 0.0,  0.0, 0.0, -1.0, 0.0]]

    # poses = [[1.0, -1.0, 0.0,  0.0, 0.0, 0.0, 1.0],
    #          [1.0, 1.0, 0.0,  0.0, 0.0, -0.707, -0.707],
    #          [-1.0, 1.0, 0.0,  0.0, 0.0, -1.0, 0.0],
    #          [-1.0, -1.0, 0.0,  0.0, 0.0, -0.707, 0.707]]

    # poses = [[0.0, 0.0, 0.0,  0.0, 0.0, -0.707, -0.707],
    #          [0.0, 0.0, 0.0,  0.0, 0.0, -1.0, 0.0],
    #          [0.0, 0.0, 0.0,  0.0, 0.0, -0.707, 0.707],
    #
    #          [1.0, -1.0, 0.0,  0.0, 0.0, 0.0, 1.0],
    #          [1.0, 1.0, 0.0,  0.0, 0.0, -0.707, -0.707],
    #          [-1.0, 1.0, 0.0,  0.0, 0.0, -1.0, 0.0],
    #          [-1.0, -1.0, 0.0,  0.0, 0.0, -0.707, 0.707],
    #
    #          [1.0, -1.0, 0.0,  0.0, 0.0, 0.0, 1.0],
    #          [1.0, 1.0, 0.0,  0.0, 0.0, -0.707, -0.707],
    #          [-1.0, 1.0, 0.0,  0.0, 0.0, -1.0, 0.0],
    #          [-1.0, -1.0, 0.0,  0.0, 0.0, -0.707, 0.707],
    #
    #          [1.0, -1.0, 0.0,  0.0, 0.0, 0.0, 1.0],
    #          [1.0, 1.0, 0.0,  0.0, 0.0, -0.707, -0.707],
    #          [-1.0, 1.0, 0.0,  0.0, 0.0, -1.0, 0.0],
    #          [-1.0, -1.0, 0.0,  0.0, 0.0, -0.707, 0.707],
    #
    #          [1.0, -1.0, 0.0,  0.0, 0.0, 0.0, 1.0],
    #          [1.0, 1.0, 0.0,  0.0, 0.0, -0.707, -0.707],
    #          [-1.0, 1.0, 0.0,  0.0, 0.0, -1.0, 0.0],
    #          [-1.0, -1.0, 0.0,  0.0, 0.0, -0.707, 0.707]]


    poses = [[0.0, 0.0, 0.0,  0.0, 0.0, -0.707, -0.707],
             [0.0, 0.0, 0.0,  0.0, 0.0, -1.0, 0.0],
             [0.0, 0.0, 0.0,  0.0, 0.0, -0.707, 0.707]]


    try:
        rospy.init_node('movebase_client_py')
        for i in poses:
            result = movebase_client(i)
            if result:
                rospy.loginfo("Goal execution done!")
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
