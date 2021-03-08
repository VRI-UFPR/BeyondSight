import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

def movebase_client(currentPose):

    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

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
