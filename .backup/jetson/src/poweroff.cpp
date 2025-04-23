#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <cstdlib>

void shutdownCallback(const std_msgs::Bool::ConstPtr& msg) {
    if (msg->data) {
        ROS_WARN("Shutdown command received. Shutting down now...");
        system("sudo shutdown now");
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "shutdown_node");
    ros::NodeHandle nh;

    ros::Subscriber sub = nh.subscribe("/shutdown_cmd", 10, shutdownCallback);

    ROS_INFO("Shutdown node is running. Waiting for shutdown command...");

    ros::spin();
    return 0;
}
