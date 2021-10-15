#include "ros/ros.h"
#include "turtle_control/first_srv.h"

int main(int argc, char **argv)
{
       ros::init(argc, argv,"addmuti_client");
    ros::NodeHandle nh;

    ros::ServiceClient client = nh.serviceClient<turtle_control::first_srv>("addmuti");

    turtle_control::first_srv srv;
    srv.request.input_req1 = atoll(argv[1]);
    srv.request.input_req2 = atoll(argv[2]);
    if(client.call(srv))
        ROS_INFO("Finish send data");
    else
    {
        ROS_ERROR("Failed to call service");
        return 1;
    }
    
    return 0;
    
 

}