#include "ros/ros.h"
#include "turtle_control/first_srv.h"

int add(int a,int b)
{
  
   return a+b;
}

int muti(int a,int b)
{
    
    return a*b;
}
int sub(int a,int b)
{
    
    return a-b;
}
float divi(int a,int b)
{
    
    return a/b;
}

bool service_request(turtle_control::first_srv::Request &req, turtle_control::first_srv::Response &res)
{
    ROS_INFO("Request Num = %d",req.input_req1);
    ROS_INFO("Request Num = %d",req.input_req2);
    res.output_res1=add(req.input_req1,req.input_req2);
    res.output_res2=muti(req.input_req1,req.input_req2);
    res.output_res3= sub(req.input_req1,req.input_req2);
    res.output_res4= divi(req.input_req1,req.input_req2);
    ROS_INFO("Response: %d is add ", (res.output_res1));
    ROS_INFO("Response: %d is muti ", (res.output_res2));
    ROS_INFO("Response: %d is sub " , (res.output_res3));
    ROS_INFO("Response: %f is div " , (res.output_res4));
    return true;
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "addmuti_server");
    ros::NodeHandle nh;
    ros::ServiceServer service = nh.advertiseService("addmuti",service_request);
    ros::spin();

    return 0;

}
