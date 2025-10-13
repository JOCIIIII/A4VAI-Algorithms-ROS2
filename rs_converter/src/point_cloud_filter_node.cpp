#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/bool.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <pcl/common/common.h>
#include <random>
#include <vector>
#include <algorithm>

class PointCloudFilter : public rclcpp::Node
{
public:
    PointCloudFilter() : Node("point_cloud_filter_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        // Subscriber and Publisher Initialization
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10, std::bind(&PointCloudFilter::pointCloudCallback, this, std::placeholders::_1));
        
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/ground_truth/state", 10, std::bind(&PointCloudFilter::odomCallback, this, std::placeholders::_1));

        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered/pointcloud", 10);
        pub_rand_point_flag_ = this->create_publisher<std_msgs::msg::Bool>("/ca_rand_point_flag", 10);
        min_dist_pub_ = this->create_publisher<std_msgs::msg::Float32>("/min_distance", 10);
        closest_point_pub_ = this->create_publisher<geometry_msgs::msg::PointStamped>("/closest_point", 10);

        RCLCPP_INFO(this->get_logger(), "Point Cloud Filter Node Initialized...");
    }

private:
    bool is_rand_point = false;
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr input_cloud_msg)
    {
        // Convert ROS2 PointCloud2 to PCL PointCloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*input_cloud_msg, *input_cloud);

        // Conditional Removal Filter
        pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>());
        range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
            new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::GT, 0.4)));
        range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
            new pcl::FieldComparison<pcl::PointXYZ>("x", pcl::ComparisonOps::LT, 7.0)));
        range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
            new pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::GT, -7.0)));
        range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
            new pcl::FieldComparison<pcl::PointXYZ>("y", pcl::ComparisonOps::LT, 7.0)));
        range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
            new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::GT, -4.0)));
        range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(
            new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::LT, 4.0)));

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::ConditionalRemoval<pcl::PointXYZ> condrem;
        condrem.setInputCloud(input_cloud);
        condrem.setCondition(range_cond);

        condrem.filter(*filtered_cloud);

        is_rand_point = ensureMinimumPoints(filtered_cloud, 256);

        if (is_rand_point) {
            std_msgs::msg::Bool rand_point_msg;
            rand_point_msg.data = is_rand_point;
            pub_rand_point_flag_->publish(rand_point_msg);
        } else {
            std_msgs::msg::Bool rand_point_msg;
            rand_point_msg.data = false;
            pub_rand_point_flag_->publish(rand_point_msg);
        };

        filtered_cloud->width  = static_cast<uint32_t>(filtered_cloud->points.size());
        filtered_cloud->height = 1;         

        // Publish filtered point cloud
        sensor_msgs::msg::PointCloud2 filtered_msg;
        pcl::toROSMsg(*filtered_cloud, filtered_msg);
        filtered_msg.header.stamp = this->now();
        filtered_msg.header.frame_id = "velodyne";
        pub_->publish(filtered_msg);

        // Compute and publish minimum distance
        pcl::PointXYZ closest_point;
        float min_distance = computeMinimumDistance(filtered_cloud, closest_point);
        std_msgs::msg::Float32 min_distance_msg;
        min_distance_msg.data = min_distance;
        min_dist_pub_->publish(min_distance_msg);

        // Publish the closest point coordinates
        publishClosestPointCoordinates(closest_point);
    }

    bool ensureMinimumPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, int min_points)
    {
        bool is_rand_point = false;
        while (cloud->points.size() < min_points)
        {
            pcl::PointXYZ point;
            point.x = 60.0;
            point.y = std::rand() % 14 - 7.0;
            point.z = std::rand() % 8 - 4.0;
            cloud->points.push_back(point);
            is_rand_point = true;
        }
        std::cout << "is_rand_point: " << is_rand_point << std::endl;
        return is_rand_point;
    }

    float computeMinimumDistance(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::PointXYZ &closest_point)
    {
        float min_distance = std::numeric_limits<float>::max();
        for (const auto &point : cloud->points)
        {
            float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            if (distance < min_distance)
            {
                min_distance = distance;
                closest_point = point;
            }
        }
        return min_distance;
    }

    void publishClosestPointCoordinates(const pcl::PointXYZ &point)
    {
        geometry_msgs::msg::PointStamped closest_point_msg;
        closest_point_msg.header.stamp = this->now();
        closest_point_msg.header.frame_id = "velodyne";

        closest_point_msg.point.x = point.x;
        closest_point_msg.point.y = point.y;
        closest_point_msg.point.z = point.z;

        try
        {
            geometry_msgs::msg::PointStamped transformed_point;
            tf_buffer_.transform(closest_point_msg, transformed_point, "base_link", tf2::durationFromSec(1.0));
            closest_point_pub_->publish(transformed_point);
        }
        catch (const tf2::TransformException &ex)
        {
            RCLCPP_WARN(this->get_logger(), "Transform failed: %s", ex.what());
        }
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        latest_odom_ = *msg;
    }

    // ROS2 Parameters
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pub_rand_point_flag_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr min_dist_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr closest_point_pub_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    nav_msgs::msg::Odometry latest_odom_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudFilter>());
    rclcpp::shutdown();
    return 0;
}

