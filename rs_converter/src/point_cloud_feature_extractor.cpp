#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>
#include <vector>
#include <limits>

// Structure to store points within a voxel
struct VoxelData
{
    std::vector<Eigen::Vector3f> points;
};

class PointCloudFeatureExtractor : public rclcpp::Node
{
public:
    PointCloudFeatureExtractor()
        : Node("point_cloud_feature_extractor")
    {
        sub_pointcloud_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/filtered/pointcloud", 1, std::bind(&PointCloudFeatureExtractor::pointCloudCallback, this, std::placeholders::_1));

        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/fmu/out/vehicle_odometry", 1, std::bind(&PointCloudFeatureExtractor::odomCallback, this, std::placeholders::_1));

        pub_nearest_point_ = this->create_publisher<geometry_msgs::msg::PointStamped>(
            "/nearest_feature_point", 10);

        pub_features_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/pointcloud_features", 10);

        RCLCPP_INFO(this->get_logger(), "PointCloud Feature Extractor Initialized.");
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr input_cloud_msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*input_cloud_msg, *input_cloud);

        // Check if the point cloud is empty
        if (input_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received an empty point cloud. Filling with default points.");

            // Fill with 256 default points
            for (size_t i = 0; i < 256; ++i)
            {
                pcl::PointXYZ default_point;
                default_point.x = 10.0;
                default_point.y = 10.0;
                default_point.z = 10.0;
                input_cloud->points.push_back(default_point);
            }

            input_cloud->width = input_cloud->points.size();
            input_cloud->height = 1; // Unstructured point cloud
            input_cloud->is_dense = true;

            RCLCPP_INFO(this->get_logger(), "Filled empty cloud with 256 default points.");
        }

        // Voxel Grid Filtering
        pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
        voxel_filter.setInputCloud(input_cloud);
        voxel_filter.setLeafSize(0.5f, 0.5f, 0.5f);

        pcl::PointCloud<pcl::PointXYZ>::Ptr voxelized_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        voxel_filter.filter(*voxelized_cloud);

        if (voxelized_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Voxel grid filtering resulted in an empty point cloud.");
            return;
        }

        voxelized_cloud->width = voxelized_cloud->size();
        voxelized_cloud->height = 1;
        voxelized_cloud->is_dense = true;

        // Calculate features and publish
        pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        calculateFeatures(voxelized_cloud, feature_cloud);

        if (feature_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "No features were calculated from the point cloud.");
            return;
        }

        findNearestFeature(feature_cloud);

        // Publish the feature cloud
        sensor_msgs::msg::PointCloud2 feature_msg;
        pcl::toROSMsg(*feature_cloud, feature_msg);
        feature_msg.header = input_cloud_msg->header;
        pub_features_->publish(feature_msg);
    }

    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        drone_position_.x() = msg->pose.pose.position.x;
        drone_position_.y() = msg->pose.pose.position.y;
        drone_position_.z() = msg->pose.pose.position.z;
    }

    void calculateFeatures(const pcl::PointCloud<pcl::PointXYZ>::Ptr &voxelized_cloud,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr &feature_cloud)
    {
        for (const auto &point : voxelized_cloud->points)
        {
            pcl::PointXYZI feature_point;
            feature_point.x = point.x;
            feature_point.y = point.y;
            feature_point.z = point.z;
            feature_point.intensity = sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
            feature_cloud->points.push_back(feature_point);
        }
        feature_cloud->width = feature_cloud->points.size();
        feature_cloud->height = 1;
        feature_cloud->is_dense = true;
    }

    void findNearestFeature(const pcl::PointCloud<pcl::PointXYZI>::Ptr &feature_cloud)
    {
        double min_distance = std::numeric_limits<double>::max();
        Eigen::Vector3f nearest_point;

        for (const auto &point : feature_cloud->points)
        {
            double distance = (drone_position_ - Eigen::Vector3f(point.x, point.y, point.z)).norm();
            if (distance < min_distance)
            {
                min_distance = distance;
                nearest_point = Eigen::Vector3f(point.x, point.y, point.z);
            }
        }

        geometry_msgs::msg::PointStamped nearest_point_msg;
        nearest_point_msg.header.frame_id = "map";
        nearest_point_msg.header.stamp = this->get_clock()->now();
        nearest_point_msg.point.x = nearest_point.x();
        nearest_point_msg.point.y = nearest_point.y();
        nearest_point_msg.point.z = nearest_point.z();

        pub_nearest_point_->publish(nearest_point_msg);

        RCLCPP_INFO(this->get_logger(), "Published nearest feature point: (%.2f, %.2f, %.2f)",
                    nearest_point_msg.point.x, nearest_point_msg.point.y, nearest_point_msg.point.z);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pointcloud_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_features_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr pub_nearest_point_;

    Eigen::Vector3f drone_position_{0.0, 0.0, 0.0};
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PointCloudFeatureExtractor>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
