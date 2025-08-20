#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <random>

std::default_random_engine gen;
std::string lidar_topic = "/airsim_node/SimpleFlight/lidar/points/velo"; 

// VLP-16 
int N_SCAN = 16;
int Horizon_SCAN = 1800;

static int RING_ID_MAP_16[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 15, 14, 13, 12, 11, 10, 9, 8
};

struct PointXYZIRT {
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIRT,
    (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)
    (uint16_t, ring, ring)(float, time, time))

class RSConverter : public rclcpp::Node {
public:
    RSConverter()
        : Node("rs_converter")
    {
        // Publisher and subscriber
        sub_pc_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic, 10, std::bind(&RSConverter::lidar_handle, this, std::placeholders::_1));

        pub_pc_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/velodyne_points", 10);

        RCLCPP_INFO(this->get_logger(), "Listening to lidar topic...");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pc_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_pc_;

    template <typename T>
    void publish_points(T &new_pc, const sensor_msgs::msg::PointCloud2 &old_msg) {
        // pc properties
        new_pc->is_dense = true;

        // Publish
        sensor_msgs::msg::PointCloud2 pc_new_msg;
        pcl::toROSMsg(*new_pc, pc_new_msg);
        pc_new_msg.header = old_msg.header;
        pc_new_msg.header.frame_id = "velodyne";
        pub_pc_->publish(pc_new_msg);
    }

    void add_noise_to_point_cloud(pcl::PointCloud<PointXYZIRT>::Ptr &pc, double mean = 0, double std_dev = 0.00001) {
        for (auto &point : pc->points) {
            double noise_x = std::normal_distribution<double>(mean, std_dev)(gen);
            double noise_y = std::normal_distribution<double>(mean, std_dev)(gen);
            double noise_z = std::normal_distribution<double>(mean, std_dev)(gen);
            double noise_intensity = std::normal_distribution<double>(mean, std_dev)(gen);

            point.x += noise_x;
            point.y += noise_y;
            point.z += noise_z;
            point.intensity += noise_intensity;
        }
    }

    void lidar_handle(const sensor_msgs::msg::PointCloud2::SharedPtr pc_msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr pc(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<PointXYZIRT>::Ptr pc_new(new pcl::PointCloud<PointXYZIRT>());
        pcl::fromROSMsg(*pc_msg, *pc);

        // Convert to new point cloud
        for (size_t point_id = 0; point_id < pc->points.size(); ++point_id) {
            PointXYZIRT new_point;
            new_point.x = pc->points[point_id].x;
            new_point.y = -pc->points[point_id].y;
            new_point.z = -pc->points[point_id].z;
            new_point.intensity = 2550;

            // 16 ring. Index range is 0~15, from up to down.
            float ang_bottom = 15.0 + 0.1;
            float ang_res_y = 2.0;
            float verticalAngle = atan2(new_point.z, sqrt(new_point.x * new_point.x + new_point.y * new_point.y)) * 180 / M_PI;
            float rowIdn = (verticalAngle + ang_bottom) / ang_res_y;

            new_point.ring = static_cast<uint16_t>(rowIdn);
            new_point.time = (static_cast<float>(point_id) / N_SCAN) * 0.1 / Horizon_SCAN;

            pc_new->points.push_back(new_point);
        }

        // Add noise to the point cloud
        add_noise_to_point_cloud(pc_new, 0, 0.003);

        publish_points(pc_new, *pc_msg);
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<RSConverter>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

