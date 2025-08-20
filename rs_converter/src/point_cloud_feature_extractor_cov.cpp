#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include <Eigen/Dense>

#include <random>
#include <vector>
#include <unordered_map>
#include <algorithm>

struct VoxelData {
  std::vector<Eigen::Vector3f> points;
};

class PointCloudFeatureExtractor : public rclcpp::Node {
public:
  PointCloudFeatureExtractor()
  : rclcpp::Node("pointcloud_feature_extractor_cov")
  {
    // Parameters (you can override at runtime)
    declare_parameter<std::string>("input_topic", "/filtered/pointcloud");
    declare_parameter<std::string>("output_topic", "/pointcloud_features");
    declare_parameter<double>("leaf_size_x", 0.5);
    declare_parameter<double>("leaf_size_y", 0.5);
    declare_parameter<double>("leaf_size_z", 0.5);
    declare_parameter<int>("target_points", 256);

    input_topic_  = get_parameter("input_topic").as_string();
    output_topic_ = get_parameter("output_topic").as_string();
    leaf_x_       = get_parameter("leaf_size_x").as_double();
    leaf_y_       = get_parameter("leaf_size_y").as_double();
    leaf_z_       = get_parameter("leaf_size_z").as_double();
    target_pts_   = get_parameter("target_points").as_int();

    // Sensor-data QoS (best-effort, low latency)
    auto qos = rclcpp::SensorDataQoS();

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, qos,
      std::bind(&PointCloudFeatureExtractor::pointCloudCallback, this, std::placeholders::_1));

    pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(output_topic_, 1);

    RCLCPP_INFO(get_logger(), "Listening on '%s', publishing '%s' (leaf=%.2f,%.2f,%.2f, target=%d)",
                input_topic_.c_str(), output_topic_.c_str(), leaf_x_, leaf_y_, leaf_z_, target_pts_);
  }

private:
  void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // Convert ROS2 -> PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *input_cloud);

    // If empty or too small, add demo points to reach at least target_pts_
    if (input_cloud->empty() || static_cast<int>(input_cloud->points.size()) < target_pts_) {
      generateDemoPoints(*input_cloud, target_pts_ - static_cast<int>(input_cloud->points.size()));
    }

    // Voxel grid downsample (but we keep a mapping of original points to voxel index)
    pcl::VoxelGrid<pcl::PointXYZ> voxel;
    voxel.setInputCloud(input_cloud);
    voxel.setLeafSize(static_cast<float>(leaf_x_),
                      static_cast<float>(leaf_y_),
                      static_cast<float>(leaf_z_));
    voxel.setSaveLeafLayout(true);

    pcl::PointCloud<pcl::PointXYZ>::Ptr voxelized(new pcl::PointCloud<pcl::PointXYZ>);
    voxel.filter(*voxelized);

    // Map: voxel index -> points inside voxel
    std::unordered_map<int, VoxelData> voxel_map;
    voxel_map.reserve(input_cloud->points.size());

    for (const auto &p : input_cloud->points) {
      int idx = voxel.getCentroidIndex(p);
      if (idx != -1) {
        voxel_map[idx].points.emplace_back(p.x, p.y, p.z);
      }
    }

    // Compute features per voxel; output PointXYZI (intensity = sum of eigenvalues)
    pcl::PointCloud<pcl::PointXYZI>::Ptr feature_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    calculateFeatures(voxel_map, *feature_cloud);

    // Ensure exactly target_pts_
    ensureTargetCount(*feature_cloud, target_pts_);

    // Convert to ROS2 message and publish
    sensor_msgs::msg::PointCloud2 out_msg;
    pcl::toROSMsg(*feature_cloud, out_msg);
    out_msg.header = msg->header;  // keep frame/time
    pub_->publish(out_msg);
  }

  // --- Utilities ---

  void generateDemoPoints(pcl::PointCloud<pcl::PointXYZ> &cloud, int n)
  {
    if (n <= 0) return;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis_y(-7.f, 7.f);
    std::uniform_real_distribution<float> dis_z(-4.f, 4.f);

    cloud.points.reserve(cloud.points.size() + static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      pcl::PointXYZ pt;
      pt.x = 7.0f;
      pt.y = dis_y(gen);
      pt.z = dis_z(gen);
      cloud.points.push_back(pt);
    }
    cloud.width = static_cast<uint32_t>(cloud.points.size());
    cloud.height = 1;
    cloud.is_dense = false;
  }

  static Eigen::Vector3f centroidOf(const std::vector<Eigen::Vector3f> &pts)
  {
    Eigen::Vector3f c(0.f, 0.f, 0.f);
    for (const auto &p : pts) c += p;
    c /= static_cast<float>(pts.size());
    return c;
  }

  static Eigen::Matrix3f covarianceOf(const std::vector<Eigen::Vector3f> &pts, const Eigen::Vector3f &c)
  {
    Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
    for (const auto &p : pts) {
      Eigen::Vector3f d = p - c;
      cov += d * d.transpose();
    }
    cov /= static_cast<float>(pts.size());
    return cov;
  }

  void calculateFeatures(const std::unordered_map<int, VoxelData> &voxels,
                         pcl::PointCloud<pcl::PointXYZI> &out_cloud)
  {
    out_cloud.clear();
    out_cloud.reserve(voxels.size());

    for (const auto &kv : voxels) {
      const auto &vpts = kv.second.points;
      if (vpts.size() < 3) continue; // need at least 3 for covariance

      Eigen::Vector3f c = centroidOf(vpts);
      Eigen::Matrix3f cov = covarianceOf(vpts, c);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
      Eigen::Vector3f evals = es.eigenvalues();

      pcl::PointXYZI fp;
      fp.x = c.x();
      fp.y = c.y();
      fp.z = c.z();
      fp.intensity = evals.sum(); // store sum of eigenvalues
      out_cloud.points.push_back(fp);
    }

    out_cloud.width = static_cast<uint32_t>(out_cloud.points.size());
    out_cloud.height = 1;
    out_cloud.is_dense = false;
  }

  static void ensureTargetCount(pcl::PointCloud<pcl::PointXYZI> &cloud, int target)
  {
    if (static_cast<int>(cloud.points.size()) > target) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::shuffle(cloud.points.begin(), cloud.points.end(), gen);
      cloud.points.resize(static_cast<size_t>(target));
    } else if (static_cast<int>(cloud.points.size()) < target) {
      std::random_device rd;
      std::mt19937 gen(rd());

      while (static_cast<int>(cloud.points.size()) < target) {
        if (!cloud.points.empty()) {
          std::uniform_int_distribution<size_t> dist(0, cloud.points.size() - 1);
          cloud.points.push_back(cloud.points[dist(gen)]);  // duplicate random points
        } else {
          // if empty, seed with demo-style dummy point
          pcl::PointXYZI p;
          p.x = 7.0f;
          p.y = std::uniform_real_distribution<float>(-7.f, 7.f)(gen);
          p.z = std::uniform_real_distribution<float>(-4.f, 4.f)(gen);
          p.intensity = 255.0f;
          cloud.points.push_back(p);
        }
      }
    }

    cloud.width = static_cast<uint32_t>(target);
    cloud.height = 1;
    cloud.is_dense = false;
  }

private:
  // params
  std::string input_topic_, output_topic_;
  double leaf_x_{0.5}, leaf_y_{0.5}, leaf_z_{0.5};
  int target_pts_{256};

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudFeatureExtractor>());
  rclcpp::shutdown();
  return 0;
}

