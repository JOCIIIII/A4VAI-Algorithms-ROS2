# LIO third-party notices

- FAST-LIO upstream: `https://github.com/hku-mars/FAST_LIO`, GPL-2.0.
- ROS 2 patch target: `https://github.com/Ericsii/FAST_LIO_ROS2`, commit
  `2fffc570a25d0df172720bac034fbdb6a13d2162`. The delivery includes a small
  patch and the upstream license in `licenses/FAST_LIO_GPL-2.0.txt`; it does
  not vendor the full source tree.
- Livox ROS Driver 2: `https://github.com/Livox-SDK/livox_ros_driver2`.
  The source contract was checked at commit
  `21445540f0d100dc86a7e6df312dd70bbdb4afdf`. Its upstream license is copied
  to `licenses/livox_ros_driver2_LICENSE.txt`.
The missing runtime binaries and ROS workspaces are deliberately not represented
as included artifacts. Recipients build them in their own ROS 2 workspace.
