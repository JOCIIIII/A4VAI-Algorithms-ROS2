# 통합측 변경 내용 

## EstimatorStates 변경
EstimatorStates의 위치, 속도, 각속도, 자세 데이터 받는 부분 
VehicleLocalPosition, VehicleAttitude, VehicleAngularVelocity 메세지로 변경

## waypoint 받는 부분 추가 
wp_type_selection = 3 으로 Path planning의 waypoint사용하도록 변경

## heartbeat check 
	- 1초에 한번씩 heartbeat publish
	- 모든 모듈 heartbeat True 일시 heartbeat 수신시 실행하도록 main_attitude_control 변경
	
	- heartbeat check function 추가
	- heartbeat_publisher 추가 
	- heartbeat_subscriber 추가 
	- heartbeat flag 추가
	- heartbeat_timer 추가

## offboard control 삭제
	- offboard control부 Controller로 이전 

## 자세 명령 publisher 추가
