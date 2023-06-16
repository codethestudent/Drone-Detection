# Drone-Detection
경상국립대학교 캡스톤 디자인 프로젝트 - 비행체 추적을 위한 미사일 자세제어 시스템(딥러닝 객체 탐지 및 추적)

**프로젝트 목표**

![image](https://github.com/codethestudent/Drone-Detection/assets/96714243/565d900d-9b01-4450-a855-c31d9f231cdb)

![image](https://github.com/codethestudent/Drone-Detection/assets/96714243/bdb839e6-6bc9-493b-bb7f-d19768c7cad9)

**전체 시스템 구상도**

![image](https://github.com/codethestudent/Drone-Detection/assets/96714243/ccb31683-dd86-4163-b588-c110791e2223)


요구사항 수집 및 분석: 사물 인식을 통한 미사일 자세제어를 위한 요구사항을 수집하고 시나리오 작성을 통해 분석한다.
사물인식을 활용한 **비행체 추적 시스템 개발 절차**는 다음과 같다. 

**1. 비행체 추적 시스템 개념 설계**: 미사일 전방에 탑재할 카메라를 선정하고 구동 시스템의 프로세스를 설계한다.

**2. 사물 인식 모델 구현**: 요구사항에 적합한 비행체를 선정하고 해당 비행체에 대한 지속적인 학습으로 비행체의 인식 정확도를 높인다. 딥러닝 모델을 테스트해보면서 적합한 모델을 선정하고, 커스텀 데이터셋, 공개 데이터셋을 활용했을 때의 장단점을 분석한다. 또한, Tracker 등을 활용했을 때 생기는 이점들을 조사한다. 이 과정에서 mAP(정확도 지수)는 직접 대상 비행체를 촬영한 사진으로 테스트하여 0.60 이상이 되는 것을 목표로 한다. 

**3. 카메라 짐벌 제어**: 카메라가 비행체를 인식해 비행체가 카메라 화면의 중앙에 위치하도록 서보모터를 돌리는 짐벌 시스템을 구축한다. 비행체가 움직인다면 카메라의 정중앙에 비행체를 위치시키기 위해 서보모터를 돌리며 카메라 짐벌이 회전한다. 비행체의 중심점이 카메라의 너비와 폭의 각각 10%이내에 있을 때 중앙에 위치한다고 정의한다. 카메라에 인식된 비행체가 화면의 중앙에서 떨어진 거리로 회전각을 추정한다. 이 때 목표 오차범위는 5% 이다.

**4. 모듈 성능 테스트**: 사물 인식 카메라와 결합한 짐벌 시스템을 통합하여 카메라가 비행체를 0.5초 이내에 인식한 후 짐벌이 비행체를 지속적으로 추적하는지 테스트한다. 카메라가 작동하고 있을 때 카메라의 화면 속에 비행체를 날려 지나가게 하고 카메라 짐벌의 각도가 자동 조절되는지 확인한다.

2축 자세 시뮬레이터 개발 절차는 다음과 같다. 

**1. 비행체 추적 테스트 베드 설계**: 카메라 짐벌이 추적한 각도와 미사일의 동체 각도를 일치시켜야 한다. 프로젝트에서 실제 비행을 시키지 않으므로 카나드 제어 등과 같은 비행 제어로 인해 생기는 힘을 미사일 동체와 연결되는 2축 짐벌의 서보모터로 모사한다.

**2. 비행체 추적 테스트 베드 개발**: 미사일 모듈의 무게 중심 부분에 짐벌을 고정해 원하는 각도만큼 회전하는 테스트 베드를 개발해야 한다. 테스트 베드는 피치와 요에 해당하는 2축 짐벌로 구성해야 하고, 섬세한 각도 조절을 위해 서보모터를 부착해 자세 시뮬레이터를 개발한다.

**3. 구동 테스트 및 계측**: 테스트 베드의 반응속도가 요구조건에 충족하는지를 확인한다. 테스트베드는 모드 변경 신호 시 0.2초 이내에 카메라 짐벌 각만큼 서보모터를 조정해 미사일 동체 시선을 카메라 시선과 일치시키는지 확인한다.

Jetson Nano 4GB SUB에서 YOLOv5s와 OpenCV Tracker를 활용하여 객체(드론)를 탐지 및 추적
실제 추적 화면
![Screenshot from 2023-06-04 22-41-07](https://github.com/codethestudent/Drone-Detection/assets/96714243/445eecf3-9460-4c0d-9532-6ae2dbfae02b)
![image](https://github.com/codethestudent/Drone-Detection/assets/96714243/5c44287a-960a-4f52-9fcb-266d7648ad3a)
![image](https://github.com/codethestudent/Drone-Detection/assets/96714243/f308a5b3-1643-46b0-be98-108a24c0faa4)


구현시 어려웠던 점 : GPU 활용을 위한 젯슨 나노의 설정 과정(OpenCV 업데이트 및 CUDA라이브러리 설치), 미사일 테스트베드와 카메라 짐벌 시스템의 통합 테스트
