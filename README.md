# Self-Driving-Applied-Deep-Learning

## The Complete Self - Driving car course : Applied Deep Learning
**Learn to use Deep Learning, Computer Vision and Machine Learning techniques to Build an Autonomous Car with Python**

Udemy Course Certification : [DL based Self Driving .pdf](https://github.com/user-attachments/files/16344291/default.pdf)


#### Table of Contents
* [1. 🛣 Lane Detection](#1-lane-detection)
    * [1.1 Basic OpenCV img Transform](#11-basic-opencv-img-transform)
    * [1.2 Real-Time Lane Detection and Road Visualization](#12-real-time-lane-detection-and-road-visualization)
* [2. 🚗 Behavioral Cloning](#2-behavioral-cloning)
    * [2.1 Collecting Data](#21-collecting-data)
    * [2.2 Balancing Data](#21-balancing-data)
* [3. 📸 Autonomous Simulation](#3-autonomous-simulation)

## 1. 🛣Lane Detection
**차선 감지**는 자율 주행 시스템에서 매우 중요한 구성 요소로, 차량이 차선 경계를 유지하며 도로를 안전하게 주행할 수 있도록 돕습니다.
OpenCV를 사용하여 실시간으로 차선을 감지하는 방법과 기술을 구현하였습니다.         

```
이미지 전처리 단계, 즉 그레이스케일 변환, 블러링, 엣지 감지와 같은 기본 단계와 함께 ROI, Hough Transformation을 적용하였습니다.
```
### 1.1 Basic OpenCV img Transform

<table>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/60e39dbb-9d31-4fad-a944-c0aab435dfe8" width="200" height="200" alt="Original Image"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/1a2fa6a6-0847-4779-bf87-2a0751948fea" width="200" height="200" alt="GrayScale Image"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/9be28f70-e80b-4948-8a81-bd12e9e39b39" width="200" height="200" alt="Gaussian Blur Image"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/0588fb10-c727-478e-a233-e8f6082f9d03" width="200" height="200" alt="Canny Image"></td>
  </tr>
  <tr>
    <td align="center"><strong>Original Image</strong></td>
    <td align="center"><strong>GrayScale Image</strong></td>
    <td align="center"><strong>Gaussian Blur Image</strong></td>
    <td align="center"><strong>Canny Image</strong></td>
  </tr>
  <tr>
    <td align="center">원본 이미지</td>
    <td align="center">GrayScale 적용</td>
    <td align="center">GrayScale + Blur</td>
    <td align="center">GrayScale + Blur + Canny</td>
  </tr>
</table>

### 1.2 Real-Time Lane Detection and Road Visualization

<table>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/2ff22b91-825e-4368-829e-953206a67d7c" width="200" height="200" alt="ROI Image"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/f495433c-a7f7-4f92-b9b6-3d39c2f96875" width="200" height="200" alt="Binary&Bitwise_and Image"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/01472c6f-4c57-4305-bccc-fd7b6052d6b2" width="200" height="200" alt="Hough Transform Image"></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/af2cb3b8-704d-45f9-9e67-14a332d58e49" width="200" height="200" alt="Optimization Image"></td>
  </tr>
  <tr>
    <td align="center"><strong>ROI Image</strong></td>
    <td align="center"><strong>Binary&Bitwise_and Image</strong></td>
    <td align="center"><strong>Hough Transform Image</strong></td>
    <td align="center"><strong>Optimization Image</strong></td>
  </tr>
  <tr>
    <td align="center">관심 영역 마스크 적용</td>
    <td align="center">이진화 및 비트 연산 강조</td>
    <td align="center">허프 변환을 통한 차선 감지</td>
    <td align="center">최적화된 최종 시각화</td>
  </tr>
</table>

| ![lane detection](https://github.com/user-attachments/assets/054b4761-34e9-4a39-aa4b-22844a812c99) |![fill road](https://github.com/user-attachments/assets/62265cae-4527-429b-9f01-dae37b9db441) |
|:---:|:---:|
|**Lane Detection**|**Road Filling**|
|The process of detecting lane boundaries |The lane detection process combined with the road filling |

## 2. 🚗Behavioral Cloning

**- Udacity SELF-DRIVING CAR ENGINEER PROGRAM**
[self-driving-car-sim](https://github.com/udacity/self-driving-car-sim)

*Usage : how to train cars how to navigate road courses using deep learning*

### 2.1 Collecting Data
<img src="https://github.com/user-attachments/assets/7e70e771-70f4-4481-857d-006794e2ca3a">
<img src="https://github.com/user-attachments/assets/3d0fe140-1ec0-42b3-adac-63456d3b578c">
<img src="https://github.com/user-attachments/assets/0269d30e-93f4-4064-aec8-3328dbe62e7f">

### 2.2 Balancing Data
- 문제점 : 데이터 불균형 문제
- 해결 방안 : 데이터셋에서 steering 값의 분포를 균형 있게 조정하기 위한 작업 수행

| 상태            | 총 데이터 수 | 제거된 데이터 수 | 남은 데이터 수 | 그래프                  |
|-----------------|------------|-----------------|----------------|-------------------------|
| **이전 그래프** | 1808       | 927             | 881            | <img src="https://github.com/user-attachments/assets/7850ca47-a8a3-450d-a13d-741d04a6407b" width = "300" height = "200"> |
| **수정된 그래프** | 881        | 0               | 881            |  <img src="https://github.com/user-attachments/assets/9da2f2be-fa53-4b08-b572-b3f8c70636b2" width = "300" height = "200"> |

- 모델에 학습할 수 있게 적절하게 변환
   - *데이터프레임에서 이미지 경로와 조향 각도*
     - 데이터 증강 : 세 개의 다른 카메라(중앙, 왼쪽, 오른쪽)에서 촬영된 이미지를 사용하여 데이터셋을 증강
     - 조향 각도 조정: 왼쪽 및 오른쪽 카메라 이미지의 조향 각도를 각각 ±0.15로 조정하여 차량이 중앙에서 벗어난 위치에서 복구하는 방법을 학습
     - 데이터 분할 : 훈련용 80%, 검증용 20% 분할

### 2.3 Data & Image Preprocessing

| **Features**                   | **Explaination** | **Image** |
|------------------------------|----------|---------------|
| **`Zoom`**                     | 이미지의 스케일을 조절하여 확대 또는 축소 | <img src="https://github.com/user-attachments/assets/d51ab763-2464-49da-abe6-91a5cfe7d472" width = "1200" height = "200">
| **`Pan`**                      | 이미지를 수평 및 수직으로 이동 | <img src="https://github.com/user-attachments/assets/e7f8f724-713e-439a-816a-27601d566572" width = "1200" height = "200">
| **`Brightness`**               | 이미지의 밝기를 랜덤하게 조정 | <img src="https://github.com/user-attachments/assets/cf549086-cbe2-4ca9-a120-ac4682196445" width = "1200" height = "200">
| **`Flip`**                     | 이미지를 좌우로 뒤집기 | <img src="https://github.com/user-attachments/assets/b46fe651-37a3-42a9-9fcd-8b63694f3fa2" width = "1200" height = "200">
| **`Augmentation`**     | 이미지와 조향 각도에 대해 확대/축소, 이동, 밝기 조절, 좌우 반전 | <img src="https://github.com/user-attachments/assets/996228f9-53aa-40c9-ad6a-eb565b88a5d2" width = "1200" height = "200" >
| **`Preprocess`**     | 영역 자르기/YUV 색상/가우시안 블러/픽셀 조정/정규화 | <img src="https://github.com/user-attachments/assets/400dbd43-fb89-4cca-a3b8-99829a79d8bf" width = "1200" height = "200" >
| **`Batch_generator`**    | 배치 단위로 이미지와 조향 각도를 생성 | <img src="https://github.com/user-attachments/assets/546bde63-e6e7-4d73-bf1d-177e326d5455" width = "1200" height = "200" >

### 2.4 Model Training

**Network Architecture**
- Using End-to-End Deep Learning for Self-Driving Cars Method

| **Network Architecture**   | **Model Summary** |
|----------------------------|-------------------|
|<img src="https://github.com/user-attachments/assets/52fdec75-5838-4431-9409-e056f9e7a861" width = "400" height = "400"> | <img src="https://github.com/user-attachments/assets/884e369b-a062-404e-873f-be987470690e" width = "500" height = "400"> |

| **Training Parameters**                       | **Results** |
|-----------------------------------------------|-------------|
| **Steps per Epoch:** 300 <br>   **Number of Epochs:** 10  <br>  **Validation Steps:** 200 <br>   **Verbose:** 1  <br>  **Shuffle:** True  | <img src="https://github.com/user-attachments/assets/4444dfbc-5b98-4aa6-b65a-1b2019681338"> |

### 2.5 Applying a trained deep learning model to a Flask application

## 3. 📸 Autonomous Simulation
