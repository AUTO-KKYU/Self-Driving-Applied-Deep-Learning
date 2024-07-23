# Self-Driving-Applied-Deep-Learning

## The Complete Self - Driving car course : Applied Deep Learning
**Learn to use Deep Learning, Computer Vision and Machine Learning techniques to Build an Autonomous Car with Python**

Udemy Course Certification : [DL based Self Driving .pdf](https://github.com/user-attachments/files/16344291/default.pdf)


#### Table of Contents
* [1. 🛣 Lane Detection](#1-lane-detection)
    * [1.1 Basic OpenCV img Transform](#11-basic-opencv-img-transform)
    * [1.2 Real-Time Lane Detection and Road Visualization](#12-real-time-lane-detection-and-road-visualization)
* [2. 🚥 German Traffic Signs](#2-german-traffic-signs)
* [3. 🚗 Behavioral Cloning](#3-behavioral-cloning)

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






## 2. 🚥German Traffic Signs
## 3. 🚗Behavioral Cloning
