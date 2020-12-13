# Computer Vision Final-term Project

1. Improve the accuracy of the hand pose estimation network you implemented in the Assignment 2 using **Data Augmentation, Data Generation** methods or **changing the network architecture** etc.
2. Develop a **new function** for the hand pose estimation network that can **classify `Rock-paper-scissors’** poses according to their estimated poses. It may require collecting some **new data and their annotation**s for the classification labels.

## Schedule

`12월 22일(화)`: 중간점검 (오프라인)

`12월 24일(목)`: 프로젝트 코드작업 마무리

`12월 27일(일)`: 프로젝트 레포트 제출 마감일

## Task Management

> 작업 시작할 때 꼭 pull 받고, 어느정도 작업한 뒤 push할지 생각하기
>
> 다른 사람이 작업하는 범위 너무 많이 변경하지 말기

### Solang

- **Data augmentation (with blender)**
- 손 이미지, 손 이미지에 따른 joints 값을 test와 train으로 나눠서

### Jiun

- **Changing the network architecture**
- 하이퍼파라미터, 네트워크 구조 등 어떻게 바꿔야될지 생각 및 구현

### Yuho

- **New function for Rock-paper-scissors & Collecting new data and annotations**
- joint 상관 없이 가위-바위-보만 구별하면 될 듯 

## Links

### For project

- [Project documentation](https://docs.google.com/document/d/1iPUQRkmHUErm2oIvBxyi5orJ25cLtbkMdiuc6MEMdkc/edit)
- [Project tutorial](https://www.notion.so/CSE48001_Final-Project-Data-Augmentation-Tutorial_kor-c07a3b43a122429db5b2d3553213a65c)

### Datasets

- [Rock Paper Scissors Dataset](https://www.kaggle.com/sanikamal/rock-paper-scissors-dataset)
- [Rock-Paper-Scissors Images](https://www.kaggle.com/drgfreeman/rockpaperscissors)
- [TensorFlow rock paper scissors dataset](https://www.tensorflow.org/datasets/catalog/rock_paper_scissors)