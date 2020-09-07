# Face-Recognition-Using-CNN
CNN을 이용한 얼굴인식 출결시스템 구현 ( 상명대 상생플러스 )
<hr />
<b> Refer Link </b>

- 1. [AISangam - Facenet_Real-time-face-recognition-using-deep-learning-TensorFlow](https://github.com/merassom/Facenet-Real-time-face-recognition-using-deep-learning-Tensorflow)

- 2. [sumantrajoshi - Face-recognition-using-deep-learning](https://github.com/merassom/Face-recognition-using-deep-learning)



### Directory
 - class : compressed File pickle
 <pre> import pickle 
 
 with open('[pickleName]', 'rb') as f:
    data = pickle.load(f)</pre>

 - find_file : 각 파일 설명
 <pre> cropping_face.py : OpenCV를 이용한 얼굴 인식 및 Cropping
 identify_face_image.py : 정지영상 이미지 인식
 identify_face_vide.py : 3차원이미지(영상) 인식</pre>
 
 
 - model
 <pre> 20170511-185253.pb : 모델
 import_pb_to_tensorboard.py : 모델을 확인하기 위한 스크립트
 tensorboard.py : 모델을 확인하기 위한 스크립트</pre>
 
 - module
 <pre> haarcasade_frontalface_default.xml : 사람 얼굴의 특징을 이용한 이미지 속 얼굴 추출  함수 ( 알고리즘 )
 facenet.py : 실시간 이미지 인식을 위한 모듈 </pre>
 
 - pre_img
 <pre> 원본 이미지 -> cropping_face.py를 거친 전처리 된 이미지 ( 얼굴만 crop한 )를 담아두는 Directory </pre>
 
 - train_img
 <pre> pre_img로부터 train된 이미지 ( Face_Recognition_Integraion.py 실행 후 )</pre>
 
 
 
 <hr/>
 
 ![img1](https://github.com/merassom/Face-Recognition-Using-CNN/blob/master/Documents/%EA%B2%B0%EA%B3%BC%EB%B3%B4%EA%B3%A0%EC%84%9C_%EC%95%9E%EB%A9%B4%ED%91%9C%EC%A7%80.png)
 ![img2](https://github.com/merassom/Face-Recognition-Using-CNN/blob/master/Documents/project_img.PNG)
 
 ![img3](https://github.com/merassom/Face-Recognition-Using-CNN/blob/master/Documents/Movie_Realtime.PNG)
 
 ![img4](https://github.com/merassom/Face-Recognition-Using-CNN/blob/master/Documents/%ED%99%9C%EB%8F%99%EC%82%AC%EC%A7%843.jpg)

 
 <hr/>
 
 
 
 
 
 
  

