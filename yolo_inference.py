<<<<<<< HEAD
from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.track(
                      'input_videos/input_video.mp4', conf = 0.3,
                       save = True,
                       project ='C:/Users/Smoke Nandos/Desktop/Computer Science Uni/Machine Learning/Projects/Tennis Analysis System/runs/detect', 
                       name='predict')
                      
# print(result)
# print("boxes:")

# for box in result[0].boxes:
#     print(box)

=======
from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.track(
                      'input_videos/input_video.mp4', conf = 0.3,
                       save = True,
                       project ='C:/Users/Smoke Nandos/Desktop/Computer Science Uni/Machine Learning/Projects/Tennis Analysis System/runs/detect', 
                       name='predict')
                      
# print(result)
# print("boxes:")

# for box in result[0].boxes:
#     print(box)

>>>>>>> bfdb4cb364210466e9fd888ce9ea3534b1b61d09
