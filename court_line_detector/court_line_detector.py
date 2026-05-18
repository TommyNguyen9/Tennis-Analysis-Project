import torch
import torchvision.transforms as transforms
import cv2
import torchvision.models as models
import numpy as np

class CourtLineDetector:

    def __init__(self, model_path):
        self.model = models.resnet50(pretrained = False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)
        self.model.load_state_dict(torch.load(model_path, map_location = 'cpu'))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])


    def predict(self, image):

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
     
        original_h, original_w = img_rgb.shape[:2]

        keypoints[::2] *= original_w/ 224.0
        keypoints[1::2] *= original_h/ 224.0

        keypoints = self.correct_keypoints_2_and_5(keypoints)
        keypoints = self.correct_keypoint_10(keypoints)
        keypoints = self.correct_keypoints_12(keypoints)
        keypoints = self.correct_keypoint_11(keypoints)
        keypoints = self.correct_keypoint_13(keypoints)
        keypoints = self.correct_keypoints_7_and_3(keypoints)
        keypoints = self.correct_keypoints_8_and_9(keypoints)
        
        return keypoints
    
    def get_point(self, keypoints, idx):
        return np.array([keypoints[idx *2], keypoints[idx * 2 + 1]], dtype = float)
    
    def set_point(self, keypoints, idx, point):
        keypoints[idx * 2] = point[0]
        keypoints[idx * 2 + 1] = point[1]

    def line_from_points(self, p1, p2):
        return np.cross(
            np.array([p1[0], p1[1], 1.0]),
            np.array([p2[0], p2[1], 1.0])
        )
    
    def intersection(self, line1, line2):
        point = np.cross(line1, line2)

        if abs(point[2]) < 1e-6:
            return None
        
        return np.array([point[0] / point[2], point[1] / point[2]],
                         dtype = float)
    

    def correct_keypoints_2_and_5(self, keypoints):
        p3 = self.get_point(keypoints, 3)
        p7 = self.get_point(keypoints, 7)

        # Tennis court proportions:

        alley = 1.37 
        singles_width = 8.23
        doubles_width = 10.97

        right_singles_fraction = (alley + singles_width) / doubles_width
        left_singles_fraction = alley / doubles_width

        # Solving equation for correct keypoint 2:

        p2_new = (p7 - right_singles_fraction * p3) / (1 - right_singles_fraction)
        p5_new = p2_new + left_singles_fraction * (p3 - p2_new)

        p2_new[0] += 80
        p5_new[0] += 60

        # Fine tuning to ensure points are fully accurate:

        p2_new[0] += 5
        p2_new[1] -= 5

        p5_new[0] += 5
        p5_new[1] -= 5

        self.set_point(keypoints, 2, p2_new)
        self.set_point(keypoints, 5, p5_new)
        
        return keypoints

    def correct_keypoint_10(self, keypoints):
        p4 = self.get_point(keypoints, 4)
        p5 = self.get_point(keypoints, 5)
        p13 = self.get_point(keypoints, 13)
        p11 = self.get_point(keypoints, 11)

        left_singles_line = self.line_from_points(p4, p5)
        bottom_service_line = self.line_from_points(p13, p11)

        new_p10 = self.intersection(left_singles_line, bottom_service_line)

        if new_p10 is not None:

            new_p10[0] -= 10 # Move it slightly left
            new_p10[1] += 13 #Move it slightly down.

            self.set_point(keypoints, 10, new_p10)

        return keypoints
    
    def correct_keypoint_11(self, keypoints):
        p10 = self.get_point(keypoints, 10)
        p13 = self.get_point(keypoints, 13)

        new_p11 = 2 * p13 - p10

        # manual tuning:
        new_p11[0] += 15 
        new_p11[1] -= 10

        self.set_point(keypoints, 11, new_p11)
        
        return keypoints
    
    def correct_keypoints_12(self, keypoints):
        # print("12/13 KEYPOINTS RUNNING")
        # print("OLD 12:", self.get_point(keypoints, 12))
       
        p4 = self.get_point(keypoints, 4)
        p5 = self.get_point(keypoints, 5)
        p6 = self.get_point(keypoints, 6)
        p7 = self.get_point(keypoints, 7)

        p8 = self.get_point(keypoints, 8)
        p9 = self.get_point(keypoints, 9)
        p10 = self.get_point(keypoints, 10)
        p11 = self.get_point(keypoints, 11)
        p13 = self.get_point(keypoints, 13)

        # print("P10:", p10)
        # print("P11:", p11)
        # print("OLD 13:", self.get_point(keypoints, 13))
        # print("MIDPOINT 13:", (p10 + p11) / 2)

    
        # Moving down the court:

        left_singles_line = self.line_from_points(p4, p5)
        right_singles_line = self.line_from_points(p6, p7)

        # Vanishing point for the court length direction:
        vanishing_point = self.intersection(left_singles_line, right_singles_line)

        if vanishing_point is None:
            return keypoints
        
        top_service_line = self.line_from_points(p8, p9)
        bottom_service_line = self.line_from_points(p10, p11)

        centre_service_line = self.line_from_points(p13, vanishing_point)

        new_p12 = self.intersection(centre_service_line, top_service_line)
        new_p13 = self.intersection(centre_service_line, bottom_service_line)

        # print("NEW 12:", new_p12)
        # print("NEW 13:", new_p13)

        if new_p12 is not None:
            new_p12[0] -= 14
            new_p12[1] += 10
            self.set_point(keypoints, 12, new_p12)

        if new_p13 is not None:
            new_p13[0] -= 21
            new_p13[1] += 12
            self.set_point(keypoints, 13, new_p13)

        # print("FINAL 12:", self.get_point(keypoints, 12))
        # print("FINAL 13:", self.get_point(keypoints, 13))

        return keypoints
    
    def correct_keypoint_13(self, keypoints):
        p10 = self.get_point(keypoints, 10)
        p11 = self.get_point(keypoints, 11)

        new_p13 = (p10 + p11) / 2

        new_p13[0] += 0
        new_p13[1] += 0

        self.set_point(keypoints, 13, new_p13)

        return keypoints
    
    def correct_keypoints_7_and_3(self, keypoints):
        p2 = self.get_point(keypoints, 2)
        p5 = self.get_point(keypoints, 5)
        
        alley = 1.37
        singles_width = 8.23
       
        # From doubles sideline to singles sideline (left side):
        alley_vector = p5 - p2

        # Scaling to match:

        singles_vector = alley_vector * (singles_width / alley)

        new_p7 = p5 + singles_vector
        new_p3 = new_p7 + alley_vector

        new_p7[0] -= 21 # left
        new_p7[1] -= 3 # up

        new_p3[0] -= 20
        new_p3[1] -= 3

        self.set_point(keypoints, 7, new_p7)
        self.set_point(keypoints, 3, new_p3)

        return keypoints
    
    def correct_keypoints_8_and_9(self, keypoints):
        p4 = self.get_point(keypoints, 4)
        p5 = self.get_point(keypoints, 5)
        p6 = self.get_point(keypoints, 6)
        p7 = self.get_point(keypoints, 7)

        p8 = self.get_point(keypoints, 8)
        p9 = self.get_point(keypoints, 9)
        p12 = self.get_point(keypoints, 12)

        left_singles_line = self.line_from_points(p4, p5)
        right_singles_line = self.line_from_points(p6, p7)

        # Using 8 & 9 to define top service line:
        top_service_line = self.line_from_points(p8, p9)

        new_p8 = self.intersection(left_singles_line, top_service_line)
        new_p9 = self.intersection(right_singles_line, top_service_line)

        if new_p8 is not None:
            new_p8[0] -= 25
            new_p8[1] -= -9
            self.set_point(keypoints, 8, new_p8)
        
        if new_p9 is not None:
            new_p9[0] -= 14
            new_p9[1] -= -8
            self.set_point(keypoints, 9, new_p9)

        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])

            cv2.putText(image, str(i//2), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2 )
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1) # -1 = filled.
        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames







        