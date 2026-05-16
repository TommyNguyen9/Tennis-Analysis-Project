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
        keypoints = self.correct_keypoints_12_and_13(keypoints)

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
    
    def correct_keypoints_12_and_13(self, keypoints):
        p8 = self.get_point(keypoints, 8)
        p9 = self.get_point(keypoints, 9)
        p10 = self.get_point(keypoints, 10)
        p11 = self.get_point(keypoints, 11)

        new_p12 = (p8 + p9) / 2
        new_p13 = (p10 + p11) / 2

        new_p12[0] -= 8
        new_p13[0] -= 25

        self.set_point(keypoints, 12, new_p12)
        self.set_point(keypoints, 13, new_p13)

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







        