from ultralytics import YOLO
import cv2
import pickle
import random
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.prev_center = None

        self.fake_ball_pos = None
        self.velocity = [8, -5]

        
    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # Convert list -> pandas DF:
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values:

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions




    def detect_frames(self, frames, read_from_stub = False, stub_path = None):

    

    
        ball_detections = []

        for frame in frames:
            h, w, _ = frame.shape

        # Initialize position:

            if self.fake_ball_pos is None:
                self.fake_ball_pos = [w // 2, int(h * 0.75)]

            # Moving the ball:

            self.fake_ball_pos[0] += self.velocity[0]
            self.fake_ball_pos[1] += self.velocity[1]

            # Bounce on floor:

            if self.fake_ball_pos[1] >= int(h * 0.75):
                self.velocity[1] *= -1

            # Bounce on top:

            if self.fake_ball_pos[1] <= 0:
                self.velocity[1] *= -1

            # Bounce off walls:

            if self.fake_ball_pos[0] <= 0 or self.fake_ball_pos[0] >= w:
                self.velocity[0] *= -1

            x, y = self.fake_ball_pos

            # Bounding box:

            bbox = (x - 5, y - 5, x + 5, y + 5)

            ball_detections.append({1: bbox})
            h, w, _ = frame.shape

        # Initialize position:

            if self.fake_ball_pos is None:
                self.fake_ball_pos = [w // 2, int(h * 0.75)]

            # Moving the ball:

            self.fake_ball_pos[0] += self.velocity[0]
            self.fake_ball_pos[1] += self.velocity[1]

            # Bounce on floor:

            if self.fake_ball_pos[1] >= int(h * 0.75):
                self.velocity[1] *= -1

            # Bounce off walls:

            if self.fake_ball_pos[0] <= 0 or self.fake_ball_pos[0] >= w:
                self.velocity[0] *= -1

            x, y = self.fake_ball_pos

            # Bounding box:

            bbox = (x - 5, y - 5, x + 5, y + 5)

            ball_detections.append({1: bbox})

        return ball_detections


        # if read_from_stub and stub_path is not None:
        #     with open(stub_path, 'rb') as f:
        #         ball_detections = pickle.load(f)
        #     return ball_detections

        # for frame in frames:
        #     ball_dict = self.detect_frame(frame)
        #     ball_detections.append(ball_dict)
        
        # if stub_path is not None:
        #     with open(stub_path, 'wb') as f:
        #         pickle.dump(ball_detections, f)

        # return ball_detections


        # if read_from_stub and stub_path is not None:
        #     with open(stub_path, 'rb') as f:
        #         ball_detections = pickle.load(f)
        #     return ball_detections

        # for frame in frames:
        #     ball_dict = self.detect_frame(frame)
        #     ball_detections.append(ball_dict)
        
        # if stub_path is not None:
        #     with open(stub_path, 'wb') as f:
        #         pickle.dump(ball_detections, f)

        # return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf = 0.4)[0]
        results = self.model.predict(frame, conf = 0.4)[0]

        ball_dict = {}

        if results.boxes is None or len(results.boxes) == 0:
            return ball_dict
        
        h, w = frame.shape[:2]
        
        filtered_boxes = []
        
        # Removing detections in unlikely areas for ball detection:
        h, w = frame.shape[:2]
        
        filtered_boxes = []
        
        # Removing detections in unlikely areas for ball detection:

        for box in results.boxes:
            cls = int(box.cls.item())

            print("Class ID:", cls)

            # Only sports ball is kept:
            # if cls != 32:
            #     continue

            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            cx = (x1 + x2) / 2 # Center point of detection.
            cy = (y1 + y2) / 2

            print("Class ID:", cls)

            # Only sports ball is kept:
            # if cls != 32:
            #     continue

            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            cx = (x1 + x2) / 2 # Center point of detection.
            cy = (y1 + y2) / 2

            if cy < h * 0.1: # Ignoring top
                continue
            if cx < w * 0.1 or cx > w * 0.9:
                continue

            filtered_boxes.append(box)
       
            if cy < h * 0.1: # Ignoring top
                continue
            if cx < w * 0.1 or cx > w * 0.9:
                continue

            filtered_boxes.append(box)
       
        if len(filtered_boxes) == 0:
            return ball_dict
        
        def get_center(box):
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            return((x1 + x2) / 2, (y1 + y2) / 2)
        
        if self.prev_center is not None:
            best_box = min(
                filtered_boxes,
                key = lambda box: (
                    (get_center(box)[0] - self.prev_center[0]) ** 2 + 
                    (get_center(box)[1] - self.prev_center[1])**2
                    )
                )
        
        else:
            best_box = max(filtered_boxes, key = lambda box: float(box.conf))

        # Updating Tracking:
        cx, cy = get_center(best_box)
        self.prev_center = (cx, cy)

        print("Confidence: ", float(best_box.conf))
        
       
        def get_center(box):
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            return((x1 + x2) / 2, (y1 + y2) / 2)
        
        if self.prev_center is not None:
            best_box = min(
                filtered_boxes,
                key = lambda box: (
                    (get_center(box)[0] - self.prev_center[0]) ** 2 + 
                    (get_center(box)[1] - self.prev_center[1])**2
                    )
                )
        
        else:
            best_box = max(filtered_boxes, key = lambda box: float(box.conf))

        # Updating Tracking:
        cx, cy = get_center(best_box)
        self.prev_center = (cx, cy)

        print("Confidence: ", float(best_box.conf))
        
       
        result = best_box.xyxy.tolist()[0]
        ball_dict[1] = result

        return ball_dict

        
    
    def draw_bbboxes(self, video_frames, ball_detections): # Bounding boxes
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):

            # Drawing bounding boxes:
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # print("Ball Center:", cx, cy)
                # cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                # cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames




