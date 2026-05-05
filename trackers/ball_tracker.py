from ultralytics import YOLO
import cv2
import pickle
import random
import pandas as pd

print("Running file:", __file__)

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO("yolov8n.pt")
        self.prev_center = None

        self.centers = []
        self.prev_bbox = None

        print("Model:", self.model.model_name if hasattr(self.model, "model_name") else self.model)

        
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
        self.centers = []

        self.fake_ball_pos = None

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        

        return ball_detections
    
    def compute_movement(self):
        movements = []

        for i in range(1, len(self.centers)):
            x1, y1 = self.centers[i-1]
            x2, y2 = self.centers[i]

            dx = x2 - x1
            dy = y2 - y1

            movements.append((dx, dy))
        
        return movements

    def detect_hits(self, movements):
        hits = [0]

        for i in range(1, len(movements)):
            dx_prev, dy_prev = movements[i-1]
            dx_curr, dy_curr = movements[i]

            if dx_curr * dx_prev < 0:
                print("Hit detected at index:", i)
                hits.append(1)
            else:
                hits.append(0)

        return hits

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf = 0.25)[0]
        
        print("Boxes detected:", 0 if results.boxes is None else len(results.boxes))
        ball_dict = {}

        if results.boxes is None or len(results.boxes) == 0:
            return ball_dict
        
        
        # Removing detections in unlikely areas for ball detection:

        def get_center(box):
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            return ((x1 + x2) / 2, (y1 + y2) / 2)
        
        h, w = frame.shape[:2]
        
        filtered_boxes = []

        for b in results.boxes:
            cls = int(b.cls.item())

            if cls != 32:
                continue

            x1, y1, x2, y2 = b.xyxy.tolist()[0]
            w_box = x2 - x1
            h_box = y2 - y1

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            if w_box > 200 or h_box > 200:
                continue

            if cy < h * 0.1:
                continue

            filtered_boxes.append(b)

        if self.prev_center is not None and len(filtered_boxes) == 0:
           if len(self.centers) >= 2:
               dx = self.centers[-1][0] - self.centers[-2][0]
               dy = self.centers[-1][1] - self.centers[-2][1]

               max_speed = 20

               dx = max(-max_speed, min(max_speed, dx))
               dy = max(-max_speed, min(max_speed, dy))

               cx = self.prev_center[0] + dx
               cy = self.prev_center[1] + dy

               dx *= 0.8
               dy *= 0.8

               if cx < 0 or cx > frame.shape[1] or cy < 0 or cy > frame.shape[0]:
                   return {1: self.prev_bbox}

               size = 10
               bbox = (cx - size, cy - size, cx + size, cy + size)

               self.prev_center = (cx, cy)
               self.prev_bbox = bbox

               return {1: bbox}
           
           return {1: self.prev_bbox}
        
        if len(filtered_boxes) > 0:
            boxes_to_use = filtered_boxes
        else:
            boxes_to_use = results.boxes
            
        
        if self.prev_center is not None:
            best_box = min(
                boxes_to_use,
                key = lambda b: (
                    (get_center(b)[0] - self.prev_center[0]) ** 2 +
                    (get_center(b)[1] - self.prev_center[1]) ** 2
                )
            )
        else:
            best_box = max(boxes_to_use, key = lambda b: float(b.conf))

        # Extracting coordinates:
        x1, y1, x2, y2 = best_box.xyxy.tolist()[0]    

        # Updating Tracking:

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        if self.prev_center is not None:
            alpha = 0.7

            cx = alpha * self.prev_center[0] + (1 - alpha) * cx
            cy = alpha * self.prev_center[1] + (1 - alpha) * cy

        self.prev_center = (cx, cy)
        self.centers.append((cx, cy))

        self.prev_bbox = (x1, y1, x2, y2)

        ball_dict[1] = (x1, y1, x2, y2)

        return ball_dict
        
    
    def draw_bbboxes(self, video_frames, ball_detections, hits): # Bounding boxes
        output_video_frames = []
        for i, (frame, ball_dict) in enumerate(zip(video_frames, ball_detections)):
            if i < len(hits) and hits[i] == 1:
                cv2.putText(frame, "HIT", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 3)


            # Drawing bounding boxes:
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox               
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames




