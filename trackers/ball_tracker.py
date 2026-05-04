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
        self.centers = []
        self.prev_bbox = None

        
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

        for i, frame in enumerate(frames):
            h, w, _ = frame.shape

     
        # Initialize position:

            if self.fake_ball_pos is None:
                self.fake_ball_pos = [w // 2, int(h * 0.75)]

            # Moving the ball:

            self.fake_ball_pos[0] += self.velocity[0]
            self.fake_ball_pos[1] += self.velocity[1]

            if i % 60 == 0:
                self.velocity[0] = random.choice([-12, -10, 10, 12])
                self.velocity[1] = random.choice([-8, -6, 6, 8])
          
            # Bounce on floor:

            floor_y = int(h * 0.85) # Virtual floor.

            if self.fake_ball_pos[1] >= h or floor_y or self.fake_ball_pos[1] <= 0:
                self.velocity[1] *= -1

            # Bounce off walls:

            # if self.fake_ball_pos[0] <= 0 or self.fake_ball_pos[0] >= w:
            #     self.velocity[0] *= -1

            x, y = self.fake_ball_pos
            self.centers.append((x, y))


            # Bounding box:

            bbox = (x - 5, y - 5, x + 5, y + 5)

            ball_detections.append({1: bbox})

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
        results = self.model.predict(frame, conf = 0.4)[0]
        
        ball_dict = {}

        if results.boxes is None or len(results.boxes) == 0:
            return ball_dict
        
        h, w = frame.shape[:2]
        
        filtered_boxes = []
        
        
        # Removing detections in unlikely areas for ball detection:

        for box in results.boxes:
            cls = int(box.cls.item())

            # Only sports ball is kept:
            if cls != 32:
                continue

            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            cx = (x1 + x2) / 2 # Center point of detection.
            cy = (y1 + y2) / 2


            if cy < h * 0.1: # Ignoring top
                continue
            if cx < w * 0.1 or cx > w * 0.9:
                continue

            filtered_boxes.append(box)


        if len(filtered_boxes) == 0:
            if self.prev_bbox is not None:
                ball_dict[1] = self.prev_bbox
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
        
       
        result = best_box.xyxy.tolist()[0]
        ball_dict[1] = result

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
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames




