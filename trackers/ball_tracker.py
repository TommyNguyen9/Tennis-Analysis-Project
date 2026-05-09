from ultralytics import YOLO
import cv2
import pickle
import random
import pandas as pd
import numpy as np

print("Running file:", __file__)

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO("models/best.pt")
        self.prev_center = None

        self.centers = []
        self.prev_bbox = None

        self.missed_frames = 0
        self.hit_frames = set()
        self.hit_cooldown = 0

        self.kalman = cv2.KalmanFilter(4, 2)

        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)

        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)

        self.kalman.processNoiseCov = np.eye(4, dtype = np.float32) * 0.03



    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # Convert list -> pandas DF:
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        # Interpolate missing values:

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
    def get_ball_shot_frames(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]

        # Convert list -> pandas DF:

        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])

        df_ball_positions["ball_hit"] = 0

        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2'])/2
        df_ball_positions['delta_y'] = (df_ball_positions['mid_y'].diff())
        df_ball_positions["mid_y_rolling_mean"] = df_ball_positions['mid_y'].rolling(window = 5, min_periods = 1, center = False).mean()


        minimum_change_frames_for_hit = 3

        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2 ) + 1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1


        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions["ball_hit"] == 1].index.tolist()
        return frame_nums_with_ball_hits


    def detect_frames(self, frames, read_from_stub = False, stub_path = None):

        ball_detections = []
        self.centers = []

        self.fake_ball_pos = None

        for i, frame in enumerate(frames):
            print(f"\n--- FRAME {i} ---")

            if i in self.hit_frames:
                self.hit_cooldown = 5

            ball_dict = self.detect_frame(frame, i)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)


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

            MIN_HIT_SPEED = 20

            if (dx_curr * dx_prev < 0
                 and abs(dx_prev) > MIN_HIT_SPEED
                 and abs(dx_curr) > MIN_HIT_SPEED
            ):
                hits.append(1)
            else:
                hits.append(0)

        return hits

    def detect_frame(self, frame, frame_idx):
        prediction = self.kalman.predict()

        pred_x = prediction[0][0]
        pred_y = prediction[1][0]

        if frame_idx in self.hit_frames:
            self.prev_center = None
            self.centers = []

        results = self.model.predict(frame, conf = 0.2)[0]

        MAX_DIST = 140
        MAX_DIST_SQ = MAX_DIST ** 2
        
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

            # if cls != 32:
            #     continue

            x1, y1, x2, y2 = b.xyxy.tolist()[0]

            conf = float(b.conf)

            w_box = x2 - x1
            h_box = y2 - y1


            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2


            # Rejecting the edge detections
            if cx < w * 0.08 or cx > w * 0.92:
                continue

            if w_box > 30 or h_box > 30:
                continue

            ratio = w_box / h_box if h_box != 0 else 0

            if ratio < 0.5 or ratio > 2:
                continue

            if cy < h * 0.25:
                continue

            filtered_boxes.append(b)

    
        if self.prev_center is not None and len(filtered_boxes) == 0:
           
           print("USING PREDICTION")

       
           if len(self.centers) >= 3:
               
               dx1 = self.centers[-1][0] - self.centers[-2][0]
               dy1 = self.centers[-1][1] - self.centers[-2][1]

               dx2 = self.centers[-2][0] - self.centers[-3][0]
               dy2 = self.centers[-2][1] - self.centers[-3][1]

               dx = (dx1+ dx2) / 2
               dy = (dy1 + dy2) / 2

           elif len(self.centers) >= 2:
               dx = self.centers[-1][0] - self.centers[-2][0]
               dy = self.centers[-1][1] - self.centers[-2][1]
            
           else: 
               dx, dy = 0, 0

           max_speed = 40

           dx = max(-max_speed, min(max_speed, dx))
           dy = max(-max_speed, min(max_speed, dy))

           cx = self.prev_center[0] + dx
           cy = self.prev_center[1] + dy

           if cx < 0 or cx > frame.shape[1] or cy < 0 or cy > frame.shape[0]:
              return {1: self.prev_bbox} if self.prev_bbox is not None else {}

           size = 10
           bbox = (cx - size, cy - size, cx + size, cy + size)

           self.prev_center = (cx, cy)
           self.prev_bbox = bbox

           return {1: bbox}
        
        if len(filtered_boxes) == 0:
            boxes_to_use = []
        else:
            boxes_to_use = filtered_boxes

        if len(boxes_to_use) == 0:
            self.missed_frames += 1

            if self.missed_frames > 5:
                self.prev_center = None
                self.prev_bbox = None
                return {}
            
            if self.prev_bbox is None:
                return {}
            
            print("USING PREV BBOX")
            return {}
          

           
        if self.prev_center is not None:
            best_box = None
            best_dist = float("inf")

            if len(self.centers) >= 2:
                prev_dx = self.centers[-1][0] - self.centers[-2][0]
                prev_dy = self.centers[-1][1] - self.centers[-2][1]
            else:
                prev_dx, prev_dy = 0, 0


            for b in boxes_to_use:
                conf = float(b.conf)

                if conf < 0.15:
                    continue

                cx, cy = get_center(b)

                if 930 < cx < 980 and 280 < cy < 330:
                    continue

                if cy > h * 0.68:
                    continue
             
                dx = cx - self.prev_center[0]
                dy = cy - self.prev_center[1]

                movement_dist = (dx**2 + dy**2) ** 0.5

                if abs(dy) < 2 and movement_dist > 150:
                   continue

                if conf > 0.6 and movement_dist > 150:
                    best_box = b
                    break

                if movement_dist > 180 and conf < 0.5:
                    continue

                if movement_dist < 5:
                    continue

                pred_dx = cx - pred_x
                pred_dy = cy - pred_y

                prediction_dist = (pred_dx**2 + pred_dy**2) ** 0.5

                direction_score = (dx * prev_dx + dy * prev_dy) * 0.2

                direction_score = max(min(direction_score, 200), -200)

                if 205 <= frame_idx <= 215:
                    print(
                        f"CANDIDATE -> "
                        f"cx:{cx:.1f}, cy:{cy:.1f}, "
                        f"move:{movement_dist:.1f}, "
                        f"pred:{prediction_dist:.1f}, "
                        f"conf:{conf:.2f}, "
                        f"dir:{direction_score:.1f}"
                        )


                dist = (cx - pred_x)**2 + (cy - pred_y)**2

                if self.hit_cooldown > 0:
                    score = dist
                   
                else:
                    score = dist * 0.3 - direction_score * 0.3 - conf * 400

                    score += movement_dist * 0.2

                if score < best_dist:
                    best_dist = score
                    best_box = b

            if self.hit_cooldown > 0:
                self.hit_cooldown -= 1

            if best_box is None:
                self.missed_frames += 1

                if self.prev_bbox is None:
                    return {}

                return {}
          
        else:
            best_box = max(boxes_to_use, key = lambda b: float(b.conf))
                        
        
        # Extracting coordinates:
        x1, y1, x2, y2 = best_box.xyxy.tolist()[0]    

        # Updating Tracking:

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        self.kalman.correct(measurement)
        
        if self.prev_center is not None:
            dx = cx - self.prev_center[0]
            dy = cy - self.prev_center[1]

            movement = abs(dx) + abs(dy)

            if movement > 30:
                alpha = 0.0
            else:
                alpha = 0.3


            cx = alpha * self.prev_center[0] + (1 - alpha) * cx
            cy = alpha * self.prev_center[1] + (1 - alpha) * cy
            
            # Smoothing the bounding box:
            if self.prev_bbox is not None:

                prev_x1, prev_y1, prev_x2, prev_y2 = self.prev_bbox

                x1 = alpha * prev_x1 + (1 - alpha) * x1
                y1 = alpha * prev_y1 + (1 - alpha) * y1
                x2 = alpha * prev_x2 + (1 - alpha) * x2
                y2 = alpha * prev_y2 + (1 - alpha) * y2


        self.prev_center = (cx, cy)
        
        # print(f"TRACKED CENTER -> {self.prev_center}")

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
                if bbox is None or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = bbox

                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames




