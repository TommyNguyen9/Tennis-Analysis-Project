from ultralytics import YOLO
import cv2
import pickle

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

        print("Model Classes:", self.model.names)

    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf = 0.15)[0]

        ball_dict = {}

        if results.boxes is None or len(results.boxes) == 0:
            return ball_dict
        
        ball_boxes = []

        for box in results.boxes:
            cls = int(box.cls.item())
            print("Class:", cls, "Conf:", float(box.conf))

            if cls == 0:
                ball_boxes.append(box)

        if len(ball_boxes) == 0:
            return ball_dict
        
        # Picking the best box w/ the highest confidence score:
        best_box = max(results.boxes, key = lambda box: float(box.conf))

        result = best_box.xyxy.tolist()[0]
        ball_dict[1] = result

        return ball_dict

        
    
    def draw_bbboxes(self, video_frames, ball_detections): # Bounding boxes
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):

            # Drawing bounding boxes:
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames




