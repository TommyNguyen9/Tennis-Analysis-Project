from utils import (read_video, 
                   save_video)

from trackers import PlayerTracker, BallTracker

def main():
    # Reading video:

    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    # Detecting players & ball:

    player_tracker = PlayerTracker(model_path = 'yolov8x')
    ball_tracker = BallTracker(model_path = 'models/yolo5_last.pt')

    player_detections = player_tracker.detect_frames(video_frames, 
                                                     read_from_stub = True,
                                                     stub_path = "tracker_stubs/player_detections.pk1"
                                                     )
    
    ball_detections = ball_tracker.detect_frames(video_frames, 
                                                     read_from_stub = False,
                                                     stub_path = "tracker_stubs/ball_detections.pk1"
                                                     )

    # Draw outputs:

    # Player bounding boxes:

    output_video_frames = player_tracker.draw_bbboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bbboxes(output_video_frames, ball_detections)



    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()