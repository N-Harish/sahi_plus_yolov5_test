import re
from unittest import result
from sahi.utils.yolov5 import download_yolov5s6_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_sliced_prediction
import cv2
import youtube_dl
from tqdm import tqdm
import numpy as np
import time
from imutils import resize


link = 'https://youtu.be/WvhYuDvH17I'

# get youtube video for inference without downloading
def videoUrl(vidurl):
  with youtube_dl.YoutubeDL(dict(forceurl=True)) as ydl:
    r = ydl.extract_info(vidurl, download=False)
  urlList = list(filter(lambda item: item['ext'] == 'mp4' and item['format_note']=='720p' and item['vcodec'][0:4]!='av01', r['formats']))
  url = urlList[0]['url']
  return url


url = videoUrl(link)

# # download model run below commented code once
# download_yolov5s6_model("./yolov5s6.pt")


# initialize detection model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov5',
    model_path='./yolov5s6.pt',
    confidence_threshold=0.4,
    device='cpu'
)


prev_frame_t = 0
new_frame_t = 0

# url = './dereck.webm'

cap=cv2.VideoCapture(url)


# check if video could be opened for inference
while cap.isOpened():
    succ, frame = cap.read()
    
    if succ:
        # resize frame to height of 500px
        frame = resize(frame, height=500)
        
        # get sliced prediction using SAHI
        result = get_sliced_prediction(
            frame, 
            detection_model, 
            640, 
            640, 
            overlap_height_ratio=0.4,
            overlap_width_ratio=0.4
        )

        # convert results to coco format
        preds = result.to_coco_annotations()
        
        # get only bboxes of person from the converted results
        bboxes = [i['bbox'] for i in preds if i['category_name'] =='person']
        
        # draw bbox
        for i in bboxes:
            cv2.rectangle(frame, (i[0], i[1]), (i[0]+i[2], i[1]+i[3]), (144, 238, 144), thickness=2)
    
        # calculate fps
        new_frame_t = time.time()
        fps = 1 / (new_frame_t - prev_frame_t)
        prev_frame_t = new_frame_t
        fps = str(int(fps))
        fps = "FPS:- " + fps

        # get text size of put text
        (w, h), _ = cv2.getTextSize(fps, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)

        # draw rectangle and put fps inside that
        x1 = 6
        y1 = 75
        cv2.rectangle(frame, (x1, y1 - 60), (x1 + w, y1 + h - 40), (0, 0, 0), -1)

        # add red border in the black rectangle for fps
        cv2.rectangle(frame, (x1, y1 - 60), (x1 + w, y1 + h - 40), (255, 0, 0), 2)

        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 255), 3, cv2.LINE_AA)

        # show the image
        cv2.imshow("pred", frame)

        # exit code by pressing 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    else:
        break

# release video
cap.release()
cv2.destroyAllWindows()
