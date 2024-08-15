from ultralytics import YOLO
import cv2
from flask import *


app = Flask(__name__, static_url_path='/static')


@app.route('/')
def home():
    return render_template('pai.html')


def get_video():
    model = YOLO('yolov5m.pt')
    url = 'vio.mp4' 
    cap = cv2.VideoCapture(url)
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        
        ditek = model.track(frame)[0]
        for box_yolo in ditek.boxes:
            x1,y1,x2,y2 = box_yolo.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,255),3)
            #print(x1,y1,x2,y2)
            
        ret2,butter = cv2.imencode('.jpg',frame)
        if not ret2:
            break
        frame = butter.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
      
@app.route('/video')
def video():
    return Response(get_video(), mimetype='multipart/x-mixed-replace; boundary=frame')     
    
    




if __name__ == '__main__':
    app.run(debug=True)
