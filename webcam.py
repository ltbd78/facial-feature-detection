import cv2
from fastai.vision import *
from fastai.vision.image import pil2tensor, Image


class WebCam(object):
    def __init__(self, Feature, learner_path, cascade_path, tolerance):
        self.Feature = Feature
        self.learn = load_learner(learner_path.rsplit('\\', 1)[0], learner_path.split('\\')[-1])
        self.cascade = cv2.CascadeClassifier(cascade_path)
        self.tolerance = tolerance

    def feed(self):
        def extractFaceCoords(img, cascade, tolerance):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_coords = cascade.detectMultiScale(gray, 1.2, tolerance, minSize=(60, 60))
            return face_coords

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            face_coords = extractFaceCoords(frame, self.cascade, self.tolerance)
            if face_coords is not None:
                for coords in face_coords:
                    x, y, w, h = coords
                    face_rgb = cv2.cvtColor(frame[y:y + h, x:x + h], cv2.COLOR_BGR2RGB)
                    img_fastai = Image(pil2tensor(face_rgb, np.float32).div_(255))
                    prediction = self.learn.predict(img_fastai)
                    if int(prediction[0]) == 1:
                        result = self.Feature + ': True'
                    else:
                        result = self.Feature + ': False'
                    p = prediction[2].tolist()
                    prob = 'probability: ' + str(round(p[1], 3))
                    cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 0),
                                  thickness=1)  # color in BGR
                    cv2.putText(img=frame, text=prob, org=(x, y - 13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5,
                                color=(0, 255, 0), thickness=1)
                    cv2.putText(img=frame, text=result, org=(x, y - 26), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=.5,
                                color=(0, 255, 0), thickness=1)

            ret, jpeg = cv2.imencode('.jpg', frame)
            output = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output + b'\r\n\r\n')
