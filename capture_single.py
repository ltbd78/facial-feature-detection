import cv2
from fastai.vision import *
from fastai.vision.image import pil2tensor, Image

path = "C:\\Users\\Linsu Han\\Documents\\[GITHUB]\\facial-feature-detection"
fc = cv2.CascadeClassifier(path + '\\resources\\haarcascade_frontalface_default.xml')
learn = load_learner(path + '\\resources\\models\\', 'smiling_resnet50.pkl')


def extractFaceCoords(img, fc, tolerance):
    # H, W, D = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_coords = fc.detectMultiScale(gray, 1.2, tolerance, minSize=(60, 60))

    if len(face_coords) == 1:
        x, y, w, h = face_coords[0]
    elif len(face_coords) == 0:
        return None
    else:
        max_area = 0
        index = 0
        for i in range(len(face_coords)):
            _, _, wi, hi = face_coords[i]
            area = wi * hi
            if area > max_area:
                max_area = area
                index = i
        x, y, w, h = face_coords[index]

    return x, y, w, h


cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    # H, W, D = frame.shape
    face_coords = extractFaceCoords(frame, fc, 3)

    if face_coords is not None:
        x, y, w, h = face_coords
        face_rgb = cv2.cvtColor(frame[y:y + h, x:x + h], cv2.COLOR_BGR2RGB)
        img_fastai = Image(pil2tensor(face_rgb, np.float32).div_(255))
        prediction = str(learn.predict(img_fastai))
        cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=1)
        cv2.putText(img=frame, text=prediction, org=(x, y - 13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5,
                    color=(255, 255, 255))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
