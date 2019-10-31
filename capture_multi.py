import cv2
from fastai.vision import *
from fastai.vision.image import pil2tensor, Image
from glob import glob

path = "C:\\Users\\Linsu Han\\Documents\\[GITHUB]\\facial-feature-detection"
fc = cv2.CascadeClassifier(path + '\\resources\\haarcascade_frontalface_default.xml')

os.chdir(path + '\\resources\\models\\')
models = glob('*.pkl')
print('Models')
for i, k in enumerate(models):
    print(i, k)
model_num = int(input("Choose model #"))
print('Loading Model: ', models[model_num])

learn = load_learner(path + '\\resources\\models\\', models[model_num])
model_name = ' '.join(models[model_num].split('_')[:-1]).title()


def extractFaceCoords(img, fc, tolerance):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_coords = fc.detectMultiScale(gray, 1.2, tolerance, minSize=(60, 60))
    return face_coords


cap = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
os.chdir(path + '\\demo')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while True:
    ret, frame = cap.read()
    # H, W, D = frame.shape
    face_coords = extractFaceCoords(frame, fc, 3)

    if face_coords is not None:
        for coords in face_coords:
            x, y, w, h = coords
            face_rgb = cv2.cvtColor(frame[y:y + h, x:x + h], cv2.COLOR_BGR2RGB)
            img_fastai = Image(pil2tensor(face_rgb, np.float32).div_(255))
            prediction = learn.predict(img_fastai)

            if int(prediction[0]) == 1:
                result = model_name + ': True'
            else:
                result = model_name + ': False'
            p = prediction[2].tolist()
            prob = 'Probability: ' + str(round(p[1], 3))
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 0), thickness=1)  # color in BGR
            cv2.putText(img=frame, text=prob, org=(x, y - 13), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5,
                        color=(0, 255, 0), thickness=1)
            cv2.putText(img=frame, text=result, org=(x, y - 26), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.5,
                        color=(0, 255, 0), thickness=1)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # quit
        break

    out.write(frame)
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()
