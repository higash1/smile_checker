import cv2
import os
from retinaface.pre_trained_models import get_model

import warnings

warnings.simplefilter('ignore',UserWarning)

pic_path = './real_smile'
# baby_list = sorted([i for i in os.listdir(pic_path) if '.JPG' in i])
baby_list = ['./baby_picture/sample.jpg']

smile_cascade = cv2.CascadeClassifier('./haarcascade_smile.xml')

face_count = 0
smile_count = 0

for i,img_path in enumerate(baby_list):
    print(img_path,'START')
    
    img = cv2.imread(os.path.join(pic_path,img_path))
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    model = get_model("resnet50_2020-07-20", max_size=2048)
    model.eval()
    annotation = model.predict_jsons(rgb_img)
    face_ok:bool = False
    smile_ok:bool = False
    if len(annotation) > 0:
        face_ok:bool = True
        face_count += 1
        annotate = annotation[0]
        # for annotate in annotation:
        if not annotate['bbox']: continue
        if not annotate['score']: continue
        if annotate['score'] < 0.80: continue
        cv2.rectangle(img,(int(annotate['bbox'][0]),int(annotate['bbox'][1])),(int(annotate['bbox'][2]),int(annotate['bbox'][3])),color=(0,255,0),thickness=3,lineType=cv2.LINE_4,shift=0)
        gray_roi = gray[int(annotate['bbox'][1]):int(annotate['bbox'][3]),int(annotate['bbox'][0]):int(annotate['bbox'][2])]
        smiles= smile_cascade.detectMultiScale(gray_roi,scaleFactor= 1.2, minNeighbors=10, minSize=(20, 20))#笑顔識別
        if len(smiles) >0 :
            smile_ok:bool = True
            smile_count += 1
            for(sx,sy,sw,sh) in smiles:
                # cv2.rectangle(img,(int(annotate['bbox'][0]) + int(sx),int(annotate['bbox'][1]) + int(sy)),(int(annotate['bbox'][2]) + int(sx) + int(sw/2),int(annotate['bbox'][3]) + int(sy) + int(sh/2)),(255, 0, 0),2)#red
                cv2.circle(img,(int(annotate['bbox'][0]+sx+sw/2),int(annotate['bbox'][1]+sy+sh/2)),int(sw/2),(255, 0, 0),2)#red
        if smile_ok:
            cv2.imwrite(os.path.join('./result',f'{img_path}'),img)
            print(img_path,'COMPLETE')

print(f"{smile_count}")    
print('accuracy', face_count/len(baby_list))
print("accuracy", smile_count/face_count)

    
    