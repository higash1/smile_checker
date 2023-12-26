# smile_checker
spresenseを利用して取得した画像の笑顔を認識するためのdocker

※installやsetup足りていない場合があります。issueにてお知らせいただけると幸いです。

### setup
```sh
cd ~/smile_checker
mkdir shared_dir
./docker/build-docker.sh
```
### docker run
```sh
./docker/run-docker.sh
```
### Installing the face detection model
```sh
pip install -U retinaface_pytorch
pip install -r requirements.txt
```

※retinafaceが動かない、またはinstallできない場合はpipのupdateおよびopencv関連のモジュールの更新を行ってください

### download smile detection model (opencv cascade model)
[smile_detection_model](https://github.com/Aparajit-Garg/Face-and-smile-detection)

### test
```sh
python retinaface_check.py
```

### result sample
![sample](https://github.com/higash1/smile_checker/assets/106146319/9cfc682e-4d94-4dca-ae2f-278fc5a8401c)