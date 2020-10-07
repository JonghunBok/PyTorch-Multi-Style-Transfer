FROM  pytorch/pytorch

RUN apt-get update -y
RUN apt-get install -y libgl1-mesa-glx libglib2.0-0 libx11-xcb1

RUN pip install --upgrade pip
RUN pip install opencv-python 
RUN pip install flask
RUN pip install flask-cors
RUN pip install pillow
RUN pip install numpy
RUN pip install pyqt5
RUN pip install torchfile

RUN mkdir /app
WORKDIR /app
ENTRYPOINT  ["python", "/app/camera_demo_web.py"]
