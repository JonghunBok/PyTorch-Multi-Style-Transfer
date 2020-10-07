from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
import base64
import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from io import BytesIO

from net import Net
from option import Options
import utils
from utils import StyleLoader

app = Flask(__name__)
CORS(app)
args = ''
idx = 0

@app.route('/health', methods=['GET'])
def health():
    return 'OK'

def app_error(e):
    return jsonify({'message': str(e)}), 400

@app.route('/', methods=['GET'])
def home():
    return '200 OK'


@app.route('/image', methods=['POST'])
def image():
    data_url = request.json['imageBase64']
    print("Image recieved")
    img_bytes = base64.b64decode(data_url.split(';')[1].split(',')[1])
    iobytes = BytesIO(img_bytes)
    print("iobytes", iobytes)
    print(Image.open)
    try:
        img = Image.open(iobytes)
    except Exception as ex:
        print('에러!', ex)

    #print("img")
    #img = np.array(img)

    #print(img.shape)
    #img = img[:240,:320,:]
    #print(img.shape)
    #img.save('test2.jpg')

    try:
        new_img = run_demo(img).copy()
    #    img = cv2.imdecode(img,cv2.IMREAD_COLOR)
    #    cv2.imshow('', img)
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()
        new_img = Image.fromarray(new_img, 'RGB')

        rawBytes = BytesIO()
        new_img.save(rawBytes, "JPEG")
        rawBytes.seek(0)
        img_base64 = base64.b64encode(rawBytes.read())
        return jsonify({'new_image':str(img_base64)})
    except Exception as ex:
       print('에러!', ex)
       return jsonify({'new_image': 'too bad'})



def run_demo(img, mirror=False):
	global idx
	idx += 1
	style_model = Net(ngf=args.ngf)
	model_dict = torch.load(args.model)
	model_dict_clone = model_dict.copy()
	for key, value in model_dict_clone.items():
		if key.endswith(('running_mean', 'running_var')):
			del model_dict[key]
	style_model.load_state_dict(model_dict, False)
	style_model.eval()
	if args.cuda:
		style_loader = StyleLoader(args.style_folder, args.style_size)
		style_model.cuda()
	else:
		style_loader = StyleLoader(args.style_folder, args.style_size, False)

	# Define the codec and create VideoWriter object
	#height =  args.demo_size
	#width = int(4.0/3*args.demo_size)
	#swidth = int(width/4)
	#sheight = int(height/4)
	height =  args.demo_size
	width = int(4.0/3*args.demo_size)
	swidth = int(width/4)
	sheight = int(height/4)

	# read frame
	if mirror: 
		img = cv2.flip(img, 1)
	cimg = img.copy()
	img = np.array(img).transpose(2, 0, 1)

	print("read frmae")
	# changing style 
	style_v = style_loader.get(int(idx/20))
	style_v = Variable(style_v.data)
	style_model.setTarget(style_v)

	img=torch.from_numpy(img).unsqueeze(0).float()
	if args.cuda:
		img=img.cuda()

	img = Variable(img)
	img = style_model(img)

	print("var and sm")
	if args.cuda:
		simg = style_v.cpu().data[0].numpy()
		img = img.cpu().clamp(0, 255).data[0].numpy()
	else:
		simg = style_v.data.numpy()
		img = img.clamp(0, 255).data[0].numpy()
	print("if else")
	simg = np.squeeze(simg)
	print("squuezed")
	img = img.transpose(1, 2, 0).astype('uint8')
	print("img transposed")
	simg = simg.transpose(1, 2, 0).astype('uint8')
	print("simg transposed")

	# display
	#simg = cv2.resize(simg,(swidth, sheight), interpolation = cv2.INTER_CUBIC)
	#cimg[0:sheight,0:swidth,:]=simg
	#img = np.concatenate((cimg,img),axis=1)
	#cv2.imshow('MSG Demo', img)
	#cv2.imwrite('stylized/%i.jpg'%idx,img)
	print("done")
	return img

def main():
	# getting things ready
	global args 
	global idx
	idx = 1
	args = Options().parse()
	if args.subcommand is None:
		raise ValueError("ERROR: specify the experiment type")
	if args.cuda and not torch.cuda.is_available():
		raise ValueError("ERROR: cuda is not available, try running on CPU")


	app.register_error_handler(Exception, app_error)
	app.run(host='0.0.0.0', port=4000)

if __name__ == '__main__':
	main()
