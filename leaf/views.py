from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.urls import reverse
from leaf.models import PicUpload
from leaf.forms import ImageForm
from django.conf import settings
import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt
import math
import argparse
from django.http import HttpResponse, JsonResponse
import base64

# Create your views here.
def index(request):
    return render(request, 'index.html')

def detect1(img):
    CONF_THRESH, NMS_THRESH = 0, 0.5
    img1 = img.copy()
    # Load the network
    net = cv2.dnn.readNet('static/yolov4-custom.cfg', 'static/yolov4-custom_last.weights')
    #net = cv2.dnn_DetectionModel('yolov4-custom.cfg', 'yolov4-custom_last.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Get the output layer from YOLO
    layers = net.getLayerNames()
    output_layers = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    class_ids, confidences, b_boxes = [], [], []

    if(layer_outputs!= None):
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > CONF_THRESH:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype('int')

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    b_boxes.append([x, y, int(w), int(h)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

    # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
        indices = cv2.dnn.NMSBoxes(b_boxes, confidences, CONF_THRESH, NMS_THRESH)
        if(len(indices)!=0):
            indices = indices.flatten().tolist()


    # Draw the filtered bounding boxes with their class to the image
            with open('static/obj.names', "r") as f:
                classes = [line.strip() for line in f.readlines()]
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            for index in indices:
                x, y, w, h = b_boxes[index]
                cv2.rectangle(img1, (x, y), (x + w, y + h), (0,255,0), 1)
                cv2.putText(img1, classes[class_ids[index]], (x + 5, y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    return img1

def detect(request):
    image_path = ''
    image_path1 = ''
    ctx = {}
    src= 'pic_upload/'
    print('hello-1')
    if request.method=='POST':
        '''for image_file_name in os.listdir(src):
            if image_file_name.endswith(".jpg") :
                os.remove(src + image_file_name)
            if image_file_name.endswith(".png") :
                os.remove(src + image_file_name)'''
        form = ImageForm(request.POST, request.FILES)
        print('hello0')

        if form.is_valid():
            newdoc = PicUpload(imagefile=request.FILES['imagefile'])
            newdoc.save()
            print('hello1')
            return HttpResponseRedirect(reverse('detect'))
    elif request.method=='GET':
        form = ImageForm()
    else:
        print('Hello3')

    documents = PicUpload.objects.all()
    for document in documents:
        image_path = document.imagefile.name
        image_path1 = '/'+image_path
        document.delete()
    image_path2 = 'pic_upload/zindagikanamegana.jpg'
    image_path3 = '/pic_upload/zindagikanamegana.jpg'
    request.session['image_path'] = image_path
    #print(image_path)
    if image_path:
        myplant = request.session['image_path']
        img_path = myplant
        request.session.pop('image_path',None)
        request.session.modified = True
        img = cv2.imread(image_path)
        img = detect1(img)
        #plt.imshow(img)
        #cv2.imshow('detected',img)
        cv2.imwrite(image_path2,img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        return render(request, 'detect.html',{'documents': documents,'image_path1':image_path1,'form':form, 'image_path3':image_path3})
    else:
        for image_file_name in os.listdir(src):
            if image_file_name.endswith(".jpg") :
                os.remove(src + image_file_name)
            if image_file_name.endswith(".png") :
                os.remove(src + image_file_name)
    
        return render(request, 'detect.html',{'documents': documents,'form':form})


def result(request):
    return render(request, 'result.html')
