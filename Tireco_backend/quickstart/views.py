from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from rest_framework import permissions
from Tireco_backend.quickstart.serializers import UserSerializer, GroupSerializer
from django.http import HttpResponse, JsonResponse
from rest_framework.parsers import JSONParser

from django.views.decorators.csrf import csrf_exempt

from django.shortcuts import render

import sys
import os

import tireco.tireco as T

import numpy as np
import cv2 as cv
import base64

# Create your views here.
class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all().order_by('-date_joined')
    serializer_class = UserSerializer
    permission_classes = [permissions.IsAuthenticated]

class GroupViewSet(viewsets.ModelViewSet):
    """
    API endpoint that allows groups to be viewed or edited.
    """
    queryset = Group.objects.all()
    serializer_class = GroupSerializer
    permission_classes = [permissions.IsAuthenticated]

@csrf_exempt
def test(request):
    res = {"data": "Hello~!"}
    
    if request.method == "POST":
        IMG_PATH = "./tireco/data/sample.jpg"

        t = T.Tireco()

        titles = t.extract_title_with_img_path(IMG_PATH)
        res["titles"] = []

        for idx, title in enumerate(titles):
            res.get("titles").append({ "id": str(idx), "title": title})

    return JsonResponse(res)


@csrf_exempt
def timetable_recognition(request):
    res = {"data": "Hello~!"}
    
    if request.method == "POST":
        IMG_PATH = "./tireco/data/sample.jpg"

        req_json = JSONParser().parse(request)
        
        '''        
        res["titles"] = []

        for idx, title in enumerate(["과목1", "과목2", "과목3"]):
            res.get("titles").append({ "id": str(idx), "title": title})
        '''
        encoded_base64_img = req_json["file"]["_parts"][0][1]["data"]
        decoded_base64_img = base64.b64decode(encoded_base64_img)

        decoded_img = np.fromstring(decoded_base64_img, dtype=np.uint8)

        print("디코드된 이미지")
        print(decoded_img)

        t = T.Tireco()

        titles = t.extract_title_with_img_file(decoded_img)
        res["titles"] = []

        for idx, title in enumerate(titles):
            res.get("titles").append({ "id": str(idx), "title": title})
        

    return JsonResponse(res)