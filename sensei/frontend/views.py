from django.shortcuts import render

# Create your views here.

from rest_framework import viewsets
from .serializers import FrontEndSerializer
from .models import FrontEnd

# Create your views here.


class FrontEndView(viewsets.ModelViewSet):
    serializer_class = FrontEndSerializer
    queryset = FrontEnd.objects.all()