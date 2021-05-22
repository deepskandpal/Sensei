from rest_framework import serializers
from .models import FrontEnd


class FrontEndSerializer(serializers.ModelSerializer):
    class Meta:
        model = FrontEnd
        fields = ('id', 'title', 'description', 'completed')