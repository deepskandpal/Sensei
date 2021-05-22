from django.contrib import admin

# Register your models here.
from .models import FrontEnd


class FrontEndAdmin(admin.ModelAdmin):
    list_display = ('title', 'description', 'completed')

# Register your models here.


admin.site.register(FrontEnd, FrontEndAdmin)