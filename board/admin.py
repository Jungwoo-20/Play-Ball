from django.contrib import admin
from .models import Admin
# Register your models here.

class UserAdmin(admin.ModelAdmin):
    list_display = ("admin_id","admin_pw")

admin.site.register(Admin,UserAdmin)