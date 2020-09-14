from django.conf.urls import url,include
from django.urls import path

from .import views
urlpatterns=[
    path('', views.main, name="Main"),
    path('index',views.index,name="index"),
    path('admin_main', views.admin_main, name="admin_main"),
    path('board_write',views.board_write,name="board_write"),
    path('board_insert',views.board_insert,name="board_insert"),
    path('board_detail/<int:no>/',views.board_detail,name="board_detail"),
    path('rating_insert',views.rating_insert,name="rating_insert"),
    path('log_in',views.log_in,name="log_in"),
    path('log_out',views.log_out,name="log_out"),
    path('reinforce_view',views.reinforce_view,name="reinforce_view"),
    path('delete_admin',views.delete_admin,name="delete_admin"),
    path('delete',views.delete,name="delete"),
    path('function_answer', views.function_answer, name="function_answer"),
]