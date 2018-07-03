from django.urls import path
from . import views

# start with laf_info
urlpatterns = [
    # http://localhost:8000/laf_info/
    path('', views.laf_info_list, name='laf_info_list'),
    path('l_info', views.l_info_list, name='l_info_list'),
    path('f_info', views.f_info_list, name='f_info_list'),
    path('l_info/<int:laf_info_pk>', views.l_info_detail, name="l_info_detail"),
    path('f_info/<int:laf_info_pk>', views.f_info_detail, name="f_info_detail"),
    #path('<int:laf_info_pk>', views.laf_info_detail, name="laf_info_detail"),
    #path('laf_info_release', views.laf_info_release, name="laf_info_release"),
]
