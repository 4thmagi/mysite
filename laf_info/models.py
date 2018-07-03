from django.db import models
from login.models import User
from imagekit.models import ProcessedImageField
from imagekit.processors import ResizeToFill

# Create your models here.

class Laf_info(models.Model):

    info_kind = (
        ('lost', "失物招领"),
        ('found', "寻物启事"),
    )

    things_kind = (
        ('1', "证件类型"),
        ('0', "非证件类型"),
    )

    title = models.CharField(max_length=50)  #信息标题
    content = models.TextField()  #信息内容
    author = models.ForeignKey(User, on_delete=models.CASCADE)  #信息作者
    kind = models.CharField(max_length=32, choices=info_kind, default="失物招领")  #信息种类
    c_time = models.DateTimeField(auto_now_add=True)  #创建时间
    tkind = models.CharField(max_length=32, choices=things_kind, default="非证件类型") #失物种类
    pic = models.ImageField(upload_to='pic', null=True, blank=True) #物品照片
    doc_info = models.TextField(default="无")  #识别出的证件信息
    #place = models.CharField(max_length=50)  #失物、寻物地点

    def __str__(self):
        return self.title

    class Meta:
        ordering = ["-c_time"]
        verbose_name = "失物招领及寻物启事信息"
        verbose_name_plural = "失物招领及寻物启事信息"
