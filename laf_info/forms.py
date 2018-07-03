from django import forms
from captcha.fields import CaptchaField

class InfoForm(forms.Form):
    info_kind = (
        ('lost', "失物招领"),
        ('found', "寻物启事"),
    )
    things_kind = (
        ('1', "证件类型"),
        ('0', "非证件类型"),
    )
    title = forms.CharField(label='信息标题', max_length=50, widget=forms.TextInput(attrs={'class': 'form_control'}))
    content = forms.CharField(label='信息内容', max_length=10000, widget=forms.TextInput(attrs={'size': '40'}))
    kind = forms.ChoiceField(label='信息种类', choices=info_kind)
    tkind = forms.ChoiceField(label='失物种类', choices=things_kind)
    pic = forms.ImageField(label='信息图片（若无可不上传）', required=False)
    #captcha = CaptchaField(label='验证码')
