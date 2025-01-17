from django.shortcuts import render
from django.shortcuts import redirect
from . import models
from . import forms
import hashlib

# Create your views here.


def hash_code(s, salt='mysite'):
    h = hashlib.sha256()
    s += salt
    h.update(s.encode())  # update方法只接收bytes类型
    return h.hexdigest()

def index(request):
    pass
    return render(request, 'login/index.html')


def login(request):
    if request.session.get('is_login', None):
        return redirect("/index/")
    if request.method == "POST":
        login_form = forms.UserForm(request.POST)
        message = "请检查填写的内容！"
        if login_form.is_valid():
            username = login_form.cleaned_data['username']
            password = login_form.cleaned_data['password']
            try:
                user = models.User.objects.get(name=username)
                if user.password == hash_code(password):  # 哈希值和数据库内的值进行比对
                    request.session['is_login'] = True
                    request.session['user_id'] = user.id
                    request.session['user_name'] = user.name

                    return redirect('/index/')
                else:
                    message = "密码不正确！"
            except:
                message = "用户不存在！"
        return render(request, 'login/login.html', locals())

    login_form = forms.UserForm()
    return render(request, 'login/login.html', locals())


def register(request):
    if request.session.get('is_login', None):
        # 登录状态不允许注册。
        return redirect("/index/")
    if request.method == "POST":
        register_form = forms.RegisterForm(request.POST)
        message = "请检查填写的内容！"
        if register_form.is_valid():  # 获取数据
            username = register_form.cleaned_data['username']
            password1 = register_form.cleaned_data['password1']
            password2 = register_form.cleaned_data['password2']
            email = register_form.cleaned_data['email']
            phone = register_form.cleaned_data['phone']
            sex = register_form.cleaned_data['sex']
            if password1 != password2:  # 判断两次密码是否相同
                message = "两次输入的密码不同！"
                return render(request, 'login/register.html', locals())
            else:
                same_name_user = models.User.objects.filter(name=username)
                if same_name_user:  # 用户名唯一
                    message = '用户已经存在，请重新选择用户名！'
                    return render(request, 'login/register.html', locals())
                same_email_user = models.User.objects.filter(email=email)
                if same_email_user:  # 邮箱地址唯一
                    message = '该邮箱地址已被注册，请使用别的邮箱！'
                    return render(request, 'login/register.html', locals())

                # 当一切都OK的情况下，创建新用户

                new_user = models.User.objects.create()
                new_user.name = username
                new_user.password = hash_code(password1)  # 使用加密密码
                new_user.email = email
                new_user.phone = phone
                new_user.sex = sex
                new_user.save()
                return redirect("http://127.0.0.1:8000/login/")  # 自动跳转到登录页面
    register_form = forms.RegisterForm()
    return render(request, 'login/register.html', locals())


def logout(request):
    if not request.session.get('is_login', None):
        # 如果本来就未登录，也就没有登出一说
        return redirect("http://127.0.0.1:8000/index/")
    request.session.flush()
    # 或者使用下面的方法
    # del request.session['is_login']
    # del request.session['user_id']
    # del request.session['user_name']
    return redirect("http://127.0.0.1:8000/login/")


def changepwd(request):
    if request.method == "POST":
        change_form = forms.ChangeForm(request.POST)
        message = "请检查填写的内容！"
        if change_form.is_valid():  # 获取数据
            oldpassword = change_form.cleaned_data['oldpassword']
            newpassword1 = change_form.cleaned_data['newpassword1']
            newpassword2 = change_form.cleaned_data['newpassword2']

            user_name = request.session.get('user_name')
            user_id = request.session.get('user_id')
            oldpwd = models.User.objects.filter(user_name=user_name, pwd=oldpassword)
            if oldpwd: #判断旧密码是否输入正确
                change_user = models.User.objects.filter(user_name=user_name,pwd=oldpassword)
                if newpassword1 != newpassword2:  # 判断两次输入新密码密码是否相同
                    message = "两次输入的新密码不同！"
                    return render(request, 'login/changepwd.html', locals())
                else:
                    change_user.password = newpassword1
                    request.session.flush()
                    return redirect("http://127.0.0.1:8000/login")
            else:
                message = "旧密码输入错误！"
                return render(request, 'login/changepwd.html', locals())
    change_form = forms.ChangeForm()
    return render(request, 'login/changepwd.html', locals())