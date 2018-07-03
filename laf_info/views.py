from django.shortcuts import redirect, render
from .models import Laf_info
from django.shortcuts import render_to_response, get_object_or_404
from login.models import User
from . import forms
from IDCard.ID import *
import os
from django.db.models import Q

# Create your views here.


def laf_info_list(request):
    context = {}
    context['laf_infos'] = Laf_info.objects.all()
    context['l_infos'] = Laf_info.objects.filter(kind='lost')
    context['f_infos'] = Laf_info.objects.filter(kind='found')
    request.session['is_login'] = True
    return render(request, 'laf_info/laf_info_list.html', context)


def l_info_list(request):
    context = {}
    context['l_infos'] = Laf_info.objects.filter(kind='lost')
    request.session['is_login'] = True
    return render(request, 'laf_info/l_info_list.html', context)


def f_info_list(request):
    context = {}
    context['f_infos'] = Laf_info.objects.filter(kind='found')
    request.session['is_login'] = True
    return render(request, 'laf_info/f_info_list.html', context)


def laf_info_detail(request, laf_info_pk):
    context = {}
    context['l_infos'] = Laf_info.objects.filter(kind='lost')
    context['f_infos'] = Laf_info.objects.filter(kind='found')
    context['laf_info'] = get_object_or_404(Laf_info, pk=laf_info_pk)
    return render_to_response('laf_info/laf_info_detail.html', context)


def l_info_detail(request, laf_info_pk):
    context = {}
    context['laf_info'] = get_object_or_404(Laf_info, pk=laf_info_pk)
    context['l_infos'] = Laf_info.objects.filter(kind='lost')
    return render_to_response('laf_info/l_info_detail.html', context)


def f_info_detail(request, laf_info_pk):
    context = {}
    context['laf_info'] = get_object_or_404(Laf_info, pk=laf_info_pk)
    context['f_infos'] = Laf_info.objects.filter(kind='found')
    return render_to_response('laf_info/f_info_detail.html', context)


def laf_info_release(request):
    if request.method == "POST":
        info_form = forms.InfoForm(request.POST, request.FILES)
        if info_form.is_valid():
            title = info_form.cleaned_data['title']
            content = info_form.cleaned_data['content']
            kind = info_form.cleaned_data['kind']
            pic = info_form.cleaned_data['pic']
            tkind = info_form.cleaned_data['tkind']

            user_name = request.session.get('user_name')
            user_id = request.session.get('user_id')
            new_info = Laf_info(
                title=title,
                content=content,
                author=User.objects.get(id=user_id),
                kind=kind,
                pic=pic,
                tkind=tkind
            )
            #new_info = Laf_info.objects.create()
            #new_info.title = title
            #new_info.content = content
            #new_info.author = User.objects.get(id=user_id)
            #new_info.kind = kind
            new_info.save()
            path = new_info.pic.name
            print(tkind)
            print(tkind == '1')
            #需要将path获得的文件
            if tkind == '1':
                real_path = os.path.join('G:/PycharmProjects/mysite2/media', path)
                img = cv2.imread(real_path, cv2.IMREAD_COLOR)
                img1 = cv2.resize(img, (428, 270), interpolation=cv2.INTER_CUBIC)
                idImg = detect(img1)
                image = Image.fromarray(idImg)
                tessdata_dir_config = '--tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata"'
                result = pytesseract.image_to_string(image, lang='chi_sim', config=tessdata_dir_config)
                show_result = result[10:]
                new_info.doc_info = show_result
                new_info.save()
            else:
                new_info.doc_info = '无'
                new_info.save()



        return redirect("http://127.0.0.1:8000/laf_info/")


    info_form = forms.InfoForm()
    return render(request, 'laf_info/laf_info_release.html', locals())

