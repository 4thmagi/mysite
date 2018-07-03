import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytesseract
from PIL import Image
debug = 1




def grayImg(img):
    # 转化为灰度图
    gray = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    #otsu二值化操作
    retval, gray = cv2.threshold(gray, 120, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return gray

def preprocess(gray):
    #二值化操作，但与前面grayimg二值化操作中不一样的是要膨胀选定区域所以是反向二值化
    ret, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    ele = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 10))
    #膨胀操作
    dilation = cv2.dilate(binary, ele, iterations=1)
    cv2.imwrite("binary.png", binary)
    cv2.imwrite("dilation.png", dilation)
    return dilation


def findTextRegion(img):
    region = []
    # 1. 查找轮廓。建立轮廓树关系，查找所有轮廓。压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)
        # 面积小的都筛选掉
        if (area < 300):
            continue
        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # 找到最小的矩形
        rect = cv2.minAreaRect(cnt)
        #函数 cv2.minAreaRect() 返回一个Box2D结构 rect：（最小外接矩形的中心（x，y），（宽度，高度），旋转角度）。
        if debug:
            print("rect is: ", rect)
        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if (height > width * 1.2):
            continue
        # 太扁的也不要
        if (height * 18 < width):
            continue
        if (width > img.shape[1] / 2 and height > img.shape[0] / 20):
            region.append(box)
    return region


def detect(img):
    # fastNlMeansDenoisingColored(InputArray src, OutputArray dst, float h=3, float hColor=3, int templateWindowSize=7, int searchWindowSize=21 )
    gray = cv2.fastNlMeansDenoisingColored(img, None, 10, 3, 3, 3)
    #cv2.fastNlMeansDenoisingColored作用为去噪
    coefficients = [0, 1, 1]
    m = np.array(coefficients).reshape((1, 3))
    gray = cv2.transform(gray, m)
    if debug:
        cv2.imwrite("gray.png", gray)
    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选数字区域
    region = findTextRegion(dilation)
    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        h = abs(box[0][1] - box[2][1])
        w = abs(box[0][0] - box[2][0])
        Xs = [i[0] for i in box]
        Ys = [i[1] for i in box]
        x1 = min(Xs)
        y1 = min(Ys)
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        #其中第一个参数image表示目标图像
        #第二个参数contours表示输入的轮廓组，每一组轮廓由点vector构成
        #第三个参数contourIdx指明画第几个轮廓，如果该参数为负值，则画全部轮廓
        #第四个参数color为轮廓的颜色
        #第五个参数thickness为轮廓的线宽，如果为负值表示填充轮廓内部
        if w > 0 and h > 0 and x1 < gray.shape[1] / 2:
            idImg = grayImg(img[y1:y1 + h, x1:x1 + w])
            #x1，y1作为一个顶点;x1+w，y1+h作为另一顶点
            cv2.imwrite("idImg.png", idImg)
            cv2.imwrite("contours.png", img)
            return idImg



def ocrIdCard(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_COLOR)
    img1 = cv2.resize(img, (428, 270), interpolation=cv2.INTER_CUBIC)
    idImg = detect(img1)
    image = Image.fromarray(idImg)
    tessdata_dir_config = '-c tessedit_char_whitelist=0123456789X --tessdata-dir "./"'
    print("checking")
    # result = pytesseract.image_to_string(image, lang='ocrb', config=tessdata_dir_config)
    tessdata_dir_config = '--tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata"'
    result = pytesseract.image_to_string(image, lang='chi_sim',config=tessdata_dir_config)
    print('the detect result is '+result)
    # print(pytesseract.image_to_string(image, lang='eng', config=tessdata_dir_config))
    if debug:
        '''
        f, axarr = plt.subplots(2, 3)
        axarr[0, 0].imshow(cv2.imread(imgPath))
        axarr[0, 1].imshow(cv2.imread("gray.png"))
        axarr[0, 2].imshow(cv2.imread("binary.png"))
        axarr[1, 0].imshow(cv2.imread("dilation.png"))
        axarr[1, 1].imshow(cv2.imread("contours.png"))
        axarr[1, 2].imshow(cv2.imread("idImg.png"))
        plt.show()
        '''
        plt.figure(num='pic', figsize=(6, 1))
        plt.subplot(1, 6, 1)
        plt.imshow(cv2.imread(imgPath))
        plt.axis('off')
        plt.subplot(1, 6, 2)
        plt.imshow(cv2.imread("gray.png"))
        plt.axis('off')
        plt.subplot(1, 6, 3)
        plt.imshow(cv2.imread("binary.png"))
        plt.title('the detect result is ' + result)
        plt.axis('off')
        plt.subplot(1, 6, 4)
        plt.imshow(cv2.imread("dilation.png"))
        plt.axis('off')
        plt.subplot(1, 6, 5)
        plt.imshow(cv2.imread("contours.png"))
        plt.axis('off')
        plt.subplot(1, 6, 6)
        plt.imshow(cv2.imread("idImg.png"))
        plt.axis('off')
        plt.show()


ocrIdCard("./2.jpg")


