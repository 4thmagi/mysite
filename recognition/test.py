from PIL import Image
import pytesseract


text = pytesseract.image_to_string(Image.open('1.jpg'),lang='chi_sim')
print(text)