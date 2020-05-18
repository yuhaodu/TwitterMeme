from PIL import Image
from pytesseract import image_to_string




img = Image.open()
text = image_to_string(img)
text = re.sub('\n', '', text)