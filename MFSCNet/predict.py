from yolo import YOLO
from utils.utils import tif_read
if __name__ == "__main__":
    yolo = YOLO()
    while True:
        img = input('Input image filename:')
        try:
            input_image = tif_read(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            image, box = yolo.detect_image(input_image)
            image.show()
