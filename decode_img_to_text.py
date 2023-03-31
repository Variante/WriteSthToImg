from WriteSthToImg import WriteSthToImg
import cv2

if __name__ == '__main__':
    for i in range(5):
        w = WriteSthToImg(left_bits=i)
        p = f'left-{i}-text.png'
        print(f'Decode file: {p}')
        buf = w.decode_img(cv2.imread(p))
        print(buf.decode('utf-8'))
        print('-' * 20)
