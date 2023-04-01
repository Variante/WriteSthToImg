from WriteSthToImg import WriteSthToImg
import cv2

if __name__ == '__main__':
    with open('image_to_enc.jpg', 'rb') as f:
        text = f.read()
    for i in range(5, 6):
        w = WriteSthToImg(left_bits=i)
        img = w.to_img(text)
        print('Encoded shape: ', img.shape)
        cv2.imwrite(f'left-img-{i}-text.png', img)
        # buf = w.decode_img(cv2.imread(f'left-{i}-text.png'))
        # print(buf.decode('utf-8'))
