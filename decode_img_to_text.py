from WriteSthToImg import WriteSthToImg
import cv2
import traceback

if __name__ == '__main__':
    for i in range(5, 6):
        w = WriteSthToImg(left_bits=i)
        p = f'd{i}.png'
        print(f'Decode file: {p}')
        try:
            buf = w.decode_img(cv2.imread(p))
            print('Decoded buf len: ', len(buf))
            print(buf.decode('utf-8'))
        except:
            traceback.print_exc()
        
        print('-' * 20)
