# coding=utf-8

import numpy as np
import cv2

class WriteSthToImg:
    def __init__(self, template=None, left_bits=0):
        if not isinstance(template, str):
            template = 'template.png'
        self.template = cv2.imread(template, cv2.IMREAD_GRAYSCALE)
        self.left_bits = left_bits # from 0 to 4
        self.head_len = 16
        self.padding_method = 'fill'
        if self.padding_method == 'fill':
            self.info_ratio = np.count_nonzero(255 - self.template) / len(self.template.reshape(-1))
            self.wh_ratio = self.template.shape[1] / self.template.shape[0]
            

    def to_img(self, buf):
        if not isinstance(buf, bytes):
            buf = buf.encode('utf-8')
        head = bytearray(len(buf).to_bytes(self.head_len, byteorder='big'))
        buf = np.array(head + bytearray(buf))
        # print(buf)
        length_required = len(buf)
        
        chn1 = np.bitwise_and(buf, 0x07)
        chn2 = np.right_shift(np.bitwise_and(buf, 0x38), 3)
        chn3 = np.right_shift(np.bitwise_and(buf, 0xc0), 6)
        
        """
        validate = np.bitwise_and(buf, 0x1f)
        for i in range(1, 8):
            validate = np.bitwise_xor(np.bitwise_and(np.right_shift(buf, i), 0x01), validate)
        validate = np.left_shift(np.bitwise_and(validate, 0x01), 2)
        chn3 = np.bitwise_or(chn3, validate)
        """
        
        # to left bits
        chn1 = np.left_shift(chn1, self.left_bits)
        chn2 = np.left_shift(chn2, self.left_bits)
        chn3 = np.left_shift(chn3, self.left_bits)
        
        if self.padding_method == 'fill':
            area_of_template = int(np.ceil(length_required / self.info_ratio) + 1)
            template_height = int(np.ceil(np.sqrt(area_of_template / self.wh_ratio)))
            template_width = int(template_height * self.wh_ratio)
            
            target_img = cv2.resize(self.template, (template_width, template_height))
            
            mask_indices = np.argwhere(target_img != 255)
            
            magic_value = (0x07 << self.left_bits) ^ 0xff
            target_img = np.stack([target_img] * 3, axis=2)
            
            assert len(mask_indices) >= len(chn1)
            l = len(chn1)
            for i, index in enumerate(mask_indices):
                target_img[index[0], index[1], 0] = chn1[i % l] + (target_img[index[0], index[1], 0] & magic_value)
                target_img[index[0], index[1], 1] = chn2[i % l] + (target_img[index[0], index[1], 1] & magic_value)
                target_img[index[0], index[1], 2] = chn3[i % l] + (target_img[index[0], index[1], 2] & magic_value)
            
            
        else:
            raise ValueError(f'Not recognized padding method: {self.padding_method}')
        return target_img
        """
        resized = cv2.resize(img, self.img_size)
        q = 50
        while q > self.min_q_color:
            _, buf = cv2.imencode(self.ext, resized, [cv2.IMWRITE_WEBP_QUALITY, q])
            print(f"Got color q {q} and len {len(buf)}")
            if len(buf) * 2 < self.enc_len:

                break
            q -= 10
        else:
            # failed to compress
            print("Failed to compress image")
            return None
        print(len(buf), q)
        
        # print(buf.shape, buf.dtype)
        # reshape code
        enc_img = np.ones(self.enc_len, dtype=np.uint8) * 128

        enc_img[:buf.shape[0] * 2:2] = np.left_shift(np.bitwise_and(buf[:, 0], 0x0f), 3) + 4  # 低8位
        enc_img[1:buf.shape[0] * 2 + 1:2] = np.right_shift(np.bitwise_and(buf[:, 0], 0xf0), 1) + 4  # 高8位
        # print(buf[:5, 0], np.right_shift(enc_img[:10], 4))
        gray_buf = enc_img.reshape(self.enc_size)
        print(gray_buf)
        return cv2.resize(gray_buf, (self.enc_size[1] * self.block_scale, self.enc_size[0] * self.block_scale), interpolation=cv2.INTER_NEAREST)
        """
        
    def decode_img(self, img):
        mask = np.bitwise_and(np.bitwise_and(img[..., 0], img[..., 1]), img[..., 2])
        mask_indices = np.argwhere(mask != 255)
        
        chn1 = np.bitwise_and(np.right_shift(img[mask_indices[:, 0], mask_indices[:, 1], 0], self.left_bits), 0x07)
        chn2 = np.bitwise_and(np.right_shift(img[mask_indices[:, 0], mask_indices[:, 1], 1], self.left_bits), 0x07)
        chn3 = np.bitwise_and(np.right_shift(img[mask_indices[:, 0], mask_indices[:, 1], 2], self.left_bits), 0x07)
        
        buf = np.bitwise_or(np.bitwise_or(chn1, np.left_shift(chn2, 3)), np.left_shift(np.bitwise_and(chn3, 0x03), 6))
        data_len = int.from_bytes(buf[:self.head_len].tobytes(), byteorder='big') 
        return buf[self.head_len:self.head_len+data_len].tobytes()


if __name__ == '__main__':
    w = WriteSthToImg()
    img = w.to_img(' '.join([str(i) for i in range(500)]))
    cv2.imwrite('dst.png', img)
    buf = w.decode_img(cv2.imread('dst.png'))
    print(buf.decode('utf-8'))