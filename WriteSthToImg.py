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
        self.scale = 1
        self.bits_per_pixel = 2
        self.colorful = False
        self.padding_method = 'fill'
        if self.padding_method == 'fill':
            self.info_ratio = np.count_nonzero(255 - self.template) / len(self.template.reshape(-1))
            self.wh_ratio = self.template.shape[1] / self.template.shape[0]
            
    def _get_bit_mask(self):
        m = 0
        for _ in range(self.bits_per_pixel):
            m = (m << 1) + 1
        return m

    def to_img(self, buf):
        if not isinstance(buf, bytes):
            buf = buf.encode('utf-8')
        head = bytearray(len(buf).to_bytes(self.head_len, byteorder='big'))
        buf = np.array(head + bytearray(buf))
        # print(buf)
        pixel_per_byte = int(np.ceil(8 / self.bits_per_pixel))
        length_required = len(buf) * pixel_per_byte
        
        bit_mask = self._get_bit_mask()
        chns = []
        for i in range(pixel_per_byte):
            # print(bit_mask, i * self.bits_per_pixel)
            chns.append(np.right_shift(np.bitwise_and(buf, bit_mask << (i * self.bits_per_pixel)), i * self.bits_per_pixel))
              
        # make the final value
        chn = np.stack(chns, axis=1).reshape(-1)
        if self.left_bits > 0:
            magic_offset = 1 << (self.left_bits - 1)
        else:
            magic_offset = 0
        chn = np.left_shift(chn, self.left_bits) + magic_offset
        
        
        if self.padding_method == 'fill':
            area_of_template = int(np.ceil(length_required / self.info_ratio) + 1)
            template_height = int(np.ceil(np.sqrt(area_of_template / self.wh_ratio)))
            template_width = int(template_height * self.wh_ratio)
            
            target_img = cv2.resize(self.template, (template_width, template_height))
            
            mask_indices = np.argwhere(target_img != 255)
   
            l = len(chn)
            assert len(mask_indices) >= l
            
            if self.colorful:
                target_img = np.stack([target_img] * 3, axis=2)
                mask_indices = np.repeat(mask_indices, 3, axis=0)
                for i, index in enumerate(mask_indices):
                    target_img[index[0], index[1], i % 3] = chn[i % l]
            else:
                for i, index in enumerate(mask_indices):
                    target_img[index[0], index[1]] = chn[i % l]
                
            
        else:
            raise ValueError(f'Not recognized padding method: {self.padding_method}')
        
        if self.scale > 1:
            return np.repeat(np.repeat(target_img, self.scale, axis=0), self.scale, axis=1)
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
        # mask = np.bitwise_and(np.bitwise_and(img[..., 0], img[..., 1]), img[..., 2])
        # mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # mask_indices = np.argwhere(mask < 250)
        # img = img[self.scale // 2::self.scale, self.scale // 2::self.scale]
        if self.scale > 1:
            # take average for larger scale
            kernel = np.ones((self.scale, self.scale), dtype=float)
            kernel = kernel / np.sum(kernel)
            img = cv2.filter2D(img, -1, kernel, anchor=(0, 0))[::self.scale, ::self.scale]
        
        if not self.colorful:
            img = np.mean(img, -1)
            img = img.astype(np.uint8)
 
        size = (img.shape[1], img.shape[0])
        target_img = cv2.resize(self.template, size)
        mask_indices = np.argwhere(target_img != 255)
        
        chn = np.bitwise_and(np.right_shift(img[mask_indices[:, 0], mask_indices[:, 1]], self.left_bits), self._get_bit_mask())
        chn = chn.reshape(-1)
        
        pixel_per_bytes = int(np.ceil(8 / self.bits_per_pixel))
        def _merge_mul_bytes(b):
            b = b[:len(b) // pixel_per_bytes * pixel_per_bytes]
            b = b.reshape(-1, pixel_per_bytes)
            res = 0
            for i in range(pixel_per_bytes):
                res += np.left_shift(b[:, i], i * self.bits_per_pixel)
            return res
           
        data_len = int.from_bytes(_merge_mul_bytes(chn[:self.head_len * pixel_per_bytes]), byteorder='big')
        
        return _merge_mul_bytes(chn[self.head_len * pixel_per_bytes: (self.head_len + data_len) * pixel_per_bytes]).tobytes()


if __name__ == '__main__':
    w = WriteSthToImg()
    img = w.to_img(' '.join([str(i) for i in range(500)]))
    print('Encoded shape: ', img.shape)
    cv2.imwrite('dst.png', img)
    buf = w.decode_img(cv2.imread('dst.png'))
    print(buf.decode('utf-8'))
    