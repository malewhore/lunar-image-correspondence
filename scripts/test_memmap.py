import numpy as np

def test():
    print("Testing memmap...")
    mov_path = "data/real_pairs_drive/pair_003/moving/data/calibrated/20210402/ch2_ohr_ncp_20210402T0546284043_d_img_d18.img"
    try:
        arr = np.memmap(mov_path, dtype=np.uint8, mode='r', shape=(78175, 12000))
        crop = arr[30000:31000, 5000:6000].copy()
        del arr
        print("Memmap successful, crop shape:", crop.shape)
        print("Mean value:", crop.mean())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test()
