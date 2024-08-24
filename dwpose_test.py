from dwpose.dwpose import DWposeDetector

if __name__ == "__main__":
    pose = DWposeDetector()
    import cv2
    test_image = r'D:\0821\extract-animation-poses\dwpose_origin\ControlNet-v1-1-nightly\test_imgs\pose1.png'
    oriImg = cv2.imread(test_image)  # B,G,R order
    import matplotlib.pyplot as plt
    poses, img_out = pose(oriImg)
    plt.imsave('result.jpg', img_out)
