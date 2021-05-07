import matplotlib.pyplot as plot
import os
import cv2

# visualize loss & accuracy
def visualize_curve(log_root):
    log_file = open(log_root, 'r')
    result_root = log_root[:log_root.rfind('/') + 1] + 'train.jpg'
    loss = []
    
    top1_i2t = []
    top10_i2t = []
    top1_t2i = []
    top10_t2i = []
    for line in log_file.readlines():
        line = line.strip().split()
        
        if 'top10_t2i' not in line[-2]:
            continue
        
        loss.append(line[1])
        top1_i2t.append(line[3])
        top10_i2t.append(line[5])
        top1_t2i.append(line[7])
        top10_t2i.append(line[9])

    log_file.close()

    plt.figure('loss')
    plt.plot(loss)

    plt.figure('accuracy')
    plt.subplot(211)
    plt.plot(top1_i2t, label = 'top1')
    plt.plot(top10_i2t, label = 'top10')
    plt.legend(['image to text'], loc = 'upper right')
    plt.subplot(212)
    plt.plot(top1_t2i, label = 'top1')
    plt.plot(top10_i2t, label = 'top10')
    plt.legend(['text to image'], loc = 'upper right')
    plt.savefig(result_root)
    plt.show()


if __name__ == '__main__':
    log_root = '/home/zhangqi/Deep-Cross-Modal-Projection-Learning-for-Image-Text-Matching/data/logs/train.log'
    visualize_curve(log_root)
