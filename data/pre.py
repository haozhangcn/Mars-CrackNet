import os
import shutil
import sys
import glob
import cv2
from albumentations import RandomCrop


if __name__ == '__main__':
    # root = 'earth_all/uneven/'
    # transfer json to images
    # jsons = glob.glob(root + '/*.json')
    # print(jsons)
    # print(len(jsons))
    # for j in jsons:
    #     os.system('labelme_json_to_dataset.exe ' + j)

    # copy files
    # dirs = os.listdir(root + '/folders/')
    # for dir in dirs:
    #     # image = cv2.imread("earth_all/uniform/folders/" + dir + "/img.png")
    #     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     # cv2.imwrite('earth_all/uniform/imgs/' + dir.split('_json')[0] + '.png', image)
    #
    #     shutil.copy(root + '/folders/' + dir + '/img.png', root + '/imgs/' + dir.split('_json')[0] + '.png',)
    #
    #     label = cv2.imread(root + '/folders/' + dir + '/label.png', cv2.IMREAD_GRAYSCALE)
    #     cv2.imwrite(root + '/masks/' + dir.split('_json')[0] + '.png', label // 37 * 255)


    # cut images from mars
    # for img in [1, 2, 3, 4, 5, 7]:
    #     folder = 'AS' + str(img)
    #     num = 50
    #     crop = RandomCrop(336, 336)
    #     image = cv2.imread("mars/" + folder + "_json/img.png")
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    #     label = cv2.imread('mars/' + folder + '_json/label.png', cv2.IMREAD_GRAYSCALE)
    #
    #     print(label)
    #     for i in range(1, num + 1):
    #         img = crop(image=image, mask=label)
    #         cv2.imwrite('mars/imgs/' + folder + '_' + str(i).zfill(3) + '.png', img['image'])
    #         cv2.imwrite('mars/masks/' + folder + '_' + str(i).zfill(3) + '.png', img['mask'] // 37 * 255)
    #
    # imgs = os.listdir('mars/imgs/')
    # print(imgs)
    # with open('lists/mars.txt', 'w') as w:
    #     for i in imgs:
    #         w.write(i.split('.')[0] + '\n')

    # for i in [1, 2, 3, 4, 5, 7]:
    #     print(str(i))
    #     shutil.copy('mars_321/' + 'AS' + str(i) + '_json/img.png', 'mars_321/test_imgs/AS' + str(i) + '.png')
    #     label = cv2.imread('mars_321/' + 'AS' + str(i) + '_json/label.png', cv2.IMREAD_GRAYSCALE)
    #     cv2.imwrite('mars_321/test_masks/AS' + str(i) + '.png', label // 37 * 255)

    # deal with new generated data
    # all = os.listdir('new/')
    # for a in all:
    #     # os.rename('new/' + a, 'new/new' + a)
    #     if 'json' in a:
    #         print(a)
    #         os.system('labelme_json_to_dataset.exe new/' + a)

    # for img in range(1, 7):
    #     folder = 'new' + str(img)
    #     num = 10

    #         img = crop(image=image, mask=label)
    #         cv2.imwrite('new/imgs/' + folder + '_' + str(i).zfill(3) + '.png', img['image'])
    #         cv2.imwrite('new/masks/' + folder + '_' + str(i).zfill(3) + '.png', img['mask'] // 37 * 255)

    # imgs = os.listdir('new/imgs/')
    # print(imgs)

    # with open('lists/mars_test.txt', 'r') as r:
    #     test = r.read()
    # test = test.split('\n')[:-1]
    # with open('lists/earth.txt', 'r') as r:
    #     train = r.read()
    # train = train.split('\n')[:-1]

    #     mean += i.mean()
    #     std += i.std()
    #
    # print(mean/len(imgs), std/len(imgs))

    # processing generated masks
    # gens = os.listdir('./gen/imgs/')
    # print(gens)
    # for img in gens:
    #     folder = img.split('.')[0]
    #     print("./gen/imgs/" + img)
    #     num = 50
    #     crop = RandomCrop(336, 336)
    #     image = cv2.imread("./gen/imgs/" + img)
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    #     label = cv2.imread('./gen/masks/' + img, cv2.IMREAD_GRAYSCALE)
    #
    #     # print(label)
    #     for i in range(1, num + 1):
    #         img = crop(image=image, mask=label)
    #         cv2.imwrite('mars/imgs/' + folder + '_' + str(i).zfill(3) + '.png', img['image'])
    #         cv2.imwrite('mars/masks/' + folder + '_' + str(i).zfill(3) + '.png', img['mask'] // 37 * 255)
    
    new = os.listdir('new/imgs/')
    with open('lists/earth.txt', 'w') as w:
        for n in new:
            w.write(n.split('.')[0] + '\n')
