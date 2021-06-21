


import os
import fnmatch
# In[25]:


import random

import cv2
from matplotlib import pyplot as plt
import albumentations as A




classes_txt = open("C:/Users/bench/PycharmProjects/AISS_CV/classes.txt", "r")
lines = classes_txt.read().splitlines()
labels_dict = {}
i = 0
for label in lines:
    labels_dict[str(i)] = label
    i += 1

# print(labels_dict)

## CONFIG #####################################:
## MY PATH EINGEBEN: PFAD WO DIE GANZE BILDER SIND
## classes_txt EINGEBEN WO  CLASSES FILE IST


mypath = "C:/Users/bench/PycharmProjects/AISS_CV/Training Dataset/"
# Wo die classes txt Datei liegt

classes_txt = open("C:/Users/bench/PycharmProjects/AISS_CV/classes.txt", "r")


mydict = {}


def read_txt_yolo():
    txt_list = []
    for file in os.listdir(mypath):
        if file.endswith(".txt"):
            txt_list.append(file)

    for txt_name in txt_list:
        txt_path = mypath + txt_name
        txt_file = open(txt_path, "r")
        lines = txt_file.read().splitlines()
        key = txt_name[:-4]
        mydict[key] = []
        for idx, line in enumerate(lines):
            value = line.split()
            x = y = w = h = cls = None
            cls = int(value[0])
            #             cls = labels_dict.get(cls)
            x = float(value[1])
            y = float(value[2])
            w = float(value[3])
            h = float(value[4])
            mydict[key].append([x, y, w, h, cls])


read_txt_yolo()


# ## Function to separate bb and Labls from the txt in the right form



def get_boxes_labels(bb_label):
    boundings = []
    labels = []

    for i in bb_label:
        labels.append(i[-1])
        boundings.append(i[:4])

    return boundings, labels






BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=5):
    """Visualizes a single bounding box on the image"""
    x, y, w, h = bbox
    dh, dw, _ = img.shape
    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1
    # print("class name = ", (class_name))
    org = (l,t)
    cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), thickness=thickness
                  )
    cv2.putText(img,class_name, org , fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5, color=color, thickness=5)
    return img


def visualize(image, bboxes, category_ids, category_id_to_name,img_name,transformation):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    cv2.imwrite("results/" +img_name +transformation +"_modified" +".jpg", img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # imS = cv2.resize(img,(960,540))
    # cv2.imshow("output",
    #            imS)
    # cv2.waitKey()

    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(img)






lines = classes_txt.read().splitlines()
category_id_to_name = {}

i = 0
for label in lines:
    category_id_to_name[(i)] = label
    i += 1

# print(category_id_to_name)


# transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.Rotate(p=0.5),
#     A.RandomBrightnessContrast(p=0.3),
#     A.Blur(blur_limit=7, always_apply=False, p=0.8),
#     A.CenterCrop(height=2500, width=3000, p=1),
#
#     A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3)
# ], bbox_params=A.BboxParams(format='yolo', label_fields=["category_ids"]))

transform = A.Compose([
    A.HorizontalFlip(p=0.8),
    A.Rotate(p=0.3),
    A.RandomBrightnessContrast(p=0.9),
    A.Blur(blur_limit=7, always_apply=False, p=0.8),
    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5),
    A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), always_apply=False, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, always_apply=False, p=0.5),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, interpolation=1, border_mode=4, value=None, mask_value=None, shift_limit_x=None, shift_limit_y=None, always_apply=False, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=["category_ids"]))

transform_to_gray = A.Compose([
    A.ToGray(p=0.9),
], bbox_params=A.BboxParams(format='yolo', label_fields=["category_ids"]))

transform_HFlip = A.Compose([
    A.HorizontalFlip(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=["category_ids"]))

transform_Blur = A.Compose([
    A.Blur(blur_limit=7, always_apply=False, p=0.9),
], bbox_params=A.BboxParams(format='yolo', label_fields=["category_ids"]))

transform_Brightness = A.Compose([
    A.RandomBrightnessContrast(p=0.9),
], bbox_params=A.BboxParams(format='yolo', label_fields=["category_ids"]))

transform_RGRShift = A.Compose([
    A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.9)
], bbox_params=A.BboxParams(format='yolo', label_fields=["category_ids"]))
txt_path = "results/"

def create_txt(txt_path,img_name,str1,transformation):

    txt_file = open( txt_path  + img_name +  transformation +"_modified" + ".txt", 'w+')
    txt_file.write(str1)
    txt_file.close()

i = 0
total_number_images = len(fnmatch.filter(os.listdir(mypath), '*.jpg'))
for file in os.listdir(mypath):
    if file.endswith(".jpg"):
        img_name = file.rstrip(".jpg")

        print("#### Processing " + img_name + " ###### " + str(i) + " out of " + str(total_number_images))
        j = 0
        bb_label = mydict.get(str(img_name))
        bboxes, category_ids = get_boxes_labels(bb_label)
        random.seed(7)
        image = cv2.imread(mypath + file)
        # print(bboxes)
        # print(category_ids)

        transformation = "_combined_transform_"
        print("##" + transformation  + " on " + img_name + " ####### " + str(j) + " out of 6" )
        j += 1
        transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        print(transformed['bboxes'])
        visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name,img_name,transformation)
        str1 = ""
        for e, f in zip(transformed['category_ids'], transformed['bboxes']):
            str1 = str1+str(e)+" "+" ".join(str(h) for h in f)+"\n"
        # str1 = "".join(str(e,f) for e,f in (transformed['category_ids'],transformed['bboxes']))
        create_txt(txt_path,img_name,str1,transformation)
        # print("str2 == ", str1
        #       )

        transformation = "_to_gray_"
        print("## " + transformation  + " on " + img_name + " ####### " + str(j) + " out of 6")
        j += 1
        transformed = transform_to_gray(image=image, bboxes=bboxes, category_ids=category_ids)
        print(transformed['bboxes'])
        visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name,
                  img_name, transformation)
        str1 = ""
        for e, f in zip(transformed['category_ids'], transformed['bboxes']):
            str1 = str1 + str(e) + " " + " ".join(str(h) for h in f) + "\n"
        # str1 = "".join(str(e,f) for e,f in (transformed['category_ids'],transformed['bboxes']))
        create_txt(txt_path, img_name, str1, transformation)
        # print("str2 == ", str1
        #       )


        transformation = "_Blur"
        print("## " + transformation + " on " + img_name + " ####### " + str(j) + " out of 6")
        j += 1
        transformed = transform_Blur(image=image, bboxes=bboxes, category_ids=category_ids)
        print(transformed['bboxes'])
        visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name,
                  img_name,transformation)
        str1 = ""
        for e, f in zip(transformed['category_ids'], transformed['bboxes']):
            str1 = str1 + str(e) + " " + " ".join(str(h) for h in f) + "\n"
        # str1 = "".join(str(e,f) for e,f in (transformed['category_ids'],transformed['bboxes']))
        create_txt(txt_path, img_name, str1,transformation)
        # print("str2 == ", str1
        #       )

        transformation = "_Brightness"
        print("## " + transformation + " on " + img_name + " ####### " + str(j) + " out of 6")
        j += 1
        transformed = transform_Brightness(image=image, bboxes=bboxes, category_ids=category_ids)
        print(transformed['bboxes'])
        visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name,
                  img_name, transformation)
        str1 = ""
        for e, f in zip(transformed['category_ids'], transformed['bboxes']):
            str1 = str1 + str(e) + " " + " ".join(str(h) for h in f) + "\n"
        # str1 = "".join(str(e,f) for e,f in (transformed['category_ids'],transformed['bboxes']))
        create_txt(txt_path, img_name, str1,transformation)
        # print("str2 == ", str1
        #       )

        transformation = "_RGB"
        print("## " + transformation + " on " + img_name + " ####### " + str(j) + " out of 6")
        j += 1
        transformed = transform_RGRShift(image=image, bboxes=bboxes, category_ids=category_ids)
        print(transformed['bboxes'])
        visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name,
                  img_name, transformation)
        str1 = ""
        for e, f in zip(transformed['category_ids'], transformed['bboxes']):
            str1 = str1 + str(e) + " " + " ".join(str(h) for h in f) + "\n"
        # str1 = "".join(str(e,f) for e,f in (transformed['category_ids'],transformed['bboxes']))
        create_txt(txt_path, img_name, str1,transformation)
        print("str2 == ", str1
              )

        transformation = "_Horizontal_Flip"
        print("## " + transformation + " on " + img_name + " ####### " + str(j) + " out of 6")
        j += 1
        transformed = transform_HFlip(image=image, bboxes=bboxes, category_ids=category_ids)
        print(transformed['bboxes'])
        visualize(transformed['image'], transformed['bboxes'], transformed['category_ids'], category_id_to_name,
                  img_name, transformation)
        str1 = ""
        for e, f in zip(transformed['category_ids'], transformed['bboxes']):
            str1 = str1 + str(e) + " " + " ".join(str(h) for h in f) + "\n"
        # str1 = "".join(str(e,f) for e,f in (transformed['category_ids'],transformed['bboxes']))
        create_txt(txt_path, img_name, str1,transformation)
        # print("str2 == ", str1
        #       )

        # cv2.imwrite("results/" + "simply_modified_" + img_name + ".jpg", transformed['image'],
        #             [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    i += 1
        # if i == 4:
        #     break








