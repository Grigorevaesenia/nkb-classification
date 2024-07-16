import nkb_classification
from nkb_classification.dataset import AnnotatedSingletaskDataset, AnnotatedYOLODataset,Transforms

import cv2
import pandas as pd
datasetYOLO = AnnotatedYOLODataset("/root/nkb-classification/coco8.yaml", fold='train', target_column=None, transform = Transforms(pipeline))

for i in range(len(datasetYOLO)):
    img, _ = datasetYOLO[i]
    cv2.imwrite(f'nkb_classification/test_val_crops/saved_img{i}.jpg', img)

for idx, class_label in datasetYOLO.idx_to_class.items():
    print(f"{idx}: '{class_label}'", end=", ")

print()

print([datasetYOLO.class_to_idx[label] for label in datasetYOLO.get_labels()])
    
# labelsYOLO = datasetYOLO.get_labels()
# print('labels AnnotatedYOLODataset:', labelsYOLO, type(labelsYOLO), '\n')


# # Initialize data to lists.
# data = [{'column_0': 'bowl',
#          'path': '/root/nkb-classification/nkb_classification/test/saved_image0.jpg', 
#          'fold': 'train'},
#         {'column_0': 'bowl',
#          'path': '/root/nkb-classification/nkb_classification/test/saved_image1.jpg', 
#          'fold': 'train'},
#         {'column_0': 'broccoli', 
#          'path': '/root/nkb-classification/nkb_classification/test/saved_image2.jpg', 
#          'fold': 'train'},
#         {'column_0': 'bowl', 
#          'path': '/root/nkb-classification/nkb_classification/test/saved_image3.jpg', 
#          'fold': 'train'}]
# # Creates DataFrame.
# df = pd.DataFrame(data)
# df.to_csv('test.csv')

# dataset = AnnotatedSingletaskDataset("/root/nkb-classification/test.csv", target_column='column_0')

# labels = dataset.get_labels()
# print('labels AnnotatedSingletaskDataset:', labels, type(labels), '\n')


# classesYOLO = datasetYOLO.get_labels()
# print('classes AnnotatedYOLODataset:', classesYOLO, type(classesYOLO), '\n')

# classes = dataset.get_labels()
# print('classes AnnotatedSingletaskDataset:', classes, type(classes), '\n')

# for i in range(4):
#     _, elemYOLO = datasetYOLO[i]
#     print('elem AnnotatedYOLODataset:', elemYOLO, type(elemYOLO), '\n')

#     _, elem = dataset[i]
#     print('elem AnnotatedSingletaskDataset:', elem, type(elem), '\n')

# y_true = [0, 20, 17, 0, 20, 0, 0, 0, 0, 0, 0, 17, 25, 0, 16, 58, 0] 

# y_pred = [58, 50, 58, 58, 50, 58, 49, 58, 58, 50, 58, 58, 49, 50, 23, 58, 50] 

# print('y_true classes', [datasetYOLO.class_to_idx[e] for e in y_true], "\n")

# print('y_pred classes', [datasetYOLO.class_to_idx[e] for e in y_pred], "\n")

# y_true2 = [50, 22, 49, 49, 49, 23, 22, 50, 75, 23, 45, 49, 75]                                                                                                                                  

# y_pred2 = [49, 23, 45, 49, 49, 22, 23, 45, 75, 45, 45, 45, 75] 

# print('y_true classes', [datasetYOLO.class_to_idx[e] for e in y_true2], "\n")

# print('y_pred classes', [datasetYOLO.class_to_idx[e] for e in y_pred2], "\n")

val_true_classes = [25, 0, 16, 17, 17, 0, 0, 0, 58, 0, 0, 0, 0, 0, 20, 20, 0]
train_true_classes = [45, 45, 50, 45, 49, 49, 49, 49, 23, 23, 58, 75, 22]

print(set(val_true_classes) | set(train_true_classes))
