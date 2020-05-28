import h5py
from explore_australia.stamp import Stamp, get_coverages
import pandas
from explore_australia.stamp import get_coverages_parallel
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


def get_unique(all_set):
    unique_set = set()
    for i_set in range(0, len(all_set)):
        try :
            mat = all_set[i_set].pop()
            unique_set.add(mat)
        except KeyError:
            continue
    return unique_set


def accumulate_commodity_labels(df):
    "Accumulate commodity labels from a dataframe with a 'commodity' column"
    commodities = set()
    for comm in df.commodity:
        for comm in comm.split(';'):
            commodities.add(comm)
    return commodities


def accumulate_commodity_X(df):
    "Accumulate commodity labels from a dataframe with a 'commodity' column"
    x = set()
    for comm in df.x:
        x.add(comm)
    return x


def accumulate_commodity_Y(df):
    "Accumulate commodity labels from a dataframe with a 'commodity' column"
    y = set()
    for comm in df.y:
        y.add(comm)
    return y


resnet152 = models.resnet152(pretrained=True)
modules=list(resnet152.children())[:-1]
resnet152=nn.Sequential(*modules)
for p in resnet152.parameters():
    p.requires_grad = False

path_to_dataset = '/media/iman/A8AC6E12AC6DDAF8/iman_ubuntu_dataset_unearthed/train/'
all_folders = os.listdir(path_to_dataset)
commodits_all = []
X_all = []
Y_all = []

# {'Ti', 'Co', 'Zr', 'REE', 'PGE', 'Th', 'Mn', 'Sb', 'U', 'Ag', 'W', 'Mo', 'Bi', 'V', 'Sn', 'Zn', 'Pb', 'Au',
# 'Ni', 'Cu', 'Ta', 'Fe'}

# df["fruit"] = df["fruit"].map({"apple": 1, "orange": 2,...})


for k in range(3, 10):
    heatmap_all = []
    features_all = []
    for i_folder in all_folders[250*k:250*(k+1)]:
        print ('Reading folder:', i_folder)
        commodits_org = pandas.read_csv(path_to_dataset+str(i_folder)+'/commodities.csv')
        ds_all = pandas.DataFrame({'stamp_id':[], 'x':[], 'y':[], 'commodity':[]})
        for i_app in range(0, len(commodits_org['commodity'])):

            ds = pandas.DataFrame({'stamp_id':commodits_org['stamp_id'][i_app], 'x':commodits_org['x'][i_app],
                                   'y':commodits_org['y'][i_app], 'commodity':str(commodits_org['commodity'][i_app])},
                                  range(len(commodits_org['commodity'][i_app].split(';'))))

            for i_row in range(0, len(commodits_org['commodity'][i_app].split(';'))):
                ds['commodity'][i_row] = ds['commodity'][i_row].split(';')[i_row]

            ds_all = ds_all.append(ds)
        commodits = ds_all

        commodits['commodity'] = commodits['commodity'].map({'Ti': 0, 'Co': 1, 'Zr': 2, 'REE': 3, 'PGE': 4, 'Th': 5,
                                                             'Mn': 6, 'Sb': 7, 'U': 8, 'Ag': 9, 'W': 10, 'Mo': 11,
                                                             'Bi': 12, 'V': 13, 'Sn': 14, 'Zn': 15, 'Pb': 16, 'Au': 17,
                                                             'Ni': 18, 'Cu': 19, 'Ta': 20, 'Fe': 21})

        material_all_org = accumulate_commodity_labels(commodits_org)
        material_all = []
        material_all_update = []
        cord_x_all = []
        cord_y_all = []

        for i_com in range(0, len(commodits)):
            material_all.append(commodits['commodity'].tolist()[i_com])

            cord_x_all.append((commodits['x'].tolist()[i_com] + 12500)/100.)
            cord_y_all.append((commodits['y'].tolist()[i_com] + 12500)/100.)

        gravity_1 = cv2.imread(path_to_dataset+str(i_folder)+'/geophysics/gravity/bouger_gravity_anomaly.tif', -1)
        gravity_1_resized = cv2.resize(gravity_1, (224, 224))
        gravity_1_resized_img = np.repeat(gravity_1_resized.reshape(1, 224, 224), 3, axis=0).reshape(1, 3, 224, 224)

        gravity_2 = cv2.imread(path_to_dataset+str(i_folder)+
                               '/geophysics/gravity/isostatic_residual_gravity_anomaly.tif', -1)
        gravity_2_resized = cv2.resize(gravity_2, (224, 224))
        gravity_2_resized_img = np.repeat(gravity_2_resized.reshape(1, 224, 224), 3, axis=0).reshape(1, 3, 224, 224)

        magnetics_1 = cv2.imread(path_to_dataset+str(i_folder)+
                                 '/geophysics/magnetics/total_magnetic_intensity.tif', -1)
        magnetics_1_resized = cv2.resize(magnetics_1, (224, 224))
        magnetics_1_resized_img = np.repeat(magnetics_1_resized.reshape(1, 224, 224), 3, axis=0).reshape(1, 3, 224, 224)

        magnetics_2 = cv2.imread(path_to_dataset+str(i_folder)+
                                 '/geophysics/magnetics/variable_reduction_to_pole.tif', -1)
        magnetics_2_resized = cv2.resize(magnetics_2, (224, 224))
        magnetics_2_resized_img = np.repeat(magnetics_2_resized.reshape(1, 224, 224), 3, axis=0).reshape(1, 3, 224, 224)

        radiometrics_1 = cv2.imread(path_to_dataset+str(i_folder)+
                                    '/geophysics/radiometrics/filtered_potassium_pct.tif', -1)
        radiometrics_1_resized = cv2.resize(radiometrics_1, (224, 224))
        radiometrics_1_resized_img = np.repeat(radiometrics_1_resized.reshape(1, 224, 224), 3, axis=0).reshape(1, 3, 224, 224)

        radiometrics_2 = cv2.imread(path_to_dataset+str(i_folder)+
                                    '/geophysics/radiometrics/filtered_terrestrial_dose.tif', -1)
        radiometrics_2_resized = cv2.resize(radiometrics_2, (224, 224))
        radiometrics_2_resized_img = np.repeat(radiometrics_2_resized.reshape(1, 224, 224), 3, axis=0).reshape(1, 3, 224, 224)

        radiometrics_3 = cv2.imread(path_to_dataset+str(i_folder)+
                                    '/geophysics/radiometrics/filtered_thorium_ppm.tif', -1)
        radiometrics_3_resized = cv2.resize(radiometrics_3, (224, 224))
        radiometrics_3_resized_img = np.repeat(radiometrics_3_resized.reshape(1, 224, 224), 3, axis=0).reshape(1, 3, 224, 224)

        radiometrics_4 = cv2.imread(path_to_dataset+str(i_folder)+
                                    '/geophysics/radiometrics/filtered_uranium_ppm.tif', -1)
        radiometrics_4_resized = cv2.resize(radiometrics_4, (224, 224))
        radiometrics_4_resized_img = np.repeat(radiometrics_4_resized.reshape(1, 224, 224), 3, axis=0).reshape(1, 3, 224, 224)
        img_batch = np.concatenate((gravity_1_resized_img, gravity_2_resized_img, magnetics_1_resized_img,
                                    magnetics_2_resized_img, radiometrics_1_resized_img, radiometrics_2_resized_img,
                                    radiometrics_3_resized_img, radiometrics_4_resized_img), axis =0)
        img_batch_torch = torch.from_numpy(img_batch)

        img_var = Variable(img_batch_torch)
        features_var = resnet152(img_var)
        commodits_all.append(material_all)
        X_all.append(cord_x_all)
        Y_all.append(cord_y_all)
        heatmap = np.zeros((256, 256, 22))

        kernel = np.ones((5, 5), np.float32) / 25

        for i_heat in range(0, len(cord_x_all)):
            img = np.zeros((256, 256))
            img = cv2.circle(img, (int(np.round(cord_x_all[i_heat])), int(np.round(cord_y_all[i_heat]))), 10, (255, 255, 255), -1)
            dst = cv2.filter2D(img, -1, kernel)
            heatmap[:,:,material_all[i_heat]] = dst

        heatmap_all.append(heatmap)
        features_all.append(features_var.cpu().detach().numpy()[:,:,0,0])

    h5f = h5py.File('/media/iman/A8AC6E12AC6DDAF8/iman_ubuntu_dataset_unearthed/h5/data_'+str(k)+'.h5','w')
    h5f.create_dataset('heatmap', data=heatmap_all)
    h5f.create_dataset('features', data=features_all)
    h5f.close()
