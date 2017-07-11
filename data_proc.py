import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
import cv2



bumble_path = '/home/pi/Videos/test_001/test_001_images/h/'
bee_path = '/home/pi/Videos/test_001/test_001_images/b/'

bumble_files = glob.glob(bumble_path + '*.jpg')
bee_files = glob.glob(bee_path + '*.jpg')

bumbles = [cv2.imread(bumble_file) for bumble_file in bumble_files]
bees = [cv2.imread(bee_file) for bee_file in bee_files]

thresh_val = 60
bumble_th = [np.zeros(bumble.shape[:-1]) + (bumble.mean(2) >= 60) for bumble in bumbles]
bee_th = [np.zeros(bee.shape[:-1]) + (bee.mean(2) >= 60) for bee in bees]

bumble_areas = np.array([bumble.sum() for bumble in bumble_th], dtype=float)
bee_areas = np.array([bee.sum() for bee in bee_th], dtype=float)

bumble_edges = np.array([np.sqrt(np.sum(
    (bumble[:-1, :-1] - bumble[1:, :-1])**2 + (bumble[:-1, :-1] - bumble[:-1, 1:])**2
    )) for bumble in bumble_th], dtype=float)
bee_edges = np.array([np.sqrt(np.sum(
    (bee[:-1, :-1] - bee[1:, :-1])**2 + (bee[:-1, :-1] - bee[:-1, 1:])**2
    )) for bee in bee_th], dtype=float)

bumble_ae = bumble_areas/bumble_edges
bee_ae = bee_areas/bee_edges
bumble_ea = 1./bumble_ae
bee_ea = 1./bee_ae

bumble_moments = [cv2.moments(bumble) for bumble in bumble_th]
bee_moments = [cv2.moments(bee) for bee in bee_th]

hu1 = lambda x_list : [x['nu02'] + x['nu20'] for x in x_list]
hu2 = lambda x_list : [(x['nu02'] - x['nu20'])**2 + 4*x['nu11']**2 for x in x_list]
hu3 = lambda x_list : [x['nu02'] + x['nu20'] for x in x_list]
hu4 = lambda x_list : [x['nu02'] + x['nu20'] for x in x_list]
bumble_hu1 = np.array(hu1(bumble_moments))
bee_hu1 = np.array(hu1(bee_moments))
bumble_hu2 = np.array(hu2(bumble_moments))
bee_hu2 = np.array(hu2(bee_moments))


bumble_matrix = np.array([bumble_ae, bumble_hu1, bumble_hu2]).T
bee_matrix = np.array([bee_ae, bee_hu1, bee_hu2]).T
data_matrix = np.concatenate([bumble_matrix, bee_matrix])
class_vector = np.zeros(data_matrix.shape[0])
class_vector[:bumble_matrix.shape[0]] = 1

U, S, V = np.linalg.svd(data_matrix)
bee_PC1 = bee_matrix.dot(V[0])
bee_PC2 = bee_matrix.dot(V[1])
bumble_PC1 = bumble_matrix.dot(V[0])
bumble_PC2 = bumble_matrix.dot(V[1])


plt.scatter(bee_PC1, bee_PC2, color='b')
plt.scatter(bumble_PC1, bumble_PC2, color='r')
plt.legend(['Bees', 'Bumblebees'])


plt.show()
