import numpy as np
import cv2

def compute_features(blob):
    thresh_val = 60
    blob_th = np.zeros(blob.shape[:-1]) + (blob.mean(2) >= 60)
    area = np.sum(blob_th)
    edges = np.sum(np.sqrt(
        (blob_th[:-1, :-1] - blob_th[1:, :-1])**2 + (blob_th[:-1, :-1] - blob_th[:-1, 1:])**2
    ))

    moments = cv2.moments(blob_th)
    hu1 = moments['nu02'] + moments['nu20']
    hu2 = np.sqrt((moments['nu02'] - moments['nu20'])**2 + 4*moments['nu11']**2)
    hu3 = np.sqrt((moments['nu30'] - 3*moments['nu12'])**2 + (3*moments['nu21'] - moments['nu03'])**2)
    hu4 = np.sqrt((moments['nu30'] + moments['nu12'])**2 + (moments['nu03'] + moments['nu21'])**2)

    return np.array([[float(edges)/area, hu1, hu2, hu3, hu4]])


def detect(features):
    intercept = -0.73607911
    weights = np.array([ 0.68399413,
                         2.11138001,
                        -3.85304383,
                         0.58279029,
                        -0.94086196 ])
    return intercept + features.dot(weights) > 0
