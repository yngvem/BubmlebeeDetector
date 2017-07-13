import numpy as np
import time
import os
import io
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import glob
from record_params import *
from compute_features import compute_features as _cf
from compute_features import detect as _detect



class TrainingSetGenerator(object):
    def __init__(self, 
                 filter_area=None, min_area=None, max_area=None,
                 filter_circ=None, min_circ=None, max_circ=None,
                 filter_convex=None, min_convex=None, max_convex=None):
        """Initializer for the training set generator.
        """
        blob_params = cv2.SimpleBlobDetector_Params()
        
        blob_params.filterByArea = filter_area if filter_area is not None else blob_params.filterByArea
        blob_params.minArea = min_area if min_area is not None else blob_params.minArea
        blob_params.maxArea = max_area if max_area is not None else blob_params.maxArea

        blob_params.filterByCircularity = filter_circ if filter_circ is not None else blob_params.filterByCircularity
        blob_params.minCircularity = min_circ if min_circ is not None else blob_params.minCircularity
        blob_params.maxCircularity = max_circ if max_circ is not None else blob_params.maxCircularity

        blob_params.filterByConvexity = filter_convex if filter_area is not None else blob_params.filterByConvexity
        blob_params.minConvexity = min_convex if min_convex is not None else blob_params.minConvexity
        blob_params.maxConvexity = max_convex if max_convex is not None else blob_params.maxConvexity

        blob_params.filterByInertia = False
        blob_params.filterByColor = False

        self.detector = cv2.SimpleBlobDetector(blob_params)
        self.blobs = []
        self.features = []
        self.scores = []
        self.image_no = 0

    def detect(self, frame, thresh=50, draw_rectangles=True):
        # Update data lists    
        self.blobs.append([])
        self.features.append([])
        self.scores.append([])

        th_err = cv2.threshold(
            src = frame.mean(axis=2).astype(np.uint8),
            thresh = thresh,
            maxval = 255,
            type = cv2.THRESH_BINARY_INV
        )[1]

        # Find blobs:
        keypoints = self.detector.detect(th_err)
        th_err2 = np.array(th_err)

        # Find ROI's
        mask = np.zeros(np.array(th_err.shape)+2, dtype=np.uint8)
        blobs = [0]*len(keypoints)
        areas = [0]*len(keypoints)
        for i, k in enumerate(keypoints):
            x = int(k.pt[0])
            y = int(k.pt[1])
            areas[i], blobs[i] = cv2.floodFill(th_err2, mask*0, (x, y), i+1)

        blobs = np.array(blobs)

        # Check if any of the ROIs contain a bumblebee
        rect_list = []
        for i in range(len(keypoints)):
            x_min = blobs[i, 0]
            x_max = x_min + blobs[i, 2]
            y_min = blobs[i, 1]
            y_max = y_min + blobs[i, 3]
            roi = frame[y_min:y_max, x_min:x_max]
            feats = self.compute_features(roi)
            self.features[self.image_no].append(feats)
            self.blobs[self.image_no].append(roi)
            self.scores[self.image_no].append(self.class_score(roi))
            rect_list.append(((x_min-3, y_min-3), (x_max+2, y_max+2)))

        # Draw rectangles
        for rect in rect_list:
            cv2.rectangle(frame, rect[0], rect[1], (255, 0, 0), 1)


        self.image_no +=1
        return True

    def compute_features(self, blob):
        """Compute the image features of given blob.
        """
        return _cf(blob)[0]
    

    def class_score(self, blob):
        """Classify a blob.
        """
        return _detect(self.compute_features(blob))

    



class Labeller(object):
    def __init__(self, images, path='', tg=None, thresh=50):
        self.fig1 = plt.figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.fig2 = plt.figure()
        self.ax2 = self.fig2.add_subplot(211)

        # Create buttons
        self.bt1_ax = self.fig2.add_subplot(234)
        self.bee_button = Button(self.bt1_ax, 'Bee')
        self.bee_button.on_clicked(self.curr_bee)
        
        self.bt2_ax = self.fig2.add_subplot(235)
        self.bumble_button = Button(self.bt2_ax, 'Bumblebee')
        self.bumble_button.on_clicked(self.curr_bumble)

        self.bt3_ax = self.fig2.add_subplot(236)
        self.other_button = Button(self.bt3_ax, 'Other')
        self.other_button.on_clicked(self.curr_other)

        self.tg = TrainingSetGenerator(
            min_area=30,
            max_area=1000,
            filter_area=True,
            filter_circ=False,
            filter_convex=False
        ) if tg is None else tg

        for i, image in enumerate(images):
            self.tg.detect(image, draw_rectangles=False, thresh=thresh)
            print('Found blobs {} images out of {}'.format(i+1, images.shape[0]))
        
        self.images = images
        self.all_blobs = self.tg.blobs
        self.blob_labels = [[None for _ in blobs] for blobs in self.all_blobs]
        self.image_no = 0
        self.blob_no = 0
        self.path = path


    def start_labelling(self):
        self.update_plot()
        plt.show()
    

    def update_plot(self):
        image = self.images[self.image_no].copy()
        image[:, :, 0] = image[:, :, 2].copy()
        image[:, :, 2] = self.images[self.image_no][:, :, 0].copy()
        
        if len(self.all_blobs[self.image_no]) > 0:
            blob = self.all_blobs[self.image_no][self.blob_no].copy()
            blob[:, :, 0] = blob[:, :, 2].copy()
            blob[:, :, 2] = self.all_blobs[self.image_no][self.blob_no][:, :, 0].copy()

            self.ax1.clear()
            self.ax1.imshow(image)
            self.ax1.set_title('Image {}'.format(self.image_no))
            self.ax2.clear()
            self.ax2.imshow(blob)
            self.ax2.set_title('Image {}, blob {}'.format(self.image_no, self.blob_no))
        else:
            self.image_no += 1
            self.update_plot()


    def next_label(self):
        if self.blob_no + 1 >= len(self.all_blobs[self.image_no]):
            self.blob_no = 0
            self.image_no += 1
        else:
            self.blob_no += 1
        if self.image_no >= len(self.all_blobs):
            plt.close('all')
        else:
            self.update_plot()


    def curr_bee(self, event):
        print('Bee')
        if not os.path.exists('{}/b/'.format(self.path)):
            os.mkdir('{}/b/'.format(self.path))
        cv2.imwrite(
            '{}/b/I{}_B{}.jpg'.format(self.path, self.image_no, self.blob_no),
            self.all_blobs[self.image_no][self.blob_no]
        )
        self.blob_labels[self.image_no][self.blob_no] = 'b'
        self.next_label()


    def curr_bumble(self, event):
        print('Bumblebee')
        if not os.path.exists('{}/h/'.format(self.path)):
            os.mkdir('{}/h/'.format(self.path))
        cv2.imwrite(
            '{}/h/I{}_B{}.jpg'.format(self.path, self.image_no, self.blob_no),
            self.all_blobs[self.image_no][self.blob_no]
        )
        self.blob_labels[self.image_no][self.blob_no] = 'h'
        self.next_label()
        


    def curr_other(self, event):
        print('Other')
        if not os.path.exists('{}/o/'.format(self.path)):
            os.mkdir('{}/o/'.format(self.path))
        cv2.imwrite(
            '{}/o/I{}_B{}.jpg'.format(self.path, self.image_no, self.blob_no),
            self.all_blobs[self.image_no][self.blob_no]
        )
        self.blob_labels[self.image_no][self.blob_no] = 'o'
        self.next_label()


    def save_images(self, event):
        path = self.path
        if not os.path.exists('b/'):
            os.mkdir('b/')
        if not os.path.exists('h/'):
            os.mkdir('h/')
        if not os.path.exists('o/'):
            os.mkdir('o/')
                     
        for i, blobs in enumerate(self.all_blobs):
            for j, blob in enumerate(blobs):
                if blob_labels[i][j] is not None:
                    cv2.imwrite(
                        '{}/I{}_B{}.jpg'.format(blob_labels[i][j], i, j),
                        blob
                    )
        print('Saved labelled blobs')
          
        

if __name__ == '__main__':
    file_loc = raw_input('Folder with images to load (warning, this process will overwrite old files without warning!): \n')
    file_loc += '/'
    file_paths = sorted(glob.glob(file_loc + '*.jpg'))

    load_total = raw_input('How many images to load (all images are loaded if this is left blank): \n')
    load_total = int(load_total) if load_total != '' else len(file_paths)
    
    start_at = raw_input('Start at image no (starts at first image if this is left blank): \n')
    start_at = int(start_at) if start_at != '' else 0
    
    save_loc = raw_input('Folder to save data in (for folder this just press enter)\n'
                         ' Warning: this process will overwrite old files without warning! \n')
    save_loc = '.' if save_loc == '' else save_loc
    
    im_shape = cv2.imread(file_paths[0]).shape
    shape = (load_total, im_shape[0], im_shape[1], 3)
    images = np.zeros(shape, dtype=np.uint8)
    for i in range(start_at, load_total):
        file_path = file_loc + '{:03d}_image.jpg'.format(i)
        images[i, :, :] = cv2.imread(file_path)
        print('Loaded image {} of {}'.format(i+1, load_total))

    labeller = Labeller(images, path=save_loc, thresh=blob_thresh)
    labeller.start_labelling()
    
