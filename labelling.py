import numpy as np
import picamera
import time
import os
import io
import cv2
from bee_camera import TrainingSetGenerator
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import glob


class Labeller(object):
    def __init__(self, images, path='', tg=None):
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
            self.tg.detect(image, draw_rectangles=False)
            print('Found blobs {} images out of {}'.format(i, images.shape[0]))
        
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

        blob = self.all_blobs[self.image_no][self.blob_no].copy()
        blob[:, :, 0] = blob[:, :, 2].copy()
        blob[:, :, 2] = self.all_blobs[self.image_no][self.blob_no][:, :, 0].copy()

        self.ax1.clear()
        self.ax1.imshow(image)
        self.ax1.set_title('Image {}'.format(self.image_no))
        self.ax2.clear()
        self.ax2.imshow(blob)
        self.ax2.set_title('Image {}, blob {}'.format(self.image_no, self.blob_no))


    def next_label(self):
        if self.blob_no + 1 >= len(self.all_blobs[self.image_no]):
            self.blob_no = 0
            self.image_no += 1
        else:
            self.blob_no += 1
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
    file_loc = raw_input('Folder with images to load: \n')
    file_loc += '/'
    file_paths = sorted(glob.glob(file_loc + '*.jpg'))

    load_total = int(raw_input('How many images to load (all images are loaded if this is 0): \n'))
    load_total = len(file_paths) if load_total == 0 else load_total

    save_loc = raw_input('Folder to save data in (for folder this just press enter): \n')
    save_loc = '.' if save_loc == '' else save_loc
    
    im_shape = cv2.imread(file_paths[0]).shape
    shape = (load_total, im_shape[0], im_shape[1], 3)
    images = np.zeros(shape, dtype=np.uint8)
    for i in range(load_total):
        file_path = file_loc + '{:03d}_image.jpg'.format(i)
        images[i, :, :] = cv2.imread(file_path)
        print('Loaded image {} of {}'.format(i+1, load_total))

    labeller = Labeller(images, path=save_loc)
    labeller.start_labelling()
    