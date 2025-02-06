import numpy as np
import cv2
from pathlib import Path
from sklearn.utils import shuffle
class shapes():

    def __init__(self):
        self.x_data = None
        self.y_data = None

    # Preprocessing    
    def load_shapes(self):
        """
        Load the paths of all shape images in the dataset.

        Returns a dictionary with the keys 'ellipse', 'rectangle', and 'triangle'.
        Each value is a list of paths to the images of the corresponding shape.
        """
        users = []
        with open("users.txt", "r") as file:
            for line in file:
                users.extend(line.split())

        path = "C:/Users/NR_Se/OneDrive/Documents/GitHub/Final_Project_Machine_Learning/hand-drawn-shapes-dataset-main/data/"
        shapes = ["ellipse", "rectangle", "triangle"]
        data = { "ellipse": [], "rectangle": [], "triangle": []}
        for user in users:
            for shape in shapes:
                file_path = Path(path + user) / "images" / shape
                if file_path.exists():
                    data[shape].extend(str(file) for file in file_path.glob("*.png"))
                    
        return data

    def preprocess_image(self, file_paths):
        """
        Preprocesses a list of image file paths by reading, resizing, and normalizing them.

        Args:
            file_paths (list of str): List of file paths to the images to be processed.

        Returns:
            np.ndarray: A numpy array of processed images, where each image is resized 
            to 28x28 pixels and pixel values are normalized to the range [0, 1].
        """
        processed_images = []

        for file in file_paths:
            image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (28, 28))
                image = image / 255
                processed_images.append(image)
        # print(len(processed_images))
        return np.array(processed_images)

    def get_labels(self, file_paths):
        """
        Gets the labels corresponding to the given file paths.

        Args:
            file_paths (dict): A dictionary with the keys 'ellipse', 'rectangle', and 'triangle'.
                Each value is a list of file paths to the images of the corresponding shape.

        Returns:
            np.ndarray: A numpy array of labels, where 0 corresponds to ellipse, 1 to rectangle, and 2 to triangle.
        """
        labels = []
        shape_to_label = { "ellipse": 0, "rectangle": 1, "triangle": 2}
        for shape, files in file_paths.items():
            labels.extend([shape_to_label[shape]] * len(files))

        # print(len(labels))
        return np.array(labels)
    
    def load_xy(self):
        shape_data = self.load_shapes()

        ellipse_data = self.preprocess_image(shape_data['ellipse'])
        rectangle_data = self.preprocess_image(shape_data['rectangle'])
        triangle_data = self.preprocess_image(shape_data['triangle'])

        self.x_data = np.concatenate((ellipse_data, rectangle_data, triangle_data), axis=0) # Combine all shape data
        self.y_data = self.get_labels(shape_data)

        # Shuffle the data 
        self.x_data, self.y_data = shuffle(self.x_data, self.y_data) 

        # Reshape the data
        self.y_data = self.y_data.reshape(-1, 1) # (20005, 1)
        self.x_data = self.x_data.reshape(-1, 784) #(20005, 784)

        self.y_data = self.y_data.T
        self.x_data = self.x_data.T

        return self.x_data, self.y_data



    

    


   
        
