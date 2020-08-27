import tensorflow as tf
import numpy as np
import cv2
import yolo_constants as yc

def getImageAdresses(imgList, path):
    imgPaths = list()
    for i in range(len(imgList)):
        imgPaths.append(path + "\\" + imgList[i]['file_name'])
    return imgPaths

def addBox(img, x, y, width, height, label=""):
    tl = (int(x), int(y))
    br = (int(x+width), int(y+height))
    img = cv2.rectangle(img, tl, br, (255,0,0), 4)
    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    return img
	
def getBoundingBoxCenter(annotation):
    box = annotation['bbox']
    x = box[0] + box[2]*0.5
    y = box[1] + box[3]*0.5
    center = (int(x) ,int(y))
    return center
	
#Code from https://www.tensorflow.org/tutorials/load_data/images
def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize_images(image, [yc.IMAGE_SIZE_WIDTH, yc.IMAGE_SIZE_HEIGHT])
    image /= 255.0  # normalize to [0,1] range
    return image

#Code from https://www.tensorflow.org/tutorials/load_data/images
def load_and_preprocess_image(path):
    image = tf.read_file(path)
    return preprocess_image(image)
	
def getNormalizedCoordinates(img, point):
    height = img["height"]
    width = img["width"]
    
    #x and y in [0,1] interval
    return (point[0] / width, point[1] / height)

def getResizeCoordinates(img, point):
    normCenter = getNormalizedCoordinates(img, point)
    return (int(normCenter[0] * yc.IMAGE_SIZE_WIDTH), int(normCenter[1] * yc.IMAGE_SIZE_HEIGHT))


def getNormalizedCellCoordinates(cell):
    return (cell[0] / yc.CELL_NUMBER_VERT, cell[1] / yc.CELL_NUMBER_HORI)

def getResizeCellCoordinates(cell):
    normCell = getNormalizedCellCoordinates(cell)
    return (int(normCell[1] * yc.IMAGE_SIZE_HEIGHT), int(normCell[0] * yc.IMAGE_SIZE_WIDTH))

#Get center coordinates relative to row and col coordinates
def isPointInsideCell(img, point, cell):
    cellCoords = getResizeCellCoordinates(cell)
    pointCoords = getResizeCoordinates(img, point)
    if (pointCoords[0] < cellCoords[0]) or (pointCoords[1] < cellCoords[1]):
        return False
    if (pointCoords[0] > (cellCoords[0] + yc.CELL_WIDTH)) or (pointCoords[1] > (cellCoords[1] + yc.CELL_HEIGHT)):
        return False
    return True
	
def getClassValuesForCell(img, classValues, cell):
    for clsID in yc.CATEGORY_IDS:
        value = 0.0
        for anno in yc.ANNOTATIONS:
            if anno['image_id']!= img["id"]:
                continue
            if clsID != anno['category_id']:
                continue
            center = getBoundingBoxCenter(anno)
            if isPointInsideCell(img, center, cell):
                value = 1.0
        classValues.append(value)
		
def getBoxParamsForCell(img, boxParams, cell):
    for bb in range(yc.COUNT_BOUNDING_BOXES):
        params = [0.0, 0.0, 0.0, 0.0]
        for anno in yc.ANNOTATIONS:
            if anno['image_id']!= img["id"]:
                continue
            if anno['category_id'] != yc.CATEGORY_IDS[0] and anno['category_id'] != yc.CATEGORY_IDS[1]:
                continue
            center = getBoundingBoxCenter(anno)
            if isPointInsideCell(img, center, cell):
                #First: add box center coordinates in normalized state to output tensor
                cellheight = img["height"] / yc.CELL_NUMBER_VERT
                cellwidth = img["width"] / yc.CELL_NUMBER_HORI
				
                centerCoords = getResizeCoordinates(img, center)
                cellCoords = getResizeCellCoordinates(cell)
                params[0] = (centerCoords[0] - cellCoords[0]) / cellwidth
                params[1] = (centerCoords[1] - cellCoords[1]) / cellheight
                
                #Second: add box width and height
                box = anno['bbox']
                params[2] = box[2] / img["width"]
                params[3] = box[3] / img["height"]
                
                #Important: The Confidence Score is not part of the labeled data, 
				#			but it IS part of the output tensor.
				#			It will be calculated in the model's loss function
    
        boxParams.append(params)
		
def createImageDataset(imgPaths):
    path_ds = tf.data.Dataset.from_tensor_slices(imgPaths)
    image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=yc.AUTOTUNE)
    return image_ds
	
def createYOLO_v1_Model(tiny = True):
	#Not the real tiny Yolo. I had to reduce the number of layers for it to run on my
	#laptop. This caused worse behaviour than the normal Tiny Yolo, 
	#but acceptable for study purposes.
    if tiny:
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), strides=1, padding='same', input_shape=(448, 448, 3)),
			tf.keras.layers.LeakyReLU(0.1),  
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
			
            tf.keras.layers.Conv2D(32, (3,3), strides=1, padding='same'),
			tf.keras.layers.LeakyReLU(0.1),  
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
			
            tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='same'),
			tf.keras.layers.LeakyReLU(0.1),  
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
			
            tf.keras.layers.Conv2D(128, (3,3), strides=1, padding='same'),
			tf.keras.layers.LeakyReLU(0.1),  
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
			
            tf.keras.layers.Conv2D(256, (3,3), strides=1, padding='same'),
			tf.keras.layers.LeakyReLU(0.1),  
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
			
            tf.keras.layers.Conv2D(512, (3,3), strides=1, padding='same'),
			tf.keras.layers.LeakyReLU(0.1),  
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
			
            tf.keras.layers.Conv2D(1024, (3,3), strides=1, padding='same'),
			tf.keras.layers.LeakyReLU(0.1),  
			
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024),
            tf.keras.layers.Dropout(0.5),  
			tf.keras.layers.LeakyReLU(0.1),  			
            tf.keras.layers.Dense(yc.CELL_NUMBER_HORI * yc.CELL_NUMBER_VERT * (yc.COUNT_BOUNDING_BOXES*5 + len(yc.CLASSES)))
        ])
        return model
    
		
