# Import các thư viện và modules cần thiết
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras import backend as K
#Là các lớp cơ bản của mạng neural

class SmallerVGGNet: #Là một lớp chứa các phương thức để xây dựng mô hình SmallerVGGNet
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        
        #'inputShape' và 'chanDim' được sử dụng để xác định kích thước của đầu vào dựa trên 'image_data_format'
        # Giúp chuyển đổi kích thước đầu vào tùy thuộc vào định dạng của ảnh

        # Block 1: CONV => RELU => POOL
        #Gồm một lớp Conv2D với 32 filters
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape)) 
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        #Áp dụng hàm kích hoạt ReLU, lớp BatchNormalization, lớp MaxPooling2D để giảm kích thước và lớp Dropout để tránh overfitting

        # Block 2: (CONV => RELU) * 2 => POOL
        #Là một chuỗi lặp lại gồm hai lớp Conv2D với 64 filters
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        #Áp dụng hàm kích hoạt ReLU, lớp BatchNormalization, và lớp MaxPooling2D

        # Block 3: (CONV => RELU) * 2 => POOL
        #Là một chuỗi lặp lại tương tự với hai lớp Conv2D và 128 filters
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Block 4: FC => RELU
        #Gồm một lớp Flatten để chuyển đổi dữ liệu thành vector 1D, sau đó một lớp Dense (fully-connected) với 1024 nút
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        #Áp dụng hàm kích hoạt ReLU, lớp BatchNormalization và lớp Dropout

        # Block 5: Softmax classifier
        #Là lớp fully-connected cuối cùng với số lượng nút bằng số lớp classes
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        #Áp dụng hàm kích hoạt softmax để thu được xác suất dự đoán cho mỗi lớp

        return model