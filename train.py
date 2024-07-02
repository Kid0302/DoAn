# USAGE
# python train.py

# Import các thư viện và modules cần thiết
#Matplotlib là một thư viện Python phổ biến cho việc vẽ biểu đồ và hình ảnh.
import matplotlib
#Sử dụng để cài đặt backend của Matplotlib thành "Agg". "Agg" là một backend không yêu cầu giao diện đồ họa
matplotlib.use("Agg")

#Thư viện cho việc xây dựng và huấn luyện mạng neural.
#Keras là một giao diện cao cấp cho TensorFlow, giúp đơn giản hóa việc xây dựng mô hình mạng neural.
from tensorflow import keras 
#Cung cấp các công cụ để tạo dữ liệu đào tạo từ hình ảnh và thực hiện các phép biến đổi dữ liệu. 
from keras.preprocessing.image import ImageDataGenerator
#Tối ưu hóa dựa trên gradient được sử dụng để cập nhật trọng số trong mô hình mạng neural trong quá trình huấn luyện.
from keras.optimizers import Adam
#Chuyển đổi hình ảnh thành mảng numpy, cho phép mạng neural xử lý hình ảnh.
from keras.preprocessing.image import img_to_array
#biến đổi các nhãn lớp thành các vectơ nhị phân (one-hot encoding) để sử dụng trong huấn luyện mạng neural
from sklearn.preprocessing import LabelBinarizer
#Sử dụng để chia dữ liệu thành tập huấn luyện và tập kiểm tra để đánh giá hiệu suất mô hình.
from sklearn.model_selection import train_test_split
#Sử dụng để xây dựng và huấn luyện mạng neural cho bài toán cụ thể.
from CNN.smallervggnet import SmallerVGGNet
#Thư viện dùng để vẽ biểu đồ và hình ảnh.
import matplotlib.pyplot as plt
#Cung cấp hàm tiện ích cho việc xử lý đường dẫn tập tin và thư mục.
from imutils import paths
#Thư viện mạnh mẽ cho tính toán số học và xử lý mảng đa chiều, được sử dụng để xử lý dữ liệu hình ảnh.
import numpy as np
#Giúp phân tích và xử lý các đối số dòng lệnh được truyền vào cho tập lệnh, cho phép người dùng chỉ định các cài đặt.
import argparse
#Sử dụng để tạo số ngẫu nhiên để xáo trộn thứ tự của dữ liệu hình ảnh.
import random
#Thư viện cho việc lưu trữ và truy xuất dữ liệu dưới dạng các tệp nhị phân.
import pickle
#Thư viện được sử dụng để xử lý hình ảnh.
import cv2
#Cung cấp các chức năng để tương tác với hệ thống tệp và thư mục
import os

# Đối số dòng lệnh
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())
# Định nghĩa các đối số dòng lệnh
# Trong trường hợp này, chỉ có một đối số là đường dẫn đến biểu đồ độ chính xác/mất mát sau quá trình huấn luyện

# Cài đặt các thông số
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)
# Cài đặt các tham số quan trọng cho quá trình huấn luyện
# Bao gồm số lượng epoch, learning rate ban đầu, batch size và kích thước ảnh đầu vào

# Khởi tạo danh sách dữ liệu và nhãn
data = []
labels = []
# Khởi tạo danh sách dữ liệu và nhãn để lưu trữ ảnh và nhãn tương ứng

# Lấy danh sách đường dẫn đến ảnh và xáo trộn chúng
print("[INFO] Loading images...")
imagePaths = sorted(list(paths.list_images('Dataset')))
random.seed(42)
random.shuffle(imagePaths)

# Duyệt qua ảnh đầu vào
for imagePath in imagePaths:
	# Tải ảnh, tiền xử lý và lưu vào danh sách dữ liệu
	image = cv2.imread(imagePath)		
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# Trích xuất nhãn lớp từ đường dẫn ảnh và cập nhật danh sách nhãn
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)
# Duyệt qua tất cả các ảnh trong thư mục, tiền xử lý ảnh (resize, chuyển đổi sang mảng numpy), và lưu vào danh sách dữ liệu và nhãn

# Chuẩn hóa độ sáng pixel về khoảng [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] Data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# Biến đổi nhãn lớp thành dạng nhị phân (one-hot encoding)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# Chuẩn hóa độ sáng của pixel về khoảng [0, 1] và biến đổi nhãn lớp thành dạng nhị phân (one-hot encoding)

# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)
# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra để đánh giá hiệu suất mô hình

# Khởi tạo công cụ tạo ảnh để tăng cường dữ liệu
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
# Tạo công cụ tạo ảnh để tăng cường dữ liệu

# Khởi tạo mô hình
print("[INFO] Compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
	depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# Khởi tạo mô hình mạng và biên dịch nó với hàm mất mát, bộ tối ưu và các độ đo hiệu suất

# Huấn luyện mạng
print("[INFO] Training network...")
H = model.fit(
	x=aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)
# Huấn luyện mô hình trên dữ liệu huấn luyện và đánh giá trên tập kiểm tra

# Lưu mô hình vào tệp
print("[INFO] Serializing network...")
model.save('animals.keras', save_format="h5")
# Lưu mô hình đã huấn luyện và trình biến đổi nhãn (LabelBinarizer) để sử dụng sau này

# Lưu trình biến đổi nhãn lớp vào tệp
print("[INFO] Serializing label binarizer...")
with open('lb.pickle', "wb") as f:
    pickle.dump(lb, f)

# Vẽ biểu đồ về sự mất mát và độ chính xác trong quá trình huấn luyện
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])
# Vẽ biểu đồ về sự mất mát và độ chính xác trên cả tập huấn luyện và tập kiểm tra
# Biểu đồ này được lưu vào đường dẫn được chỉ định bởi đối số dòng lệnh