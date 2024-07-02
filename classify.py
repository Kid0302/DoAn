import tkinter as tk # Thư viện tạo giao diện đồ họa
from tkinter import filedialog, ttk 
# Cho phép người dùng chọn tệp tin từ hệ thống
# Thư viện cung cấp các widget giao diện người dùng nâng cao
from tkinter import messagebox
from PIL import Image, ImageTk # Cho việc xử lý và hiển thị hình ảnh
import numpy as np # Thư viện cho xử lý mảng và ma trận
from tensorflow import keras # Thư viện học máy và mạng nơ-ron
import tensorflow as tf 
from keras.preprocessing.image import img_to_array # Chuyển đổi hình ảnh thành mảng numpy
import matplotlib.pyplot as plt # Vẽ biểu đồ
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # Cho việc tích hợp biểu đồ vào giao diện Tkinter
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk# Cho việc tích hợp biểu đồ vào giao diện Tkinter
import time # Thư viện xử lý thời gian
import cv2 # Thư viện xử lý ảnh và video
import pickle # Cho việc lưu và đọc dữ liệu dưới dạng nhị phân
import os # Thư viện tương tác với hệ điều hành
from sklearn.metrics import roc_curve, auc # Cho tính toán đường ROC và diện tích dưới đường ROC

#-----------Hàm-----------#

# Đường dẫn đến thư mục lưu trữ ảnh sau khi phân loại
thu_muc_luu_tru = "C:\\Users\\Admin\\Desktop\\test\\Classify"

# Thư mục chứa dữ liệu huấn luyện
thu_muc_huan_luyen = "C:\\Users\\Admin\\Desktop\\test\\Dataset"

# Hàm để đếm số lượng ảnh cho từng loại con vật
def dem_anh_theo_loai():
    # Tạo một biến để lưu trữ nội dung thông báo
    thong_bao = ""
    
    # Kiểm tra xem thư mục chứa dữ liệu huấn luyện có tồn tại không
    if os.path.exists(thu_muc_huan_luyen):
        for thu_muc_loai in os.listdir(thu_muc_huan_luyen):
            duong_dan_loai = os.path.join(thu_muc_huan_luyen, thu_muc_loai)
            if os.path.isdir(duong_dan_loai):
                so_luong_anh = len(os.listdir(duong_dan_loai))
                thong_bao += f"{thu_muc_loai}: {so_luong_anh} ảnh\n"
    
    # Hiển thị thông tin số lượng ảnh
    if thong_bao:
        messagebox.showinfo("Số lượng ảnh", thong_bao)
    else:
        messagebox.showinfo("Thông báo", "Không tìm thấy ảnh trong thư mục huấn luyện.")

def lam_moi_du_lieu():
    nhan_ket_qua.config(text="Kết quả")
    thong_tin_phan_loai.delete(1.0, tk.END)
    khung_canvas_anh.delete("all")
    canvas_bieu_do.get_tk_widget().destroy()
    canvas_bieu_do.draw()
            
# Hàm để duyệt và chọn một hình ảnh từ máy tính
def duyet_anh():
    lam_moi_du_lieu()  # Thêm dòng này để làm mới dữ liệu và xóa thông tin cũ
    duong_dan_anh = filedialog.askopenfilename()
    if duong_dan_anh:
        phan_loai_anh(duong_dan_anh)
        ve_bieu_do_roc(duong_dan_anh) # Thêm dòng này để vẽ biểu đồ ROC ngay sau khi phân loại ảnh

# Nạp mô hình đã huấn luyện và label binarizer
model = keras.models.load_model('animals.keras')
lb = pickle.loads(open('lb.pickle', "rb").read())
    
# Hàm để phân loại hình ảnh đã chọn
def phan_loai_anh(duong_dan_anh):
    global anh_da_phan_loai

    # Nạp và tiền xử lý hình ảnh
    # Đọc ảnh từ đường dẫn
    anh = cv2.imread(duong_dan_anh)
    # Chuyển ảnh sang kích thước mong muốn
    anh = cv2.resize(anh, (96, 96))
    anh = anh.astype("float") / 255.0
    anh = img_to_array(anh)
    anh = np.expand_dims(anh, axis=0)

    # Phân loại hình ảnh đầu vào
    xac_suat = model.predict(anh)[0]

    # Lấy N dự đoán hàng đầu
    top_indices = np.argsort(xac_suat)[::-1][:10]

    # Hiển thị kết quả phân loại
    ket_qua = []
    for idx in top_indices:
        nhan = lb.classes_[idx]
        xac_suat_dung = xac_suat[idx] * 100
        ket_qua_text = f"{nhan}: {xac_suat_dung:.2f}%"
        ket_qua.append(ket_qua_text)

    # Hiển thị kết quả phân loại hàng đầu trong nhãn
    nhan_ket_qua.config(text=ket_qua[0] + " (đúng)")

    # Hiển thị thông tin phân loại trong Text widget
    thong_tin_phan_loai.delete(1.0, tk.END)  # Xóa nội dung trước
    thong_tin_phan_loai.insert(tk.END, "\n".join(ket_qua))

    # Xóa biểu đồ phân loại cũ trên canvas
    plt.clf()

    # Tạo biểu đồ cột cho 5 kết quả hàng đầu
    nhan = [kq.split(":")[0] for kq in ket_qua]
    xac_suat = [float(kq.split(":")[1][1:-1]) for kq in ket_qua]
    plt.barh(nhan, xac_suat)
    plt.xlabel('Xác suất (%)', fontweight="bold")
    plt.title('Kết quả phân loại', fontweight="bold")
    plt.gca().invert_yaxis()

    # Lưu hình ảnh đã phân loại
    anh_da_phan_loai = Image.open(duong_dan_anh)
    anh_da_phan_loai.thumbnail((200, 200))  # Điều chỉnh kích thước ảnh

    # Hiển thị ảnh trong Canvas
    anh_photo = ImageTk.PhotoImage(anh_da_phan_loai)
    khung_canvas_anh.create_image(0, 0, anchor=tk.NW, image=anh_photo)
    khung_canvas_anh.photo = anh_photo  # Giữ tham chiếu đến ảnh để tránh việc thu gom rác

    # Lấy tên thư mục dự đoán
    thu_muc_nhan = os.path.join(thu_muc_luu_tru, lb.classes_[top_indices[0]])

    # Kiểm tra và tạo thư mục nếu cần
    if not os.path.exists(thu_muc_nhan):
        os.makedirs(thu_muc_nhan)

    # Tạo tên tệp ảnh dựa trên thời gian và nhãn
    thoi_gian = int(time.time())
    ten_tep_anh = f"{thoi_gian}.png"
    duong_dan_anh = os.path.join(thu_muc_nhan, ten_tep_anh)

    # Lưu ảnh vào thư mục tương ứng
    anh_da_phan_loai.save(duong_dan_anh)

    canvas_bieu_do.draw()

def ve_bieu_do_roc(duong_dan_anh=None):
    global model, lb, canvas_bieu_do

    if duong_dan_anh is not None:
        
        # Nạp và tiền xử lý hình ảnh từ đường dẫn đã chọn
        anh = cv2.imread(duong_dan_anh)
        anh = cv2.resize(anh, (96, 96))
        anh = anh.astype("float") / 255.0
        anh = img_to_array(anh)
        anh = np.expand_dims(anh, axis=0)

        # Dự đoán kết quả cho ảnh đã chọn
        predictions = model.predict(anh)

        # Chuyển định dạng nhãn thành one-hot encoding
        labels_one_hot = np.zeros_like(predictions)
        labels_one_hot[np.arange(len(predictions)), predictions.argmax(1)] = 1

        # Tính toán ROC curve cho mỗi lớp
        fpr, tpr, _ = roc_curve(labels_one_hot.ravel(), predictions.ravel())
        roc_auc = auc(fpr, tpr)

        # Vẽ biểu đồ ROC curve
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Đường ROC (diện tích = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

        # Vẽ đường tiệm cận trên và dưới đường ROC
        tpr_max = max(tpr)
        ax.plot(fpr, [tpr_max] * len(fpr), color='green', linestyle='--', label='Tiệm cận trên')
        ax.plot([fpr[np.argmax(tpr)]], [tpr_max], marker='o', markersize=8, color='green')
        ax.annotate(f'Tiệm cận trên: {tpr_max:.2f}', 
                    xy=(fpr[np.argmax(tpr)], tpr_max),
                    xytext=(fpr[np.argmax(tpr)] - 0.15, tpr_max - 0.15),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    )

        tpr_min = min(tpr)
        ax.plot(fpr, [tpr_min] * len(fpr), color='blue', linestyle='--', label='Tiệm cận dưới')
        ax.plot([fpr[np.argmax(fpr)]], [tpr_min], marker='o', markersize=8, color='blue')
        ax.annotate(f'Tiệm cận dưới: {tpr_min:.2f}', 
                    xy=(fpr[np.argmax(fpr)], tpr_min),
                    xytext=(fpr[np.argmax(fpr)] - 0.15, tpr_min + 0.25),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    )

        # Vẽ đường trung bình và xác định lớp có độ tin cậy cao nhất
        average_curve = (fpr + tpr) / 2
        max_confidence_class = np.argmax(average_curve)
        max_confidence_value = average_curve[max_confidence_class]

        # In thông tin về lớp có độ tin cậy cao nhất
        print('\nTất cả các lớp:')
        for i, confidence_value in enumerate(average_curve):
            class_name = lb.classes_[i]
            print(f'Lớp {class_name}: Độ tin cậy {confidence_value:.2f}')

        print(f'\nLớp có độ tin cậy cao nhất: {lb.classes_[max_confidence_class]} với giá trị {max_confidence_value:.2f}\n')

        # Vẽ điểm và chú thích trên đồ thị
        ax.plot(fpr[max_confidence_class], tpr[max_confidence_class], marker='o', markersize=8, color='red', label='Điểm tin cậy cao nhất')
        ax.annotate(f' ({max_confidence_value:.2f})', 
                    xy=(fpr[max_confidence_class], tpr[max_confidence_class]),
                    xytext=(fpr[max_confidence_class] - 0.2, tpr[max_confidence_class] + 0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    )
        #Lớp {lb.classes_[max_confidence_class]}
        # Chọn một ngưỡng quyết định (ví dụ: 0.5) để phân loại ảnh
        decision_threshold = 0.5

        # Dự đoán kết quả cho ảnh đã chọn
        image_prediction = model.predict(anh)[0]

        # Phân loại ảnh dựa trên ngưỡng quyết định
        predicted_class = lb.classes_[np.argmax(image_prediction)]
        predicted_confidence = image_prediction[np.argmax(image_prediction)]

        # Hiển thị thông tin về ảnh được chọn
        print(f'\nẢnh được chọn (Sử dụng từ nút chọn ảnh):')
        print(f' - Dự đoán: Lớp {predicted_class} với độ tin cậy {predicted_confidence:.2f}')
        print(f' - Ngưỡng quyết định: {decision_threshold}\n')

        # Vẽ biểu đồ độ tin cậy của ảnh được chọn
        ax.plot([decision_threshold, decision_threshold], [0, 1], color='purple', linestyle='--', lw=2, label=f'Ngưỡng quyết định ({decision_threshold})')
        ax.plot([decision_threshold], [predicted_confidence], marker='o', markersize=8, color='purple', label=f'Độ tin cậy ({predicted_confidence:.2f})')
        ax.annotate(f'Ảnh đã chọn', 
                    xy=(decision_threshold, predicted_confidence),
                    xytext=(decision_threshold + 0.1, predicted_confidence - 0.2),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    )

        # Chú thích và hiển thị đồ thị
        ax.set_xlabel('Tỷ lệ False Positive')
        ax.set_ylabel('Tỷ lệ True Positive')
        ax.set_title('Biểu đồ độ chính xác sau khi phân loại')
        ax.legend(loc='lower right')

        plt.tight_layout()  # Điều chỉnh layout để không bị chồng lấn
        plt.grid(True)
        plt.show()
        
        # Cập nhật FigureCanvasTkAgg hiện tại
        canvas_bieu_do.figure = fig
        canvas_bieu_do.draw()
              
    else:
        print("Vui lòng chọn một hình ảnh trước khi vẽ biểu đồ.")
        
# Cờ để cập nhật thống kê từ thư mục
co_cap_nhat_thong_ke = True

# Hàm để cập nhật thống kê từ thư mục con
def cap_nhat_thong_ke_tu_cac_thu_muc_con(thu_muc_goc):
    so_luong = {}
    for ten_thu_muc in os.listdir(thu_muc_goc):
        duong_dan_thu_muc = os.path.join(thu_muc_goc, ten_thu_muc)
        if os.path.isdir(duong_dan_thu_muc):
            so_luong[ten_thu_muc] = len(os.listdir(duong_dan_thu_muc))
    return so_luong

# Hàm cập nhật thống kê
def cap_nhat_thong_ke():
    global thong_ke_so_luong_dong_vat
    thong_ke_so_luong_dong_vat = cap_nhat_thong_ke_tu_cac_thu_muc_con(thu_muc_luu_tru)
    listbox_thong_ke.delete(0, tk.END)
    for nhan, so_luong in thong_ke_so_luong_dong_vat.items():
        listbox_thong_ke.insert(tk.END, f"{nhan}: {so_luong} con")

# Hàm để xóa dữ liệu thống kê
def xoa_thong_ke():
    global thong_ke_so_luong_dong_vat
    thong_ke_so_luong_dong_vat = {}  # Xóa dữ liệu thống kê
    cap_nhat_thong_ke()

# Hàm để xóa dữ liệu thống kê và các tệp trong thư mục con
def xoa_thong_ke_va_cac_tep():
    for nhan, so_luong in thong_ke_so_luong_dong_vat.items():
        thu_muc_nhan = os.path.join(thu_muc_luu_tru, nhan)
        if os.path.exists(thu_muc_nhan):
            for ten_tep in os.listdir(thu_muc_nhan):
                duong_dan_tep = os.path.join(thu_muc_nhan, ten_tep)
                os.remove(duong_dan_tep)
    xoa_thong_ke()

# Hàm cập nhật thống kê từ thư mục
def cap_nhat_thong_ke_tu_thu_muc():
    global thong_ke_so_luong_dong_vat
    thong_ke_so_luong_dong_vat = cap_nhat_thong_ke_tu_cac_thu_muc_con(thu_muc_luu_tru)
    cap_nhat_thong_ke()
    global co_cap_nhat_thong_ke
    co_cap_nhat_thong_ke = False

# Hàm để cài đặt thông tin liên quan
def cai_dat_thong_tin_lien_quan():
    thong_tin_sinh_vien.config(text="Nguyễn Tấn Phát - Bùi Lê Thanh Ngân - Nguyễn Thanh Phúc")
    thong_tin_giao_vien.config(text="Huỳnh Thị Châu Lan")
    thong_tin_de_tai.config(text="Xây dựng ứng dụng phân loại động vật")

#-----------Xử lý hàm-----------#
    
# Tạo một cửa sổ giao diện
root = tk.Tk()
root.title("Đồ án chuyên ngành: Ứng dụng phân loại động vật")

# Kích thước cửa sổ
chieu_rong_cua_so = 1340
chieu_cao_cua_so = 830
root.geometry(f"{chieu_rong_cua_so}x{chieu_cao_cua_so}")
root.configure(bg="light blue")

# Biến toàn cục để lưu trữ hình ảnh đã phân loại và số lượng động vật
anh_da_phan_loai = None
thong_ke_so_luong_dong_vat = {}

# Định nghĩa bieu_do_tin_cay như một biến toàn cục
bieu_do_tin_cay = None

# Tạo một kiểu nút tùy chỉnh
kieu_nut = ttk.Style()
kieu_nut.configure("NutTuyChinh.TButton", padding=(10, 5), font=("Helvetica", 13, "bold"))

# Tạo một khung cho việc nhập hình ảnh
khung_nhap_anh = ttk.LabelFrame(root, text="Nhập ảnh phân loại", padding=10)
khung_nhap_anh.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

khung_nhap_anh.columnconfigure(0, weight=1)
khung_nhap_anh.rowconfigure(2, weight=1)

nhan_anh = ttk.Label(khung_nhap_anh, text="Vui lòng chọn ảnh cần phân loại", font=("Helvetica", 13, "bold"))
nhan_anh.grid(row=0, column=0, padx=5, pady=5)

khung_canvas_anh = tk.Canvas(khung_nhap_anh, width=200, height=200, bg="white")
khung_canvas_anh.grid(row=1, rowspan=2, column=0, padx=5, pady=5 )

# Tạo một nút để đếm ảnh
nut_dem_anh = ttk.Button(khung_nhap_anh, text="Số lượng ảnh trong huấn luyện", command=dem_anh_theo_loai, style="NutTuyChinh.TButton")
nut_dem_anh.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")

# Tạo một nút để chọn ảnh
nut_chon_anh = ttk.Button(khung_nhap_anh, text="Hãy chọn ảnh", command=duyet_anh, style="NutTuyChinh.TButton")
nut_chon_anh.grid(row=4, column=0, padx=5, pady=5, sticky="nsew")

# Tạo một khung cho kết quả phân loại
khung_ket_qua_phan_loai = ttk.LabelFrame(root, text="Kết quả phân loại", padding=10)
khung_ket_qua_phan_loai.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
khung_ket_qua_phan_loai['width'] = 300
khung_ket_qua_phan_loai['height'] = 400

khung_ket_qua_phan_loai.columnconfigure(0, weight=1)
khung_ket_qua_phan_loai.rowconfigure(1, weight=1)

# Tạo một nhãn để hiển thị kết quả phân loại
nhan_ket_qua = ttk.Label(khung_ket_qua_phan_loai, text="Kết quả", font=("Helvetica", 13, "bold"))
nhan_ket_qua.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

# Tạo một Text widget để hiển thị thông tin phân loại
thong_tin_phan_loai = tk.Text(khung_ket_qua_phan_loai, wrap=tk.WORD, height=6, width=30, font=("Helvetica", 13, "bold"))
thong_tin_phan_loai.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

# Tạo một khung cho biểu đồ phân loại
khung_bieu_do_phan_loai = ttk.LabelFrame(root, text="Biểu đồ phân loại", padding=10)
khung_bieu_do_phan_loai.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
khung_bieu_do_phan_loai['width'] = 300
khung_bieu_do_phan_loai['height'] = 400

canvas_bieu_do = FigureCanvasTkAgg(plt.figure(figsize=(6, 6)), master=khung_bieu_do_phan_loai)
canvas_bieu_do.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)

# Thêm thanh cuộn cho Text widget
thanh_cuon = tk.Scrollbar(khung_ket_qua_phan_loai, orient="vertical", command=thong_tin_phan_loai.yview)
thanh_cuon.grid(row=1, column=1, sticky="nse")
thong_tin_phan_loai.configure(yscrollcommand=thanh_cuon.set)

# Cờ để cập nhật thống kê từ thư mục
co_cap_nhat_thong_ke = True

# Tạo khung thống kê
khung_thong_ke = ttk.LabelFrame(root, text="Thống kê phân loại", padding=10)
khung_thong_ke.grid(row=0, column=2, rowspan=2, padx=10, pady=10, sticky="nsew")
khung_thong_ke['width'] = 300
khung_thong_ke['height'] = 400

listbox_thong_ke = tk.Listbox(khung_thong_ke, height=10, font=("Helvetica", 13, "bold"))
listbox_thong_ke.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

# Thanh cuộn cho danh sách thống kê
thanh_cuon = tk.Scrollbar(khung_thong_ke, orient="vertical", command=listbox_thong_ke.yview)
thanh_cuon.grid(row=1, column=1, sticky="nse")
listbox_thong_ke.configure(yscrollcommand=thanh_cuon.set)

thong_ke = []

# Nút để cập nhật bảng thống kê
nut_cap_nhat = ttk.Button(khung_thong_ke, text="Cập nhật bảng thống kê", command=cap_nhat_thong_ke_tu_thu_muc, style="NutTuyChinh.TButton")
nut_cap_nhat.grid(row=2, column=0, padx=5, pady=5, sticky="nsew")

# Khởi tạo thống kê ban đầu
thong_ke_so_luong_dong_vat = cap_nhat_thong_ke_tu_cac_thu_muc_con(thu_muc_luu_tru)
cap_nhat_thong_ke()

# Nút để xóa dữ liệu thống kê và các tệp trong thư mục con
nut_xoa_tep = ttk.Button(khung_thong_ke, text="Xoá bảng thống kê và các tệp", command=xoa_thong_ke_va_cac_tep, style="NutTuyChinh.TButton")
nut_xoa_tep.grid(row=3, column=0, padx=5, pady=5, sticky="nsew")

# Khung thông tin liên quan
khung_thong_tin = ttk.LabelFrame(root, text="Thông tin liên quan", padding=10)
khung_thong_tin.grid(row=2, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

nhan_sinh_vien = ttk.Label(khung_thong_tin, text="Sinh viên thực hiện:", font=("Helvetica", 13, "bold"))
nhan_sinh_vien.grid(row=0, column=0, padx=5, pady=5, sticky="e")

thong_tin_sinh_vien = ttk.Label(khung_thong_tin, text="", font=("Helvetica", 13, "bold"))
thong_tin_sinh_vien.grid(row=0, column=1, padx=5, pady=5, sticky="w")

nhan_giao_vien = ttk.Label(khung_thong_tin, text="Giáo viên hướng dẫn:", font=("Helvetica", 13, "bold"))
nhan_giao_vien.grid(row=1, column=0, padx=5, pady=5, sticky="e")

thong_tin_giao_vien = ttk.Label(khung_thong_tin, text="", font=("Helvetica", 13, "bold"))
thong_tin_giao_vien.grid(row=1, column=1, padx=5, pady=5, sticky="w")

nhan_de_tai = ttk.Label(khung_thong_tin, text="Tên đề tài:", font=("Helvetica", 13, "bold"))
nhan_de_tai.grid(row=2, column=0, padx=5, pady=5, sticky="e")

thong_tin_de_tai = ttk.Label(khung_thong_tin, text="", font=("Helvetica", 13, "bold"))
thong_tin_de_tai.grid(row=2, column=1, padx=5, pady=5, sticky="w")

# Gọi hàm để cài đặt thông tin liên quan
cai_dat_thong_tin_lien_quan()

# Chạy vòng lặp chính của Tkinter
root.mainloop()