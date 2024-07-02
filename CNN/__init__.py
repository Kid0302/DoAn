from tensorflow import keras
import pickle

try:
    model = keras.models.load_model('animals.keras')
    lb = pickle.loads(open('lb.pickle', "rb").read())
except Exception as e:
    print(f"Lỗi khi tải mô hình hoặc nhãn: {str(e)}")