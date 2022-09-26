from model import model
from helper import *
import glob
import time
import numpy as np
from keras.models import load_model

# 事前に設定するパラメータ
model = load_model('models/vgg16_model.hdf5') # load
model.load_weights("weights/weights.best.hdf5")
classes = ['Other', 'Joy', 'Harry', 'Uentsu', 'Raul', 'Yuji']

# CNNで学習するときの画像のサイズを設定（サイズが大きいと学習に時間がかかる）
ImgSize = (180, 180)
input_shape = (180, 180, 3)

# 画像を読み込んで予測する
def img_predict(img):
    # 画像を読み込んで4次元テンソルへ変換
    # normalized
    x = np.uint8(normalized(img))
    x = x[np.newaxis, :, :, :]

    # 指数表記を禁止にする
    np.set_printoptions(suppress=True)

    # 画像の人物を予測
    pred = model.predict(x)[0]

    # 結果を表示する
    print(classes[np.argmax(pred)] + ':{:.2f}'.format(pred[np.argmax(pred)] * 100) + '%')
    print(classes)
    print(pred * 100)

    return pred

if __name__ == '__main__':

    Human = glob.glob('data/*')
    ih = np.random.randint(len(Human))
    test = glob.glob(Human[ih]+'/*')
    i = np.random.randint(len(test))

    orgimg = cv2.imread(test[i])
    cv2.imshow("", orgimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 表示
    start_time = time.time()
    img_predict(orgimg )
    end_time = time.time()
    print('Erapsed time: ', (end_time - start_time), '[s]')
