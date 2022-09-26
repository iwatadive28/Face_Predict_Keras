from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential

def model():
    # CNNで学習するときの画像のサイズを設定（サイズが大きいと学習に時間がかかる）
    # ImgSize = (180,180)
    ImgSize = (180, 180, 3)

    # モデルの定義
    model = Sequential()
    model.add(Conv2D(input_shape=ImgSize, filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("sigmoid"))
    model.add(Dense(128))
    model.add(Activation('sigmoid'))

    # 分類したい人数を入れる
    model.add(Dense(len(classes)))
    model.add(Activation('softmax'))

    # コンパイル
    print("Compilingmodel...")
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    print("done")
    model.summary()

    return model

if __name__ == '__main__':
    model = model()


