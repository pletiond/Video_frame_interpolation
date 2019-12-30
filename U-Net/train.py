from keras.optimizers import Adam
from keras.models import model_from_json
from dataset import get_dataset
from unet import *
import matplotlib.pyplot as plt

def train_model():
    dataset_size = X_train.shape[0]
    batches = dataset_size // batch_size

    for i in range(1, epochs + 1):
        for batch in range(batches):
            if batch % 5 == 0:
                print(f'Batch: {batch + 1} / {batches}')
            first = batch * batch_size
            last = (batch + 1) * batch_size
            res_loss = model.train_on_batch(X_train[first:last], Y_train[first:last])
            print(res_loss)
        res = model.evaluate(X_val, Y_val)
        print(f'\nEpoch {i} - {res}')


def load_model():
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


def evaluate():
    y_pred = model.predict(X_all)
    print(y_pred[0])
    for i in range(len(y_pred)):
        plt.imsave("./results/frame" + str(i+1) + "_pred.png",
                   (y_pred[i] * 255).astype("uint8"))

#--------------------------------------------------

dataset = './videos/gump_size.mp4'

X_train, Y_train, X_val, Y_val, X_all = get_dataset(dataset)

batch_size = 4
epochs = 10
do_evaluate = False

if do_evaluate:
    model = load_model()
else:
    model = get_unet(input_shape=(None, None, 6))

model.compile(loss='binary_crossentropy', optimizer=Adam())

if do_evaluate:
    evaluate()
else:
    train_model()
    evaluate()
    model_json = model.to_json()

    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")
    print("Saved model to disk")