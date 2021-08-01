import numpy as np
from os import walk
import click
import tensorflow as tf
from tensorflow import keras
from Metrics import recall_m, precision_m, f1_m
from datagenerator import DataGenerator
from FCN import FCNClassifier


@click.command()
@click.option('--saved_model', '-m', help='The model directory path', type=click.Path(exists=True))
@click.option('--test_set', '-t', help='The test dataset directory path', type=click.Path(exists=True))
@click.option('--batch', '-b', default=1, help='Batch size')
def main(saved_model, batch, test_set):

    _, _, test_files = next(walk(test_set))

    test_data = np.load(test_set + test_files[1])

    batch = test_data.shape[0]
    n_steps = test_data.shape[1]-1
    n_features = 1
    in_shape = (n_steps, n_features)
    print(in_shape)

    '''evaluate the model'''
    model = keras.models.load_model(saved_model,  custom_objects={'recall_m': recall_m, 'precision_m': precision_m, 'f1_m': f1_m})
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy', recall_m, precision_m, f1_m])

    inference_model = FCNClassifier(input_shape=in_shape, nb_classes=1, is_train=False).model
    inference_model.set_weights(model.get_weights())

    test_generator = DataGenerator(test_files, test_set, batch_size=batch, dim=in_shape, is_train=False)
    scores = inference_model.evaluate(test_generator, verbose=1)

    for i in range(len(inference_model.metrics)):
        print("%s: %.2f%%" % (inference_model.metrics_names[i], scores[i]*100))

    # test_data = np.load(test_set + test_files[1])
    # for data in test_data:
    #     test_x = data[:-1]
    #     test_y = data[-1]
    #     test_x = test_x.reshape(1, n_steps, n_features)
    #     predicted_y = model.predict(test_x)
    #     rounded_predicted_y = 0
    #     if predicted_y > 0.5:
    #         rounded_predicted_y = 1.0
    #     else:
    #         rounded_predicted_y = 0.0
    #     if rounded_predicted_y != test_y:
    #         print("predicted: ", rounded_predicted_y, "Actual: ", test_y, "Confidence: ", predicted_y[0][0]*100)


if __name__ == '__main__':
    main()
