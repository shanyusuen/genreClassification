import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import seaborn as sns

def plotHistory(test_accuracy, training_accuracy, test_loss, training_loss):

    yellow  = "#f4d03f"
    blue = "#5dade2"

    fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(7,7))

    # Accuracy

    ax0.plot(range(1, 1 + len(training_accuracy)), training_accuracy, color=blue)
    ax0.set_ylabel('Accuracy')
    ax0.set_title("Training Data")

    ax1.plot(range(1, 1 + len(test_accuracy)), test_accuracy, color=blue)
    ax1.set_xlabel('Epochs')
    ax1.set_title("Testing Data")

    # Loss

    ax2.plot(range(1, 1 + len(training_loss)), training_loss, color=yellow)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')

    ax3.plot(range(1, 1 + len(test_loss)), test_loss, color=yellow)
    ax3.set_xlabel('Epochs')

    plt.tight_layout()
    plt.show()


def plotConfusion(model, classes, test_x, test_y):
    # test labels can not be one hot encoded, assign indexes
    #test_y = np.argmax(test_y, 1)
    test_y = np.argmax(test_y, axis=1)
    pred_y = model.predict_classes(test_x)

    con_mat = tf.math.confusion_matrix(labels=test_y, predictions=pred_y).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

    """
    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=classes,
                              columns=classes)
    """
    figure = plt.figure(figsize=(8, 8))
    #sns.heatmap(con_mat, xticklabels=classes, yticklabels=classes, annot=True, fmt='g', cmap=plt.cm.Blues)
    sns.heatmap(con_mat_norm, xticklabels=classes, yticklabels=classes, annot=True, fmt='g', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


    plt.tight_layout()
    plt.show()