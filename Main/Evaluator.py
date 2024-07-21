import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


class Evaluator:
    def __init__(self, model, test_generator, label_names):
        self.model = model
        self.test_generator = test_generator
        self.label_names = label_names

    def evaluate_model(self):
        evaluate = self.model.evaluate(self.test_generator,
                                       steps=self.test_generator.n // self.test_generator.batch_size, verbose=1)
        print('Test Accuracy: {}'.format(evaluate[1]))
        return evaluate

    def predict_and_evaluate(self):
        prediction = []
        original = []
        images = []
        for i in range(len(os.listdir(self.test_generator.directory))):
            for item in os.listdir(os.path.join(self.test_generator.directory, str(i))):
                img = cv2.imread(os.path.join(self.test_generator.directory, str(i), item))
                img = cv2.resize(img, (256, 256))
                images.append(img)
                img = img / 255
                img = img.reshape(-1, 256, 256, 3)
                predict = self.model.predict(img)
                predict = np.argmax(predict)
                prediction.append(predict)
                original.append(i)

        score = accuracy_score(original, prediction)
        print("Test Accuracy: {}".format(score))

        fig, axes = plt.subplots(5, 5, figsize=(12, 12))
        axes = axes.ravel()
        for i in np.arange(0, 25):
            axes[i].imshow(images[i])
            axes[i].set_title(
                'Guess={}\nTrue={}'.format(str(self.label_names[prediction[i]]), str(self.label_names[original[i]])))
            axes[i].axis('off')
        plt.subplots_adjust(wspace=1.2)

        print(classification_report(np.asarray(original), np.asarray(prediction)))

        cm = confusion_matrix(np.asarray(original), np.asarray(prediction))
        ax = plt.subplot()
        sns.heatmap(cm, annot=True, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Original')
        ax.set_title('Confusion Matrix')
        plt.show()
