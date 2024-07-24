from DataLoader import DataLoader
from ModelBuilder import ModelBuilder
from Trainer import Trainer
from Evaluator import Evaluator


class Main:
    def __init__(self, train_Dir, test_Dir):
        self.train_dir = train_Dir
        self.test_dir = test_Dir
        self.data_loader = DataLoader(train_dir, test_Dir)
        self.model_builder = ModelBuilder()

    def run(self):
        train_generator, val_generator = self.data_loader.get_data_generators()
        model = self.model_builder.build_model()
        trainer = Trainer(model, train_generator, val_generator)
        trainer.compile_model()
        trainer.train_model()

        model.save("model.keras")
        print("Saved model to disk")

        test_generator = self.data_loader.get_test_generator()
        evaluator = Evaluator(model, test_generator, self.data_loader.label_names)
        evaluator.evaluate_model()
        evaluator.predict_and_evaluate()


if __name__ == "__main__":
    train_dir = 'Chest_X_Ray/train'
    test_dir = 'Chest_X_Ray/Test'
    main = Main(train_dir, test_dir)
    main.run()
