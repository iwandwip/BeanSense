import inquirer
import traceback
from ClassifierWrapper import ClassifierWrapper


class MainMenu:
    def __init__(self):
        self.wrapper = ClassifierWrapper()

    def select_dataset(self):
        dataset_questions = [
            inquirer.List('dataset',
                          message="Select dataset to use:",
                          choices=[
                              ('dataset4.csv (4 sensors)', '4'),
                              ('dataset6.csv (6 sensors)', '6'),
                              ('dataset8.csv (8 sensors)', '8'),
                              ('Exit', 'exit')
                          ],
                          )
        ]
        dataset_answer = inquirer.prompt(dataset_questions)
        dataset_choice = dataset_answer['dataset']

        if dataset_choice == "exit":
            return "exit"

        csv_file = self.wrapper.set_dataset(dataset_choice)
        print(f"Using dataset: {csv_file}")
        return dataset_choice

    def select_operation(self):
        operation_questions = [
            inquirer.List('operation',
                          message="Select operation:",
                          choices=[
                              ('Train model', 'train'),
                              ('Predict with model', 'predict'),
                              ('Add dataset', 'add_data'),
                              ('Pop dataset', 'pop_data'),
                              ('Delete dataset', 'delete_data'),
                              ('Back', 'back')
                          ],
                          )
        ]
        operation_answer = inquirer.prompt(operation_questions)
        operation = operation_answer['operation']

        return operation

    def select_model_type(self, operation):
        model_choices = []
        for model_key in self.wrapper.get_available_models():
            if model_key == 'adaboost_resnet':
                display_name = 'AdaBoost + ResNet'
            elif model_key == 'catboost_resnet':
                display_name = 'CatBoost + ResNet'
            elif model_key == 'lightgbm_resnet':
                display_name = 'LightGBM + ResNet'
            elif model_key == 'lightgbm_mobilenet':
                display_name = 'LightGBM + MobileNet'
            elif model_key == 'mobilenet_iccs_lightgbm':
                display_name = 'MobileNet + ICCS + LightGBM'
            elif model_key == 'autoencoder_lightgbm':
                display_name = 'Autoencoder + LightGBM'
            elif model_key == 'rbf_svm_gs':
                display_name = 'RBF SVM + GridSearch'
            else:
                display_name = model_key

            model_choices.append((display_name, model_key))

        model_choices.append(('Back', 'back'))

        model_questions = [
            inquirer.List('model_type',
                          message=f"Select model type for {operation}:",
                          choices=model_choices,
                          )
        ]
        model_answer = inquirer.prompt(model_questions)
        model_type = model_answer['model_type']

        if model_type == 'back':
            return 'back'

        if model_type == 'catboost_resnet':
            gpu_questions = [
                inquirer.Confirm('use_gpu',
                                 message="Use GPU for training (if available)?",
                                 default=False)
            ]
            gpu_answer = inquirer.prompt(gpu_questions)
            use_gpu = gpu_answer['use_gpu']
            self.wrapper.create_model(model_type, use_gpu)
        else:
            self.wrapper.create_model(model_type)

        return model_type

    def select_label(self):
        label_choices = [(label, label) for label in self.wrapper.get_labels()]

        label_questions = [
            inquirer.List('label',
                          message="Select label:",
                          choices=label_choices,
                          )
        ]
        label_answer = inquirer.prompt(label_questions)
        selected_label = label_answer['label']

        return selected_label

    def handle_training(self):
        model_type = self.select_model_type('training')

        if model_type == 'back':
            return

        if model_type == 'mobilenet_iccs_lightgbm':
            cache_questions = [
                inquirer.Confirm('use_cached',
                                 message="Use cached features if available?",
                                 default=True)
            ]
            cache_answer = inquirer.prompt(cache_questions)
            use_cached = cache_answer['use_cached']

            self.wrapper.train_model(use_cached_features=use_cached)
        elif model_type == 'autoencoder_lightgbm':
            plot_questions = [
                inquirer.Confirm('show_plots',
                                 message="Show evaluation plots?",
                                 default=False)
            ]
            plot_answer = inquirer.prompt(plot_questions)
            show_plots = plot_answer['show_plots']

            self.wrapper.train_model(show_plots=show_plots)
        else:
            self.wrapper.train_model()

        print("\n" + "-"*50 + "\n")

    def handle_prediction(self):
        model_type = self.select_model_type('prediction')

        if model_type == 'back':
            return

        prediction_loop = True
        while prediction_loop:
            input_questions = [
                inquirer.Text('sensor_values',
                              message="Enter sensor values (comma separated) or 'q' to quit:")
            ]
            input_answer = inquirer.prompt(input_questions)
            input_string = input_answer['sensor_values']

            result = self.wrapper.predict_with_model(input_string)

            if result == 'quit':
                break

            if result is None:
                prediction_loop = False
                continue

            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Prediction result: {result['predicted_class']}")
                print("Class probabilities:")
                for cls, prob in result['probabilities'].items():
                    print(f"  {cls}: {prob:.4f}")

            continue_questions = [
                inquirer.Confirm('continue',
                                 message="Would you like to make another prediction?",
                                 default=True)
            ]
            continue_answer = inquirer.prompt(continue_questions)
            if not continue_answer['continue']:
                prediction_loop = False

        print("\n" + "-"*50 + "\n")

    def handle_add_data(self):
        selected_label = self.select_label()

        num_features = int(self.wrapper.dataset_choice)

        print(f"Enter {num_features} sensor values (comma separated):")
        sensor_values = input().strip()

        self.wrapper.add_dataset_entry(selected_label, sensor_values)

        print("\n" + "-"*50 + "\n")

    def handle_pop_data(self):
        self.wrapper.pop_dataset_entry()

        print("\n" + "-"*50 + "\n")

    def handle_delete_data(self):
        selected_label = self.select_label()

        count = self.wrapper.delete_dataset_entries(selected_label)

        if not count:
            print("\n" + "-"*50 + "\n")
            return

        confirm_questions = [
            inquirer.Confirm('confirm',
                             message=f"Are you sure you want to delete all {count} data points with label '{selected_label}'?",
                             default=False)
        ]
        confirm_answer = inquirer.prompt(confirm_questions)

        if not confirm_answer['confirm']:
            print("Operation cancelled.")
            print("\n" + "-"*50 + "\n")
            return

        self.wrapper.delete_dataset_entries(selected_label, confirm=True)

        print("\n" + "-"*50 + "\n")

    def run(self):
        while True:
            dataset_result = self.select_dataset()

            if dataset_result == "exit":
                break

            while True:
                operation = self.select_operation()

                if operation == "back":
                    break

                if operation == "train":
                    self.handle_training()
                elif operation == "predict":
                    self.handle_prediction()
                elif operation == "add_data":
                    self.handle_add_data()
                elif operation == "pop_data":
                    self.handle_pop_data()
                elif operation == "delete_data":
                    self.handle_delete_data()


def main():
    try:
        menu = MainMenu()
        menu.run()
        print("Exiting program...")
    except Exception as e:
        print(f"Error occurred: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
