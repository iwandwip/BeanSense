import inquirer
import time
import os
import pandas as pd
from ClassifierWrapper import ClassifierWrapper


def train_all_models():
    wrapper = ClassifierWrapper()

    datasets = ["4", "6", "8"]
    models = [
        "adaboost_resnet",
        "catboost_resnet",
        "lightgbm_resnet",
        "lightgbm_mobilenet",
        "mobilenet_iccs_lightgbm",
        "autoencoder_lightgbm",
        "rbf_svm_gs"
    ]

    results = []

    total_combinations = len(datasets) * len(models)
    current = 0

    for dataset in datasets:
        csv_file = wrapper.set_dataset(dataset)
        print(f"\n=== Dataset: {csv_file} ===\n")

        for model_type in models:
            current += 1
            print(f"\n[{current}/{total_combinations}] Training {model_type} on dataset{dataset}")

            start_time = time.time()

            try:
                model = wrapper.create_model(model_type)
                metrics = wrapper.train_model(use_cached_features=True)

                end_time = time.time()
                training_time = end_time - start_time

                if metrics:
                    results.append({
                        "dataset": dataset,
                        "model": model_type,
                        "accuracy": metrics.get("avg_accuracy", 0),
                        "f1_score": metrics.get("avg_f1_score", 0),
                        "auc": metrics.get("avg_auc", 0),
                        "memory_used_mb": metrics.get("memory_used", 0),
                        "peak_memory_mb": metrics.get("peak_memory", 0),
                        "training_time_s": training_time,
                        "training_time_min": training_time / 60,
                        "status": "success"
                    })
                else:
                    results.append({
                        "dataset": dataset,
                        "model": model_type,
                        "status": "failed",
                        "training_time_s": training_time
                    })
            except Exception as e:
                end_time = time.time()
                training_time = end_time - start_time

                print(f"Error training {model_type} on dataset{dataset}: {str(e)}")
                results.append({
                    "dataset": dataset,
                    "model": model_type,
                    "status": f"error: {str(e)}",
                    "training_time_s": training_time
                })

    save_results(results)
    print_summary(results)
    return results


def save_results(results):
    os.makedirs("results", exist_ok=True)

    df = pd.DataFrame(results)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_path = f"results/training_results_{timestamp}.csv"
    df.to_csv(file_path, index=False)

    print(f"\nResults saved to {file_path}")


def print_summary(results):
    print("\n=== TRAINING SUMMARY ===\n")

    success_count = sum(1 for r in results if r["status"] == "success")
    fail_count = len(results) - success_count

    print(f"Total combinations trained: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

    if success_count > 0:
        successful_results = [r for r in results if r["status"] == "success"]

        best_accuracy = max(successful_results, key=lambda x: x["accuracy"])
        best_f1 = max(successful_results, key=lambda x: x["f1_score"])
        best_auc = max(successful_results, key=lambda x: x["auc"])
        fastest = min(successful_results, key=lambda x: x["training_time_s"])
        lowest_memory = min(successful_results, key=lambda x: x["memory_used_mb"])

        print("\nBest performers:")
        print(f"Best accuracy: {best_accuracy['model']} on dataset{best_accuracy['dataset']} - {best_accuracy['accuracy']:.4f}")
        print(f"Best F1-score: {best_f1['model']} on dataset{best_f1['dataset']} - {best_f1['f1_score']:.4f}")
        print(f"Best AUC: {best_auc['model']} on dataset{best_auc['dataset']} - {best_auc['auc']:.4f}")
        print(f"Fastest training: {fastest['model']} on dataset{fastest['dataset']} - {fastest['training_time_s']:.2f}s")
        print(f"Lowest memory: {lowest_memory['model']} on dataset{lowest_memory['dataset']} - {lowest_memory['memory_used_mb']:.2f} MB")


def make_prediction():
    wrapper = ClassifierWrapper()

    model_questions = [
        inquirer.List('dataset',
                      message="Select dataset type:",
                      choices=[
                          ('4 sensors', '4'),
                          ('6 sensors', '6'),
                          ('8 sensors', '8')
                      ]),
        inquirer.List('model',
                      message="Select model type:",
                      choices=[
                          ('AdaBoost + ResNet', 'adaboost_resnet'),
                          ('CatBoost + ResNet', 'catboost_resnet'),
                          ('LightGBM + ResNet', 'lightgbm_resnet'),
                          ('LightGBM + MobileNet', 'lightgbm_mobilenet'),
                          ('MobileNet + ICCS + LightGBM', 'mobilenet_iccs_lightgbm'),
                          ('Autoencoder + LightGBM', 'autoencoder_lightgbm'),
                          ('RBF SVM + GridSearch', 'rbf_svm_gs')
                      ])
    ]

    answers = inquirer.prompt(model_questions)

    wrapper.set_dataset(answers['dataset'])
    wrapper.create_model(answers['model'])

    prediction_loop = True
    while prediction_loop:
        input_questions = [
            inquirer.Text('sensor_values',
                          message="Enter sensor values (comma separated) or 'q' to quit:")
        ]
        input_answer = inquirer.prompt(input_questions)
        input_string = input_answer['sensor_values']

        if input_string.lower() == 'q':
            break

        result = wrapper.predict_with_model(input_string)

        if isinstance(result, dict):
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nPrediction result: {result['label']}")
                print(f"Accuracy: {result['accuracy']:.4f}")
                print(f"Execution time: {result['execution_time']:.4f}s")
                print(f"Memory used: {result['memory_used']:.2f} MB")

        continue_questions = [
            inquirer.Confirm('continue',
                             message="Make another prediction?",
                             default=True)
        ]
        continue_answer = inquirer.prompt(continue_questions)
        if not continue_answer['continue']:
            prediction_loop = False


def manage_dataset():
    wrapper = ClassifierWrapper()

    dataset_questions = [
        inquirer.List('dataset',
                      message="Select dataset to manage:",
                      choices=[
                          ('dataset4.csv (4 sensors)', '4'),
                          ('dataset6.csv (6 sensors)', '6'),
                          ('dataset8.csv (8 sensors)', '8')
                      ])
    ]

    dataset_answer = inquirer.prompt(dataset_questions)
    dataset_choice = dataset_answer['dataset']

    csv_file = wrapper.set_dataset(dataset_choice)

    while True:
        action_questions = [
            inquirer.List('action',
                          message=f"Select action for {csv_file}:",
                          choices=[
                              ('Add data entry', 'add'),
                              ('Remove last entry for a label', 'pop'),
                              ('Delete all entries for a label', 'delete'),
                              ('Return to main menu', 'exit')
                          ])
        ]

        action_answer = inquirer.prompt(action_questions)
        action = action_answer['action']

        if action == 'exit':
            break

        label_questions = [
            inquirer.List('label',
                          message="Select coffee label:",
                          choices=wrapper.get_labels())
        ]

        label_answer = inquirer.prompt(label_questions)
        selected_label = label_answer['label']

        if action == 'add':
            sensor_questions = [
                inquirer.Text('values',
                              message=f"Enter {dataset_choice} sensor values (comma separated):")
            ]

            sensor_answer = inquirer.prompt(sensor_questions)
            sensor_values = sensor_answer['values']

            result = wrapper.add_dataset_entry(selected_label, sensor_values)
            if result:
                print(f"Successfully added entry with label '{selected_label}'")

        elif action == 'pop':
            result = wrapper.pop_dataset_entry(selected_label)
            if result:
                print(f"Successfully removed last entry with label '{selected_label}'")
            else:
                print(f"No entries found with label '{selected_label}'")

        elif action == 'delete':
            count = wrapper.delete_dataset_entries(selected_label, confirm=False)

            if count:
                confirm_questions = [
                    inquirer.Confirm('confirm',
                                     message=f"Are you sure you want to delete all {count} entries with label '{selected_label}'?",
                                     default=False)
                ]

                confirm_answer = inquirer.prompt(confirm_questions)
                if confirm_answer['confirm']:
                    result = wrapper.delete_dataset_entries(selected_label, confirm=True)
                    if result:
                        print(f"Successfully deleted {count} entries with label '{selected_label}'")
            else:
                print(f"No entries found with label '{selected_label}'")


def main():
    try:
        while True:
            questions = [
                inquirer.List('action',
                              message="Select action:",
                              choices=[
                                  ('Train all models on all datasets', 'train_all'),
                                  ('Train single model', 'train'),
                                  ('Make prediction', 'predict'),
                                  ('Manage dataset', 'manage'),
                                  ('Exit', 'exit')
                              ])
            ]

            answers = inquirer.prompt(questions)
            action = answers['action']

            if action == 'exit':
                print("Exiting program...")
                break

            elif action == 'train_all':
                print("\nTraining all models on all datasets...")
                train_all_models()

            elif action == 'train':
                wrapper = ClassifierWrapper()

                dataset_questions = [
                    inquirer.List('dataset',
                                  message="Select dataset to use:",
                                  choices=[
                                      ('dataset4.csv (4 sensors)', '4'),
                                      ('dataset6.csv (6 sensors)', '6'),
                                      ('dataset8.csv (8 sensors)', '8')
                                  ])
                ]

                dataset_answer = inquirer.prompt(dataset_questions)
                dataset_choice = dataset_answer['dataset']

                model_questions = [
                    inquirer.List('model',
                                  message="Select model type:",
                                  choices=[
                                      ('AdaBoost + ResNet', 'adaboost_resnet'),
                                      ('CatBoost + ResNet', 'catboost_resnet'),
                                      ('LightGBM + ResNet', 'lightgbm_resnet'),
                                      ('LightGBM + MobileNet', 'lightgbm_mobilenet'),
                                      ('MobileNet + ICCS + LightGBM', 'mobilenet_iccs_lightgbm'),
                                      ('Autoencoder + LightGBM', 'autoencoder_lightgbm'),
                                      ('RBF SVM + GridSearch', 'rbf_svm_gs')
                                  ])
                ]

                model_answer = inquirer.prompt(model_questions)
                model_type = model_answer['model']

                wrapper.set_dataset(dataset_choice)
                wrapper.create_model(model_type)

                cache_questions = [
                    inquirer.Confirm('use_cached',
                                     message="Use cached features if available?",
                                     default=True)
                ]

                cache_answer = inquirer.prompt(cache_questions)
                use_cached = cache_answer['use_cached']

                print(f"Training {model_type} on dataset{dataset_choice}...")
                metrics = wrapper.train_model(use_cached_features=use_cached)

                if metrics:
                    print("\nTraining Results:")
                    print(f"Average Accuracy: {metrics['avg_accuracy']:.4f}")
                    print(f"Average F1-Score: {metrics['avg_f1_score']:.4f}")
                    print(f"Average AUC: {metrics['avg_auc']:.4f}")
                    print(f"Memory Used: {metrics['memory_used']:.2f} MB")
                    print(f"Training Time: {metrics['execution_time']:.2f}s")

            elif action == 'predict':
                make_prediction()

            elif action == 'manage':
                manage_dataset()

            print("\n" + "-"*50 + "\n")

    except Exception as e:
        import traceback
        print(f"Error occurred: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
