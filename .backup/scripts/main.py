#!/usr/bin/env python3
import serial
import os
import time
import psutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from nnclassifier import ModelClassifier

def main():
    # Load model with dataset path
    model = ModelClassifier(dataset_path='~/catkin_ws/src/kopi/scripts/database.csv')

    # Configure serial communication
    try:
        ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return

    try:
        while True:
            # Receive input from serial
            if ser.in_waiting > 0:
                input_str = ser.readline().decode('utf-8').strip()
                print(f"Received: {input_str}")

                # Handle 'create' and 'delete' commands
                if input_str in ['create', 'delete']:
                    ser.write(b'oke\n')

                # Handle 'compare' command
                elif input_str == 'compare':
                    ser.write(b'oke\n')
                    print("Waiting for data to compare...")

                    # Wait for the next data from Arduino
                    while True:
                        if ser.in_waiting > 0:
                            data_str = ser.readline().decode('utf-8').strip()
                            print(f"Input data: {data_str}")

                            try:
                                # Predict using the model
                                prediction, accuracy, time_taken, memory_used = model.predict_class(data_str)
                                print("Predicted class: {}".format(prediction))
                                print("Model Accuracy   : {:.2f}%".format(accuracy * 100))
                                print("Prediction Time  : {:.6f} seconds".format(time_taken))
                                print("Memory Used      : {:.2f} KB".format(memory_used))

                                # Send result to Arduino
                                ser.write(f'{prediction}\n'.encode('utf-8'))
                            except Exception as e:
                                print(f"Error during prediction: {e}")
                                ser.write(b'error\n')
                            break  # Exit inner loop after processing

                # Ignore other commands
                else:
                    pass  # Do not send "Unknown command"

    except KeyboardInterrupt:
        print("Program terminated by user.")
    finally:
        ser.close()

if __name__ == "__main__":
    main()

