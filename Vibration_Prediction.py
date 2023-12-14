import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset from the CSV file
dataset = pd.read_csv('predictive-maintenance-dataset.csv')

# Extract the input features (X) and target variable (y)
X = dataset[['revolutions', 'humidity', 'x1', 'x2', 'x3', 'x4', 'x5']].values
y = dataset['vibration'].values

# Split the dataset into training and testing sets (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the MLP model
model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=X_train.shape[1:]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

# Train the model and collect the training history
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=15, verbose=1, validation_data=(X_test_scaled, y_test))

# Evaluate the model on the testing set
mse = model.evaluate(X_test_scaled, y_test, verbose=1)
print('Mean Squared Error:', mse)

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Write MAE values for every epoch to a text file
with open('mae_values.txt', 'w') as f:
    f.write("Epoch\tMAE\tValidation MAE\n")
    for epoch in range(len(history.history['mae'])):
        mae = history.history['mae'][epoch]
        val_mae = history.history['val_mae'][epoch]
        f.write(f"{epoch+1}\t{mae}\t{val_mae}\n")

# Write loss values to a text file
with open('loss_values.txt', 'w') as f:
    f.write("Epoch\tLoss\tValidation Loss\n")
    for epoch in range(len(history.history['loss'])):
        loss = history.history['loss'][epoch]
        val_loss = history.history['val_loss'][epoch]
        f.write(f"{epoch+1}\t{loss}\t{val_loss}\n")

# Plot the training and validation mae
fig_acc = plt.figure()
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("model_mae.png")

# Plot the training and validation loss
fig_acc = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss','Validation Loss'], loc='upper left')
plt.show()
fig_acc.savefig("model_regression_loss.png")

# Plot the predicted data and actual data
plt.figure(figsize=(10, 15))
plt.plot(y_pred, color="blue")
plt.plot(y_test, color="green")
plt.title('prediction')
plt.ylabel('value')
plt.xlabel('row')
plt.legend(['predicted', 'actual data'], loc='upper left')
plt.show()