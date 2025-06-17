import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# Step 1: Data Simulation (unchanged)
np.random.seed(42)
num_samples = 10000
features = 10

normal_data = np.random.normal(loc=0, scale=1, size=(num_samples, features))
anomalous_data = np.random.normal(loc=5, scale=1, size=(num_samples // 10, features))

data = np.vstack((normal_data, anomalous_data))
labels = np.hstack((np.zeros(num_samples), np.ones(num_samples // 10)))

df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(features)])
df['label'] = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('label', axis=1), df['label'], test_size=0.2, random_state=42
)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Autoencoder Model
input_dim = X_train_scaled.shape[1]
encoding_dim = 6  # slightly more capacity

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
decoded = Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Increased epochs for better convergence
autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=100, batch_size=32, validation_split=0.1, verbose=0)

# Step 3: Anomaly Detection
reconstructions = autoencoder.predict(X_test_scaled, verbose=0)
mse = np.mean(np.power(X_test_scaled - reconstructions, 2), axis=1)

threshold = np.percentile(mse, 92)
y_pred = (mse > threshold).astype(int)

# Step 4: Evaluation
conf_mat = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_mat.ravel()

# Class-specific metrics
precision_attack = tp / (tp + fp)
recall_attack = tp / (tp + fn)
precision_normal = tn / (tn + fn)
recall_normal = tn / (tn + fp)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision (Normal): {precision_normal * 100:.2f}%")
print(f"Recall (Normal): {recall_normal * 100:.2f}%")
print(f"Precision (Attack): {precision_attack * 100:.2f}%")
print(f"Recall (Attack): {recall_attack * 100:.2f}%")

# Step 5: Real-Time Kafka - Optional (commented out unless Kafka is set up)

# from kafka import KafkaProducer, KafkaConsumer
"""
# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
topic = 'iot_data'

for index, row in df.iterrows():
    producer.send(topic, row.to_dict())
    time.sleep(0.01)

# Kafka Consumer
consumer = KafkaConsumer(
    topic,
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for message in consumer:
    try:
        incoming = pd.DataFrame([message.value])
        scaled = scaler.transform(incoming.drop('label', axis=1))
        rec = autoencoder.predict(scaled)
        mse = np.mean(np.power(scaled - rec, 2), axis=1)

        if mse > threshold:
            print(f"Anomaly detected: {message.value}")
    except Exception as e:
        print("Error:", e)
"""
