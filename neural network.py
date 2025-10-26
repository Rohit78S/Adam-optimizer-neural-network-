import numpy as np
import pandas as pd

print("Generating 1000 sample dataset...")

np.random.seed(42)
dataset = []

# Generate 650 PASS samples (high scores 70-100)
for i in range(650):
    score = np.random.uniform(70, 100)
    hours = np.random.uniform(1, 20)
    label_fail = 0
    label_pass = 1
    dataset.append([score, hours, label_fail, label_pass])

# Generate 350 FAIL samples
# 200 with medium scores (40-60) + various hours
for i in range(200):
    score = np.random.uniform(40, 60)
    hours = np.random.uniform(1, 20)
    label_fail = 1
    label_pass = 0
    dataset.append([score, hours, label_fail, label_pass])

# 150 with low scores (20-40) + low hours
for i in range(150):
    score = np.random.uniform(20, 40)
    hours = np.random.uniform(0.5, 10)
    label_fail = 1
    label_pass = 0
    dataset.append([score, hours, label_fail, label_pass])

# Shuffle the dataset
np.random.shuffle(dataset)

# Create DataFrame and save to CSV
df = pd.DataFrame(dataset, columns=['Score', 'Study_Hours', 'Label_Fail', 'Label_Pass'])
df.to_csv('student_data_1000.csv', index=False)
print(f"Dataset saved to 'student_data_1000.csv' ({len(df)} samples)")
print(f"PASS samples: {(df['Label_Pass'] == 1).sum()}")
print(f"FAIL samples: {(df['Label_Fail'] == 1).sum()}\n")

print("Reading data from CSV...")
df = pd.read_csv('student_data_1000.csv')

# Extract inputs and labels
inputs = df[['Score', 'Study_Hours']].values
labels = df[['Label_Fail', 'Label_Pass']].values

mean = inputs.mean(axis=0)
std = inputs.std(axis=0)
inputs = (inputs - mean) / std

print(f"Loaded {len(inputs)} samples from CSV")
print(f"Input shape: {inputs.shape}")
print(f"Labels shape: {labels.shape}\n")

learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
epochs = 20000

input_size = 2
hidden_size1 = 16
hidden_size2 = 16
output_size = 2

names = ["student1", "student2"]

hidden_weights1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
hidden_biases1 = np.zeros(hidden_size1)
hidden_weights2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
hidden_biases2 = np.zeros(hidden_size2)
output_weights = np.random.randn(hidden_size2, output_size) * np.sqrt(1.0 / hidden_size2)
output_biases = np.zeros(output_size)


m_h_weights1 = np.zeros_like(hidden_weights1)
m_h_biases1 = np.zeros_like(hidden_biases1)
m_h_weights2 = np.zeros_like(hidden_weights2)
m_h_biases2 = np.zeros_like(hidden_biases2)
m_output_w = np.zeros_like(output_weights)
m_output_b = np.zeros_like(output_biases)

v_h_weights1 = np.zeros_like(hidden_weights1)
v_h_biases1 = np.zeros_like(hidden_biases1)
v_h_weights2 = np.zeros_like(hidden_weights2)
v_h_biases2 = np.zeros_like(hidden_biases2)
v_output_w = np.zeros_like(output_weights)
v_output_b = np.zeros_like(output_biases)
t = 0

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def adam_update(param, grad, m, v, t):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    param = param - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return param, m, v

def forward_pass(data):
    hidden_inputs1 = np.dot(data, hidden_weights1) + hidden_biases1
    hidden_outputs1 = leaky_relu(hidden_inputs1)
    hidden_inputs2 = np.dot(hidden_outputs1, hidden_weights2) + hidden_biases2
    hidden_outputs2 = leaky_relu(hidden_inputs2)
    output_inputs = np.dot(hidden_outputs2, output_weights) + output_biases
    output_probs = softmax(output_inputs)
    return output_probs, hidden_inputs1, hidden_outputs1, hidden_inputs2, hidden_outputs2

def backward_pass_adam(data, labels, output_probs, hidden_inputs1, hidden_outputs1, hidden_inputs2, hidden_outputs2):
    global output_weights, output_biases, hidden_weights1, hidden_biases1, hidden_weights2, hidden_biases2
    global m_h_weights1, m_h_biases1, m_h_weights2, m_h_biases2, m_output_w, m_output_b
    global v_h_weights1, v_h_biases1, v_h_weights2, v_h_biases2, v_output_w, v_output_b
    global t
    num_samples = data.shape[0]
    t += 1
    output_error = output_probs - labels
    output_delta = output_error
    d_output_weights = np.dot(hidden_outputs2.T, output_delta) / num_samples
    d_output_biases = np.sum(output_delta, axis=0) / num_samples

    hidden_error2 = np.dot(output_delta, output_weights.T)
    hidden_delta2 = hidden_error2 * leaky_relu_derivative(hidden_inputs2)
    d_hidden_weights2 = np.dot(hidden_outputs1.T, hidden_delta2) / num_samples
    d_hidden_biases2 = np.sum(hidden_delta2, axis=0) / num_samples

    hidden_error1 = np.dot(hidden_delta2, hidden_weights2.T)
    hidden_delta1 = hidden_error1 * leaky_relu_derivative(hidden_inputs1)
    d_hidden_weights1 = np.dot(data.T, hidden_delta1) / num_samples
    d_hidden_biases1 = np.sum(hidden_delta1, axis=0) / num_samples

    output_weights, m_output_w, v_output_w = adam_update(output_weights, d_output_weights, m_output_w, v_output_w, t)
    output_biases, m_output_b, v_output_b = adam_update(output_biases, d_output_biases, m_output_b, v_output_b, t)
    hidden_weights2, m_h_weights2, v_h_weights2 = adam_update(hidden_weights2, d_hidden_weights2, m_h_weights2, v_h_weights2, t)
    hidden_biases2, m_h_biases2, v_h_biases2 = adam_update(hidden_biases2, d_hidden_biases2, m_h_biases2, v_h_biases2 , t)
    hidden_weights1, m_h_weights1, v_h_weights1 = adam_update(hidden_weights1, d_hidden_weights1, m_h_weights1, v_h_weights1, t)
    hidden_biases1, m_h_biases1, v_h_biases1 = adam_update(hidden_biases1, d_hidden_biases1, m_h_biases1, v_h_biases1, t)

def loss_calculation(output_probs, labels):
    return -np.mean(np.sum(labels * np.log(output_probs + 1e-8), axis=1))

for epoch in range(epochs):
    output_probs, hidden_inputs1, hidden_outputs1, hidden_inputs2, hidden_outputs2 = forward_pass(inputs)
    backward_pass_adam(inputs, labels, output_probs, hidden_inputs1, hidden_outputs1, hidden_inputs2, hidden_outputs2)
    if (epoch + 1) % (epochs // 10) == 0:
        loss = loss_calculation(output_probs, labels)
        predictions = np.argmax(output_probs, axis=1)
        accuracy = np.mean(predictions == np.argmax(labels, axis=1)) * 100
        print(f"Epoch {epoch + 1:6,} | Loss: {loss:.4f} | Accuracy: {accuracy:.2f}%")

print("\nTraining Complete!\n")

output_probs, _, _, _, _ = forward_pass(inputs)
predictions = np.argmax(output_probs, axis=1)

original_inputs = inputs * std + mean

for i in range(min(5, len(inputs))):
    verdict = "PASS" if predictions[i] == 1 else "FAIL"
    confidence = output_probs[i, predictions[i]] * 100
    actual = "PASS" if np.argmax(labels[i]) == 1 else "FAIL"
    match = " " if verdict == actual else "✗"

    print(f"{match} Student {i + 1}: Score={original_inputs[i, 0]:.1f}, Hours={original_inputs[i, 1]:.1f}")
    print(f"Predicted: {verdict} ({confidence:.1f}%) | Actual: {actual}\n")

# PREDICTIONS ON USER INPUT
user_inputs_raw = np.array([
    [float(input("Enter Score for Student1: ")), float(input("Enter Study Hours for Student1: "))],
    [float(input("Enter Score for Student2: ")), float(input("Enter Study Hours for Student2: "))]
])
user_inputs = (user_inputs_raw - mean) / std
user_output_probs, _, _, _, _ = forward_pass(user_inputs)
user_predictions = np.argmax(user_output_probs, axis=1)


for i in range(len(user_inputs)):
    verdict = "PASS" if user_predictions[i] == 1 else "FAIL"
    confidence = user_output_probs[i, user_predictions[i]] * 100

    print(f"\n{names[i]}:")
    print(f"  Score: {user_inputs_raw[i, 0]:.1f}")
    print(f"  Study Hours: {user_inputs_raw[i, 1]:.1f}")
    print(f"  Prediction: {verdict}")
    print(f"  Confidence: {confidence:.1f}%")
    print(f"  Probabilities: [Fail: {user_output_probs[i, 0] * 100:.1f}%, Pass: {user_output_probs[i, 1] * 100:.1f}%]")
