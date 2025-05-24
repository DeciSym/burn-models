import json
import numpy as np
import matplotlib.pyplot as plt

# Load both logit sets
with open('rust_logits_hey.json') as f:
    rust_logits = json.load(f)
    
with open('python_logits_hey.json') as f:
    python_logits = json.load(f)

# Convert to arrays
rust_values = np.array([rust_logits[str(i)] for i in range(1000)])
python_values = np.array([python_logits[str(i)] for i in range(1000)])

# Compute relative differences
rust_relative = rust_values - rust_values.mean()
python_relative = python_values - python_values.mean()

print("Distribution Analysis:")
print(f"Rust mean: {rust_values.mean():.4f}, std: {rust_values.std():.4f}")
print(f"Python mean: {python_values.mean():.4f}, std: {python_values.std():.4f}")

print(f"\nRelative to mean:")
print(f"Rust relative std: {rust_relative.std():.4f}")
print(f"Python relative std: {python_relative.std():.4f}")

# Check if it's a scaling issue
scaling_factor = python_values.std() / rust_values.std()
print(f"\nScaling factor (Python std / Rust std): {scaling_factor:.4f}")

# Apply softmax to both
def softmax(x):
    exp_x = np.exp(x - x.max())  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

rust_probs = softmax(rust_values)
python_probs = softmax(python_values)

# Find top 10 tokens in each
rust_top10 = np.argsort(rust_probs)[-10:][::-1]
python_top10 = np.argsort(python_probs)[-10:][::-1]

print("\nTop 10 tokens by probability:")
print("Rust:", rust_top10.tolist())
print("Python:", python_top10.tolist())

# Check specific tokens
important_tokens = [187, 253, 13, 309]  # newline, the, comma, I
print("\nProbabilities for important tokens:")
for token in important_tokens:
    print(f"Token {token}: Rust={rust_probs[token]:.6f}, Python={python_probs[token]:.6f}")

# The key insight: check if the distribution shape is different
print("\nDistribution shape analysis:")
# Sort both distributions
rust_sorted = np.sort(rust_values)
python_sorted = np.sort(python_values)

# Check percentiles
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("Percentiles:")
for p in percentiles:
    rust_p = np.percentile(rust_values, p)
    python_p = np.percentile(python_values, p)
    print(f"  {p}%: Rust={rust_p:.2f}, Python={python_p:.2f}, Diff={rust_p-python_p:.2f}")

# Save plot
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.hist(rust_values, bins=50, alpha=0.7, label='Rust')
plt.hist(python_values, bins=50, alpha=0.7, label='Python')
plt.xlabel('Logit value')
plt.ylabel('Count')
plt.legend()
plt.title('Logit distributions')

plt.subplot(1, 2, 2)
plt.plot(rust_sorted, label='Rust')
plt.plot(python_sorted, label='Python')
plt.xlabel('Token rank')
plt.ylabel('Logit value')
plt.legend()
plt.title('Sorted logit values')

plt.tight_layout()
plt.savefig('logit_distributions.png')
print("\nSaved distribution plot to logit_distributions.png")