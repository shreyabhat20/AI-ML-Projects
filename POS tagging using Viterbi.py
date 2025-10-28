from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Training Data
train_data = [
    [("the", "DET"), ("cat", "NOUN"), ("sat", "VERB")],
    [("the", "DET"), ("dog", "NOUN"), ("barked", "VERB")],
    [("a", "DET"), ("dog", "NOUN"), ("sat", "VERB")],
]

transition = defaultdict(lambda: defaultdict(int))
emission = defaultdict(lambda: defaultdict(int))
start_prob = defaultdict(int)
tag_counts = defaultdict(int)

# Step 2: Counting Frequencies
for sentence in train_data:
    prev_tag = None
    for i, (word, tag) in enumerate(sentence):
        tag_counts[tag] += 1
        emission[tag][word] += 1
        if i == 0:
            start_prob[tag] += 1
        else:
            transition[prev_tag][tag] += 1
        prev_tag = tag

# Step 3: Normalize to Probabilities
def normalize(d):
    total = sum(d.values())
    return {k: v / total for k, v in d.items()}

start_prob = normalize(start_prob)
for tag in emission:
    emission[tag] = normalize(emission[tag])
for prev in transition:
    transition[prev] = normalize(transition[prev])

# Step 4: Viterbi Algorithm
def viterbi(sentence, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialization
    for state in states:
        V[0][state] = start_p.get(state, 0) * emit_p[state].get(sentence[0], 1e-6)
        path[state] = [state]

    # Recursion
    for t in range(1, len(sentence)):
        V.append({})
        new_path = {}

        for curr_state in states:
            max_prob, prev_state = max(
                (V[t - 1][y0] * trans_p[y0].get(curr_state, 1e-6) * emit_p[curr_state].get(sentence[t], 1e-6), y0)
                for y0 in states
            )
            V[t][curr_state] = max_prob
            new_path[curr_state] = path[prev_state] + [curr_state]

        path = new_path

    # Termination
    n = len(sentence) - 1
    prob, final_state = max((V[n][y], y) for y in states)
    return path[final_state], V

# Step 5: Test Sentence
test_sentence = ["a", "cat", "barked"]
states = list(tag_counts.keys())

predicted_tags, v_matrix = viterbi(test_sentence, states, start_prob, transition, emission)

# Step 6: Output
print("Input Sentence:", test_sentence)
print("Predicted POS Tags:", predicted_tags)

# Step 7: Visualization
# 1. Transition Matrix Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(
    [[transition[i].get(j, 0) for j in states] for i in states],
    annot=True, cmap="YlGnBu", xticklabels=states, yticklabels=states
)
plt.title("Transition Probability Matrix")
plt.xlabel("To State")
plt.ylabel("From State")
plt.show()

# 2. Emission Matrix Heatmap
words = sorted(set(word for s in train_data for word, _ in s))
plt.figure(figsize=(7, 4))
sns.heatmap(
    [[emission[i].get(w, 0) for w in words] for i in states],
    annot=True, cmap="Oranges", xticklabels=words, yticklabels=states
)
plt.title("Emission Probability Matrix")
plt.xlabel("Word")
plt.ylabel("POS Tag")
plt.show()

# 3. Viterbi Path Visualization
plt.figure(figsize=(7, 3))
plt.plot(range(len(test_sentence)), predicted_tags, marker='o', linestyle='-', color='purple')
plt.xticks(range(len(test_sentence)), test_sentence)
plt.title("Predicted POS Sequence (Viterbi Path)")
plt.xlabel("Words")
plt.ylabel("Predicted Tag")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
