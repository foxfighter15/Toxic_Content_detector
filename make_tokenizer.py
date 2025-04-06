import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Step 1: Load training data
df = pd.read_csv("train.csv")  # Update path if needed
texts = df["comment_text"].astype(str).tolist()


# Step 2: Create tokenizer and fit
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

# Step 3: Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ Tokenizer saved as tokenizer.pkl")
