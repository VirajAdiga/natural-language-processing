import torch
import pandas

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel


github_issues_dataset = load_dataset("lewtun/github-issues", split="train")
# Filtering data from dataset which is not required for search
github_issues_dataset = github_issues_dataset.filter(
    lambda x: (x["is_pull_request"] is False and len(x["comments"]) > 0)
)

# Filtering out non-required columns from dataset
columns = github_issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
github_issues_dataset = github_issues_dataset.remove_columns(columns_to_remove)

github_issues_dataset.set_format("pandas")
df = github_issues_dataset[:]
comments_df = df.explode("comments", ignore_index=True)

comments_dataset = Dataset.from_pandas(comments_df)
# Cleaning comments dataset which are not too short and might not be relevant
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)
comments_dataset = comments_dataset.filter(
    lambda x: x["comment_length"] > 15
)


# Function to concatenate the required columns of dataset row
def concatenate_columns(examples):
    return {
        "text": examples["title"] + "\n" + examples["body"] + "\n" + examples["comments"]
    }


comments_dataset = comments_dataset.map(concatenate_columns)


# Creating text embeddings
model_checkpoint = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)

device = torch.device("cpu")
model.to(device)


def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]


# Function to return embedding from some text
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)


embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)


# Creating FAISS index for similarity search
embeddings_dataset.add_faiss_index(column="embeddings")
question = "What is a dataset?"
question_embedding = get_embeddings([question]).detach().cpu().numpy()

scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)

result_df = pandas.DataFrame.from_dict(samples)
result_df["scores"] = scores
result_df.sort_values("scores", ascending=False, inplace=False)

# Display results which match with question
for _, row in result_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()
