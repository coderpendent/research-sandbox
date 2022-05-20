import os
import time
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForQuestionAnswering
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer


def convert_model_to_onnx():
    # load vanilla transformers and convert to onnx
    model = ORTModelForSequenceClassification.from_pretrained(model_id, from_transformers=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # save onnx checkpoint and tokenizer
    model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)


def predict_w_original(data, model_id):
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # test the model with using transformers pipeline, with handle_impossible_answer for squad_v2
    classifier = pipeline(task, model=model, tokenizer=tokenizer, handle_impossible_answer=True)
    start = time.time()
    resp = classifier(data['text'], data['labels'], multi_label=data['multi_label'])
    end = time.time()

    print(f"{resp}, \n, Time with original model: {round(end - start, 4)}")


def predict_w_compressed(data, onnx_path):
    model = ORTModelForSequenceClassification.from_pretrained(onnx_path, file_name="model.onnx")
    tokenizer = AutoTokenizer.from_pretrained(onnx_path)

    # test the model with using transformers pipeline, with handle_impossible_answer for squad_v2
    classifier = pipeline(task, model=model, tokenizer=tokenizer, handle_impossible_answer=True)
    start = time.time()
    resp = classifier(data['text'], data['labels'], multi_label=data['multi_label'])
    end = time.time()

    print(f"{resp}, \n, Time with compressed model: {round(end-start, 4)}")


# sample request to zero-shot classifier
data = {
    "text": "I loved the Italian restaurant we at at yesterday.",
    "labels": ["positive", "negative"],
    "multi_label": False,
    "use_compressed_model": True,
}

model_id = "facebook/bart-large-mnli"
onnx_path = Path("onnx")
task = "zero-shot-classification"

if not os.path.exists(onnx_path):
    convert_model_to_onnx()

predict_w_original(data, model_id)
predict_w_compressed(data, onnx_path)
