import random

import srsly
from tqdm import tqdm


ID_LIST = []
for x in range(100000):
    q_id = str(x)
    while len(q_id) < 12:
        q_id = "0" + q_id
    ID_LIST.append(q_id)

CHOICE_LETTERS = ["a", "b", "c", "d", "e"]
TF_ANSWERS = {"a": "true", "b": "false"}


def generate_id():
    random_id = random.choice(ID_LIST)
    ID_LIST.remove(random_id)
    return random_id


def generate_lesson(doc):
    lesson = ""
    for lesson_section in [
        "Apply Concepts",
        "Introduction",
        "Lesson Objetives",
        "Lesson Summary",
        "Points to Consider",
        "Recall",
        "Think Critically",
    ]:
        if lesson_section in doc["adjunctTopics"]:
            section_text = doc["adjunctTopics"][lesson_section]["content"]["text"]
            lesson = lesson + f"\n\n{lesson_section} \n{section_text}"
    lesson = lesson + "\n\nVocabulary"
    for word, meaning in doc["adjunctTopics"]["Vocabulary"].items():
        lesson = lesson + f"\n - {word}: {meaning}"
    for topic_id, topic in doc["topics"].items():
        lesson = lesson + f'\n\n{topic["topicName"]} \n{topic["content"]["text"]}'
    return lesson


def answer_randomizer(answers, correct_answer):
    out_answers = {}
    options = list(answers.values())
    correct_option = answers[correct_answer]
    for letter, answer in answers.items():
        new_answer = random.choice(options)
        out_answers[letter] = new_answer
        if new_answer == correct_option:
            out_correct_answer = letter
        options.remove(new_answer)
    return out_answers, out_correct_answer


def process_ai2d(input_path):
    out_dataset = []
    ds = [x for x in srsly.read_jsonl(input_path)]
    for doc in tqdm(ds):
        question = doc["question"]
        answers = doc["answers"]
        image_path = f"ai2d/images_fix/{doc['diagram']}"
        correct_answer = doc["correct_answer"]
        if correct_answer not in answers:
            continue
        answers, correct_answer = answer_randomizer(answers, correct_answer)
        conv = []
        human_prompt = f'<image>\nTake a look at the diagram and answer the following question by choosing one of the possible answers.\nQuestion: "{question}"\nAnswer choices:'
        for k, v in answers.items():
            human_prompt = human_prompt + f"\n{k}) {v}"
        human_prompt = human_prompt + '\nArrange your output as json such as {"answer": "<your choice>"}. Your choice must be the associated letter to your answer.'
        conv.append({"from": "human", "value": human_prompt})
        conv.append(
            {"from": "gpt", "value": '{"answer": "' + correct_answer + '"}'})
        out_dataset.append(
            {"id": generate_id(), "image": image_path, "conversations": conv})
    return out_dataset


def process_scienceqa(input_path):
    out_dataset = []
    ds = [x for x in srsly.read_jsonl(input_path)]
    for doc in tqdm(ds):
        question = doc["question"]
        answers = doc["answers"]
        correct_answer = doc["correct_answer"]
        if correct_answer not in answers:
            continue
        answers, correct_answer = answer_randomizer(answers, correct_answer)
        conv = []
        if "diagram" in doc:
            image_path = f"scienceqa/train/{doc['diagram']}"
            human_prompt = "<image>\nTake a look at the diagram and answer"
        else:
            human_prompt = "Answer"
        human_prompt += " the following question by choosing one of the possible answers.\n"
        if "context" in doc:
            human_prompt += f'Context: "{doc["context"]}"\n'
        human_prompt += f'Question: "{question}"\nAnswer choices:'
        for k, v in answers.items():
            human_prompt = human_prompt + f"\n{k}) {v}"
        human_prompt = human_prompt + '\nArrange your output as json such as {"answer": "<your choice>"}. Your choice must be the associated letter to your answer.'
        conv.append({"from": "human", "value": human_prompt})
        conv.append(
            {"from": "gpt", "value": '{"answer": "' + correct_answer + '"}'})
        if "diagram" in doc:
            out_dataset.append({"id": generate_id(), "image": image_path, "conversations": conv})
        else:
            out_dataset.append({"id": generate_id(), "conversations": conv})
    return out_dataset


def process_tqa(input_path):
    out_dataset = []
    tqa_dataset = [x for x in srsly.read_json(input_path)]
    for doc in tqdm(tqa_dataset):
        for qType in ["nonDiagramQuestions", "diagramQuestions"]:
            for _, q_doc in doc["questions"][qType].items():
                conv = []
                question = q_doc["beingAsked"]["processedText"]
                answers = {k: v["processedText"] for k, v in q_doc["answerChoices"].items()}
                correct_answer = q_doc["correctAnswer"]["processedText"]
                if qType == "diagramQuestions" or (qType == "nonDiagramQuestions" and q_doc["questionSubType"] == "Multiple Choice"):
                    if correct_answer not in answers:
                        continue
                    answers, correct_answer = answer_randomizer(answers, correct_answer)
                    if qType=="diagramQuestions":
                        image_path = f"tqa_data/train/{q_doc['imagePath']}"
                        human_prompt = "<image>\nTake a look at the diagram and answer"
                    else:
                        human_prompt = "Answer"
                    human_prompt += f' the following question by choosing one of the possible answers.\nQuestion: "{question}"\nAnswer choices:'
                    for k, v in answers.items():
                        human_prompt = human_prompt + f"\n{k}) {v}"
                    human_prompt += '\nArrange your output as json such as {"answer": "<your choice>"}. Your choice must be the associated letter to your answer.'
                else:
                    human_prompt = f'Respond if the following statement is true or false.\nStatement: "{question}"'
                    human_prompt += '\nArrange your output as json such as {"answer": "<True|False>"}.'
                    if correct_answer in answers and answers[correct_answer].lower() in ["true", "false"]:
                        correct_answer = answers[correct_answer]
                    else:
                        continue
                conv.append({"from": "human", "value": human_prompt})
                conv.append(
                    {"from": "gpt", "value": '{"answer": "' + correct_answer + '"}'})
                out_doc = {"id": generate_id(), "conversations": conv}
                if qType=="diagramQuestions":
                    out_doc["image"] = image_path
                out_dataset.append(out_doc)
    return out_dataset


def process_extras():
    out_dataset = []
    ds = load_dataset("allenai/openbookqa")
    for doc in tqdm(ds["train"]):
        answers = {letter: answer for letter, answer in zip(doc["choices"]["label"], doc["choices"]["text"])}
        correct_answer = doc["answerKey"]
        if correct_answer not in answers:
            continue
        answers, correct_answer = answer_randomizer(answers, correct_answer)
        human_prompt = f'Answer the following question by choosing one of the possible answers.\nQuestion: "{doc["question_stem"]}"\nAnswer choices:'
        for k, v in answers.items():
            human_prompt = human_prompt + f"\n{k}) {v}"
        conv = [{"from": "human", "value": human_prompt}, {"from": "gpt", "value": '{"answer": "' + correct_answer + '"}'}]
        out_dataset.append({"id": generate_id(), "conversations": conv})
    for subset in ["ARC-Easy", "ARC-Challenge"]:
        ds = load_dataset("allenai/ai2_arc", subset)
        for doc in tqdm(ds["train"]):
            answers = {letter: answer for letter, answer in zip(doc["choices"]["label"], doc["choices"]["text"])}
            correct_answer = doc["answerKey"]
            if correct_answer not in answers:
                continue
            answers, correct_answer = answer_randomizer(answers,
                                                        correct_answer)
            human_prompt = f'Answer the following question by choosing one of the possible answers.\nQuestion: "{doc["question"]}"\nAnswer choices:'
            for k, v in answers.items():
                human_prompt = human_prompt + f"\n{k}) {v}"
            human_prompt = human_prompt + '\nArrange your output as json such as {"answer": "<your choice>"}. Your choice must be the associated letter to your answer.'
            conv = [{"from": "human", "value": human_prompt}, {"from": "gpt", "value": '{"answer": "' + correct_answer + '"}'}]
            out_dataset.append({"id": generate_id(), "conversations": conv})
    return out_dataset