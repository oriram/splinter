""" Adapted from HuggingFace code for SQuAD """
import json
from tqdm import tqdm
import os


class MRQAExample:
    """
        A single training/test example for the MRQA dataset, as loaded from disk.

        Args:
            qas_id: The example's unique identifier
            question_text: The question string
            question_tokens: The tokenized question
            context_text: The context string
            context_tokens: The tokenized context
            answer_text: The answer string
            answer_tokens: The tokenized answer
            start_position_character: The character position of the start of the answer
            answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        """

    def __init__(
            self,
            qas_id,
            question_text,
            question_tokens,
            context_text,
            context_tokens,
            answer_text,
            start_position_character,
            answers=[],
            is_impossible=False
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.question_tokens = question_tokens
        self.context_text = context_text
        self.context_tokens = context_tokens
        self.answer_text = answer_text
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        self.is_impossible = is_impossible
        doc_tokens = []
        char_to_word_offset = []
        for i, (token, token_char_position) in enumerate(context_tokens):
            doc_tokens.append(token)
            char_to_word_offset.extend([i] * len(token))

            # Verifying this is not the last token:
            if i >= len(context_tokens) - 1:
                continue
            next_token_start_position = context_tokens[i + 1][1]
            chars_to_next_token = next_token_start_position - len(char_to_word_offset)
            assert chars_to_next_token >= 0
            if chars_to_next_token > 0:
                char_to_word_offset.extend([i] * chars_to_next_token)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


class MRQAProcessor:
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"

    def create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            context = entry["context"]
            context_tokens = entry["context_tokens"]
            for qa in entry["qas"]:
                qas_id = qa["id" if "id" in qa else "qid"]
                question_text = qa["question"]
                question_tokens = qa["question_tokens"]

                if is_training:
                    answer = qa["detected_answers"][0]
                    answer_text = " ".join([c_t[0] for c_t in context_tokens[answer['token_spans'][0][0]: answer['token_spans'][0][1] + 1]])
                    start_position_character = answer["char_spans"][0][0]
                    answers = []
                else:
                    start_position_character = None
                    answer_text = None
                    answers = [{"text": " ".join([c_t[0] for c_t in context_tokens[answer['token_spans'][0][0]: answer['token_spans'][0][1] + 1]])}
                               for answer in qa["detected_answers"]]

                examples.append(MRQAExample(qas_id=qas_id, question_text=question_text, question_tokens=question_tokens,
                                            context_text=context, context_tokens=context_tokens,
                                            answer_text=answer_text,
                                            start_position_character=start_position_character, answers=answers))
        return examples

    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
                os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            print(reader.readline())
            input_data = [json.loads(line) for line in reader]
        return self.create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
                os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            print(reader.readline())
            input_data = [json.loads(line) for line in reader]
        return self.create_examples(input_data, "dev")
