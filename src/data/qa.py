import json

import torch
from torch.utils.data import Dataset

from data.utils import (
    load_hf_dataset,
    preprocess_chat_instance,
    add_dataset_index,
    IGNORE_INDEX,
)


class QADataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        question_key="question",
        answer_key="answer",
        few_shot_dataset_hf_args=None,
        max_length=512,
        predict_with_generate=False,
    ):
        super(QADataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        self.fs_data = None
        if few_shot_dataset_hf_args is not None:
            raw_data = load_hf_dataset(**few_shot_dataset_hf_args)
            self.fs_data = {}
            self.fs_data[question_key] = raw_data[question_key]
            self.fs_data[answer_key] = raw_data[answer_key]
        self.template_args = template_args
        self.question_key = question_key
        self.answer_key = answer_key
        self.predict_with_generate = predict_with_generate

    def __len__(self):
        return len(self.data)

    def _process_sample(self, question, answer, index=-1):
        if self.fs_data is None:
            prompt_msgs, response_msgs = [question], [answer]
        else:
            prompt_msgs = self.fs_data[self.question_key] + [question]
            response_msgs = self.fs_data[self.answer_key] + [answer]
        tokenized_data = preprocess_chat_instance(
            self.tokenizer,
            self.template_args,
            prompt_msgs,
            response_msgs,
            self.max_length,
            self.predict_with_generate,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        if isinstance(answer, str):
            item = self._process_sample(question=question, answer=answer, index=index)
        elif isinstance(answer, list):
            item = {}
            for i, ans in enumerate(answer):
                sample_item = self._process_sample(
                    question=question, answer=ans, index=index
                )
                item[i] = sample_item
        else:
            raise NotImplementedError("answer format not found")
        return item


class QAwithIdkDataset(QADataset):
    def __init__(self, idk_path, return_original=True, *args, **kwargs):
        self.idk_path = idk_path
        self.return_original = return_original
        self.idk_responses = open(self.idk_path, "r").readlines()
        super().__init__(*args, **kwargs)

    def item_with_idk(self, question):
        rand_pos = torch.randint(0, len(self.idk_responses), (1,)).item()
        idk_response = self.idk_responses[rand_pos].strip()
        idk_item = self._process_sample(question=question, answer=idk_response)
        return idk_item

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            idk_item = self.item_with_idk(question)
            return_item["alternate"] = idk_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                idk_item = self.item_with_idk(question)
                return_item["alternate"] = idk_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]


class QAwithAlternateDataset(QADataset):
    def __init__(self, alternate_key, return_original=True, *args, **kwargs):
        self.alternate_key = alternate_key
        self.return_original = return_original
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        question = self.data[idx][self.question_key]
        if isinstance(item, dict):
            return_item = {"original": item}
            alt_item = self._process_sample(
                question=question, answer=self.data[idx][self.alternate_key]
            )
            return_item["alternate"] = alt_item
            # return_item = [item, idk_item]
        elif isinstance(item, list) or isinstance(item, tuple):
            return_item = []
            for sample_item in item:
                return_item = {"original": sample_item}
                alt_item = self._process_sample(
                    question=question, answer=self.data[idx][self.alternate_key]
                )
                return_item["alternate"] = alt_item
                # return_item.append([sample_item, idk_item])
        return return_item if self.return_original else return_item["alternate"]


class QAwithCommonWordsDataset(QADataset):
    """QA dataset with token-level UW/GW masks for TPO-style targeted unlearning.

    The dataset is expected to contain a ``common_words`` field: a list of JSON strings
    (or dicts) ``{"word": ..., "start": ..., "end": ...}`` with character-level spans
    relative to the answer text, marking the General Words (GW) that should be
    preserved during unlearning. The remaining answer tokens are the Unwanted Words
    (UW) targeted for forgetting.

    Ported from the official TPO repository
    (https://github.com/guts-yang/Unlearning-TPO, ``TOFU/data_module.py``), adapted
    to this repo's chat-template pipeline. Each item carries two label tensors:

    - ``labels``: IGNORE_INDEX on prompt/padding/GW tokens; only UW tokens are active
      (used for the logit preference / forgetting loss).
    - ``gw_labels``: IGNORE_INDEX everywhere except GW tokens (used for the
      preservation loss).
    """

    def __init__(self, common_words_key="common_words", *args, **kwargs):
        self.common_words_key = common_words_key
        super().__init__(*args, **kwargs)

    def _build_chat_texts(self, prompt_msgs, response_msgs):
        """Mirror the chat construction in ``preprocess_chat_instance``
        (data/utils.py) but additionally return the rendered prompt text so that
        character-level answer spans can be mapped onto token positions."""
        template_args = self.template_args
        tokenizer = self.tokenizer
        if template_args["apply_chat_template"]:
            chat = []
            system_prompt = template_args.get("system_prompt", None)
            if system_prompt:
                chat += [{"role": "system", "content": system_prompt}]
            for prompt, response in zip(prompt_msgs, response_msgs):
                chat += [{"role": "user", "content": prompt}]
                chat += [{"role": "assistant", "content": response}]
            date_str = template_args.get("date_string", None)
            date_info = {"date_string": date_str} if date_str is not None else {}
            full_text = tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False, **date_info
            )
            prompt_text = tokenizer.apply_chat_template(
                chat[:-1], tokenize=False, add_generation_prompt=True, **date_info
            )
            prompt_ids = tokenizer.apply_chat_template(
                chat[:-1], tokenize=True, add_generation_prompt=True, **date_info
            )
        else:
            wrapped_prompt = ""
            system_prompt_with_special_tokens = template_args.get(
                "system_prompt_with_special_tokens", None
            )
            if system_prompt_with_special_tokens:
                wrapped_prompt += system_prompt_with_special_tokens
            n_few_shot = len(prompt_msgs) - 1
            for i in range(n_few_shot):
                wrapped_prompt += (
                    template_args["user_start_tag"]
                    + prompt_msgs[i]
                    + template_args["user_end_tag"]
                    + template_args["asst_start_tag"]
                    + response_msgs[i]
                    + template_args["asst_end_tag"]
                )
            wrapped_prompt += (
                template_args["user_start_tag"]
                + prompt_msgs[-1]
                + template_args["user_end_tag"]
                + template_args["asst_start_tag"]
            )
            full_text = wrapped_prompt + response_msgs[-1]
            prompt_text = wrapped_prompt
            prompt_ids = tokenizer(
                wrapped_prompt, add_special_tokens=True
            )["input_ids"]
        return full_text, prompt_text, prompt_ids

    def _process_sample_with_masks(self, question, answer, common_words, index=-1):
        full_text, prompt_text, prompt_ids = self._build_chat_texts(
            [question], [answer]
        )
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=True,
        )
        input_ids = list(encoded["input_ids"])
        offset_mapping = list(encoded["offset_mapping"])

        # mirror preprocess_chat_instance: ensure the sequence ends with an eos token
        if input_ids[-1] != self.tokenizer.eos_token_id:
            input_ids = input_ids + [self.tokenizer.eos_token_id]
            # the appended eos lies outside `full_text`; it can never overlap a GW span
            offset_mapping = offset_mapping + [
                (len(full_text), len(full_text))
            ]

        # Llama-3 tokenizers report character-granular offsets; align each token's end
        # with the next token's start (same fix as the official TPO implementation)
        if "Llama-3" in getattr(self.tokenizer, "name_or_path", ""):
            for i in range(len(offset_mapping) - 1):
                offset_mapping[i] = (offset_mapping[i][0], offset_mapping[i + 1][0])

        len_matched = min(len(prompt_ids), len(input_ids))

        # map the character-level common word spans (relative to the answer) onto
        # token positions; a token is GW iff its span overlaps any common word span
        gw_mask = [False] * len(input_ids)
        prefix_len = len(prompt_text)
        for word in common_words:
            if isinstance(word, str):
                word = json.loads(word)
            start = prefix_len + word["start"]
            end = prefix_len + word["end"]
            for i, (encoded_start, encoded_end) in enumerate(offset_mapping):
                if encoded_start < end and encoded_end > start:
                    gw_mask[i] = True

        labels = [IGNORE_INDEX] * len(input_ids)
        gw_labels = [IGNORE_INDEX] * len(input_ids)
        for i in range(len_matched, len(input_ids)):
            if gw_mask[i]:
                gw_labels[i] = input_ids[i]
            else:
                labels[i] = input_ids[i]

        item_dct = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "gw_labels": torch.tensor(gw_labels),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "index": index,
        }
        return item_dct

    def __getitem__(self, idx):
        question = self.data[idx][self.question_key]
        answer = self.data[idx][self.answer_key]
        index = self.data[idx]["index"]
        common_words = self.data[idx][self.common_words_key]
        assert isinstance(answer, str), (
            "QAwithCommonWordsDataset expects single-answer (str) examples, "
            f"got {type(answer)}"
        )
        return self._process_sample_with_masks(
            question=question, answer=answer, common_words=common_words, index=index
        )
