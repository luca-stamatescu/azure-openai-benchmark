import logging
import math
import time
from abc import ABC, abstractmethod

import wonderwords

from .oaitokenizer import num_tokens_from_messages

CACHED_PROMPT = ""
CACHED_MESSAGES_TOKENS = 0

class BaseContextGenerator(ABC):
    """
    Abstract base class for generation of messages array.

    Derived classes should make use of CACHED_PROMPT and CACHED_MESSAGES_TOKENS
    in order to reuse the same prompt across multiple calls.
    """
    @abstractmethod
    def __call__(self, model: str, context_tokens: int, max_tokens: int = None) -> ([dict], int):
        """
        Generate `messages` array based on context_tokens and max_tokens.
        
        Returns Tuple of messages array and actual context token count.
        """
        pass


class RandomWordsThenEssayGenerator(BaseContextGenerator):
    def __call__(self, model: str, context_tokens: int, max_tokens: int = None) -> ([dict], int):
        """
        Generate `messages` array based on tokens and max_tokens.
        Output includes a message of random words, followed by a message asking for 
        an essay of 3*max_tokens.
        
        NOTE: This method can result in certain models refusing to create an essay,
        instead returning <100 words (e.g. GPT-3.5-Turbo).
        
        Returns Tuple of messages array and actual context token count.
        """
        global CACHED_PROMPT
        global CACHED_MESSAGES_TOKENS
        try:
            r = wonderwords.RandomWord()
            messages = [{"role": "user", "content": str(time.time()) + " "}]
            if max_tokens is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": str(time.time())
                        + f" write an essay in at least {max_tokens*3} words",
                    }
                )
            messages_tokens = 0

            if len(CACHED_PROMPT) > 0:
                messages[0]["content"] += CACHED_PROMPT
                messages_tokens = CACHED_MESSAGES_TOKENS
            else:
                prompt = ""
                base_prompt = messages[0]["content"]
                while True:
                    messages_tokens = num_tokens_from_messages(messages, model)
                    remaining_tokens = context_tokens - messages_tokens
                    if remaining_tokens <= 0:
                        break
                    prompt += (
                        " ".join(r.random_words(amount=math.ceil(remaining_tokens / 4)))
                        + " "
                    )
                    messages[0]["content"] = base_prompt + prompt

                CACHED_PROMPT = prompt
                CACHED_MESSAGES_TOKENS = messages_tokens

        except Exception as e:
            print(e)

        return (messages, messages_tokens)
    

class AutoContextGenerator(BaseContextGenerator):
    """
    Context generator that automatically selects the most appropriate method 
    based on anecdotal experience with each method.
    """
    def __init__(self):
        self._generators = {name: cls() for name, cls in GENERATOR_NAME_TO_CLASS.items() if name != "auto"}
        self._model_mapping = {
            "gpt-4": "random_word_essay",
            "gpt-4-0314": "random_word_essay",
            "gpt-4-32k-0314": "random_word_essay",
            "gpt-4-0613": "random_word_essay",
            "gpt-4-32k-0613": "random_word_essay",
            "gpt-3.5-turbo": "random_word_essay",
            "gpt-3.5-turbo-0613": "random_word_essay",
            "gpt-3.5-turbo-16k-0613": "random_word_essay",
        }

    def __call__(self, model: str, context_tokens: int, max_tokens: int = None) -> ([dict], int):
        """
        Defaults to the most appropriate method based on model, as defined in
        self._model_mapping.
        
        Returns Tuple of messages array and actual context token count.
        """
        return self._generators[self._model_mapping[model]](model, context_tokens, max_tokens)


GENERATOR_NAME_TO_CLASS = {
    "random_word_essay": RandomWordsThenEssayGenerator,
    "auto": AutoContextGenerator,
}
