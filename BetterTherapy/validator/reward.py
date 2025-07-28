# The MIT License (MIT)
# Copyright © 2025 BetterTherapy

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bittensor as bt
import numpy as np

from neurons import validator


def reward(self: validator.Validator, prompt: str, base_response: str, responses: list[str]) -> np.ndarray:
    """
    Reward the miner responses to the prompt. This method returns a reward
    value for each miner, which is used to update the miner's score.

    Returns:
    - np.ndarray: The reward value for the miner.
    """
    scores = self.evals.judge_responses(prompt, base_response, responses)
    bt.logging.info(
        f"In rewards, prompt val: {prompt}, base_response val: {base_response}, responses val: {responses}, scores val: {scores}"
    )
    return scores


def get_rewards(
    self: validator.Validator,
    prompt: str,
    base_response: str,
    responses: list[str],
) -> np.ndarray:
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - prompt (str): The prompt sent to the miner.
    - base_response (str): The base response from the base model for evaluation.
    - responses (List[str]): A list of responses from the miner.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    """
    scores = reward(self, prompt, base_response, responses)
    return np.array(scores)
