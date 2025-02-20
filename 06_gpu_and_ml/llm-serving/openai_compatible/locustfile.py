import logging
import random

import locust

messages = [
    {
        "role": "system",
        "content": "You are a salesman for Modal, the cloud-native serverless Python computing platform.",
    },
    {
        "role": "user",
        "content": "Give me two fun date ideas.",
    },
]


class WebsiteUser(locust.HttpUser):
    wait_time = locust.between(1, 5)
    headers = {
        "Authorization": "Bearer super-secret-key",
        "Accept": "application/json",
    }

    @locust.task
    def chat_completion(self):
        payload = {
            "model": "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w4a16",
            "messages": messages,
        }

        response = self.client.request(
            "POST", "/v1/chat/completions", json=payload, headers=self.headers
        )
        response.raise_for_status()
        if random.random() < 0.01:
            logging.info(response.json()["choices"][0]["message"]["content"])
