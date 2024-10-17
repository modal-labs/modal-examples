import random

import locust

parti_prompts = [  # mostly from parti-prompts, https://huggingface.co/datasets/nateraw/parti-prompts
    "A dignified beaver wearing glasses, a vest, and colorful neck tie. He stands next to a tall stack of books in a library.",
    "A punk rock squirrel in a studded leather jacket shouting into a microphone while standing on a stump and holding a beer on dark stage.",
    "Horses pulling a carriage on the moon's surface, with the Statue of Liberty and Great Pyramid in the background. The Planet Earth can be seen in the sky.",
    "A set of 2x2 emoji icons with happy, angry, surprised and sobbing faces. The emoji icons look like dogs. All of the dogs are wearing blue turtlenecks.",
    "A set of 2x2 emoji icons with happy, angry, surprised and sobbing faces. The emoji icons look like macaroons. All of the macaroons are wearing cowboy hats.",
    "A richly textured oil painting of a young badger delicately sniffing a yellow rose next to a tree trunk. A small waterfall can be seen in the background.",
    "A portrait of a metal statue of a pharaoh wearing steampunk glasses and a leather jacket over a white t-shirt that has a drawing of a space shuttle on it.",
    "A raccoon wearing formal clothes, wearing a tophat and holding a cane. The raccoon is holding a garbage bag. Oil painting in the style of abstract cubism.",
    "A single beam of light enter the room from the ceiling. The beam of light is illuminating an easel. On the easel there is a Rembrandt painting of a raccoon",
    "A train ride in the monsoon rain in Kerala. With a Koala bear wearing a hat looking out of the window. There is a lot of coconut trees out of the window.",
    "A group of farm animals (cows, sheep, and pigs) made out of cheese and ham, on a wooden board. There is a dog in the background eyeing the board hungrily.",
]

prompts = parti_prompts + [
    "an armchair in the shape of an avocado",
    "the word 'Modal' in neon green font on a computer screen. two people are looking at the screen, one is pointing at the word",
]


class WebsiteUser(locust.HttpUser):
    wait_time = locust.between(1, 5)
    headers = {}

    @locust.task
    def chat_completion(self):
        payload = {
            "args": {"prompt": random.choice(prompts)},
        }

        response = self.client.request(
            "POST",
            "/",
            json=payload,
            headers=self.headers,
            params={"compile": 1},
        )
        response.raise_for_status()
