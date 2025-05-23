# ---
# lambda-test: false  # auxiliary-file
# pytest: false
# ---
import random

from bs4 import BeautifulSoup
from constants import N_CHECKBOXES
from locust import HttpUser, between, task


class CheckboxesUser(HttpUser):
    wait_time = between(0.01, 0.1)  # Simulates a wait time between requests

    def load_homepage(self):
        """
        Simulates a user loading the homepage and fetching the state of the checkboxes.
        """
        response = self.client.get("/")
        soup = BeautifulSoup(response.text, "lxml")
        main_div = soup.find("main")
        self.id = main_div["hx-get"].split("/")[-1]

    @task(10)
    def toggle_random_checkboxes(self):
        """
        Simulates a user toggling a random checkbox.
        """
        n_checkboxes = random.binomialvariate(  # approximately poisson at 10
            n=100,
            p=0.1,
        )
        for _ in range(min(n_checkboxes, 1)):
            checkbox_id = int(
                N_CHECKBOXES * random.random() ** 2
            )  # Choose a random checkbox between 0 and 9999, more likely to be closer to 0
            self.client.post(
                f"/checkbox/toggle/{checkbox_id}",
                name="/checkbox/toggle",
            )

    @task(1)
    def poll_for_diffs(self):
        """
        Simulates a user polling for any outstanding diffs.
        """
        self.client.get(f"/diffs/{self.id}", name="/diffs")

    def on_start(self):
        """
        Called when a simulated user starts, typically used to initialize or login a user.
        """
        self.id = str(random.randint(1, 9999))
        self.load_homepage()
