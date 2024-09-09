from locust import HttpUser, between, task

class WebsiteUser(HttpUser):
    wait_time = between(1, 20)
    host = "http://localhost:8000"

    @task
    def upload_data(self):
        self.client.post("/respostas", json={
            "A1": 1,
            "A2": 1,
            "A3": 0,
            "A4": 0,
            "A5": 0,
            "A6": 1,
            "A7": 1,
            "A8": 0,
            "A9": 0,
            "A10": 0,
            "Age_Mons": 36,
            "Sex": "m",
            "Ethnicity": "middle eastern",
            "Jaundice": "yes",
            "Family_mem_with_ASD": "no",
            "Class_ASD_Traits": ""
        })
    
    @task
    def send_result(self):
        self.client.get(f"/resultado/1")
