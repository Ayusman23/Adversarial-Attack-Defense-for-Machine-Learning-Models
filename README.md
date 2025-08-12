# 🛡️ Adversarial Attack & Defense Dashboard

An interactive **Machine Learning security dashboard** for training CNN models, generating adversarial attacks (FGSM, PGD, DeepFool), and testing various defense strategies such as Adversarial Training, Input Preprocessing, and Defensive Distillation.

This project combines:
- **FastAPI** backend for model training, attacks, defenses, and metrics
- **PyTorch** for deep learning & adversarial robustness
- **HTML/CSS/JavaScript** frontend for a beautiful, responsive UI

---

## 📌 Features

- **Model Training**
  - Train CNN models on MNIST or CIFAR-10
  - Adjustable learning rate & epochs
  - View architecture & parameters

- **Adversarial Attacks**
  - FGSM (Fast Gradient Sign Method)
  - PGD (Projected Gradient Descent)
  - DeepFool
  - Adjustable epsilon (perturbation magnitude)
  - Visualization of original, adversarial, and perturbation images

- **Defenses**
  - Adversarial Training
  - Input Preprocessing (Gaussian Noise, Median Filtering, Compression)
  - Defensive Distillation

- **Evaluation Metrics**
  - Clean Accuracy
  - Robust Accuracy
  - Attack Success Rate
  - Real-time progress bars & metric updates

- **Results & Visualization**
  - Attack success rate comparison
  - Accuracy vs. perturbation analysis
  - Defense effectiveness charts

---

## 🗂 Project Structure



adversarial\_defense/
│── backend/
│   ├── main.py          # FastAPI backend
│   ├── model.py         # CNN architecture & training
│   ├── attacks.py       # FGSM, PGD, DeepFool implementations
│   ├── defenses.py      # Defense method implementations
│   ├── utils.py         # Dataset loading & metrics
│
│── frontend/
│   ├── index.html       # Dashboard UI
│   ├── script.js        # API calls & dynamic updates
│   ├── style.css        # Styling for dashboard
│
└── README.md



## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
bash
git clone https://github.com/YOUR_USERNAME/adversarial-defense-dashboard.git
cd adversarial-defense-dashboard


 2️⃣ Create a virtual environment

bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows


 3️⃣ Install dependencies

bash
pip install -r requirements.txt


**Example `requirements.txt`:**


fastapi
uvicorn
torch
torchvision
numpy
pillow


### 4️⃣ Run the backend

bash
cd backend
uvicorn main:app --reload


Backend will start at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

### 5️⃣ Open the frontend

Open `frontend/index.html` in your browser.
The UI will connect to the backend APIs automatically (make sure ports match).

---

## 🔌 API Endpoints

| Method | Endpoint   | Description                             |
| ------ | ---------- | --------------------------------------- |
| POST   | `/train`   | Train CNN model                         |
| POST   | `/attack`  | Generate adversarial examples           |
| POST   | `/defense` | Apply selected defense method           |
| GET    | `/metrics` | Get real-time accuracy & attack metrics |

---

## 🧪 Example API Request (Training Model)

bash
curl -X POST "http://127.0.0.1:8000/train" \
-H "Content-Type: application/json" \
-d '{"dataset": "MNIST", "epochs": 2, "lr": 0.001}'


**Response:**

json
{
  "status": "success",
  "clean_accuracy": 92.3
}


## 🎯 Roadmap

* [ ] Implement real-time WebSocket updates for metrics
* [ ] Add CIFAR-100 dataset
* [ ] Integrate live matplotlib charts in frontend
* [ ] Add more advanced defenses (e.g., randomized smoothing)

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgements

* [PyTorch](https://pytorch.org/)
* [FastAPI](https://fastapi.tiangolo.com/)
* [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

