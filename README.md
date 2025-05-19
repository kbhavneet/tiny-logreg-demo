# Tiny Logistic-Regression Demo📈🤖

A **five-minute, first-principles exercise** that demystifies model training:

* We synthesise two little clusters of 2-D points  
  * **Class 0** – roughly around **(0, 0)**  
  * **Class 1** – roughly around **(5, 5)**
* A streaming (online) **logistic regression** watches the data 10 times
* After each pass it prints accuracy and at the end, plots the learning curve

The aim is to _see_ the model adjust its weights and minimise errors - no magic, just gradient steps!

---

## Quick start

```bash
# clone the repo you just pushed
git clone https://github.com/kbhavneet/tiny-logreg-demo.git
cd tiny-logreg-demo

# set up a clean env
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# install the exact packages I used
pip install -r requirements.txt

# run it
python train_simple_model.py
```
---

## 🔄 Re‑using this template

Swap in any **2‑D** dataset and watch the loop adapt.

* **XOR‑style spiral** data to make linear models struggle.
* **Two‑moons** or **concentric circles** via `sklearn.datasets.make_moons` / `make_circles`.
* Inject **Gaussian noise** to see accuracy fluctuate.

The structure of `train_simple_model.py` stays identical – only the `X`, `y` creation block changes.

---


