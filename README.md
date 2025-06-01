# Pure‑Loop Neural Networks  🧠

> **Tiny educational repo** showing how a two‑layer neural network can be trained **without any NumPy vectorisation**. Every multiply–add is hand‑written with `for`‑loops so you can follow the maths step‑by‑step.
> Two training variants are provided:
>
> * `loop_based_neural_network.py`  – vanilla SGD (mean‑squared error)
> * `loop_based_neural_network_adam.py` – the same model, but updated with an explicit **Adam** optimiser written in loops

---

## File layout

```
.
├── loop_based_neural_network.py      # SGD trainer (loops only)
├── loop_based_neural_network_adam.py # Adam trainer (loops only)
├── README.md                         
├── requirements.txt                                 
```

### requirements.txt

```
numpy>=1.25
matplotlib>=3.8
```

## Quick start

```bash
# 1. Clone the repo
$ git clone https://github.com/<your‑user>/pure‑loop‑nn.git
$ cd pure‑loop‑nn

# 2. Create a virtual environment (optional but recommended)
$ python -m venv .venv && source .venv/bin/activate  # on Windows use .venv\Scripts\activate

# 3. Install deps
$ pip install -r requirements.txt

# 4. Run either script
$ python loop_based_neural_network.py          # SGD version
$ python loop_based_neural_network_adam.py     # Adam version
```

Each script trains on a tiny dummy dataset (three 3‑D samples → two‑class labels), prints the epoch‑wise loss and pops up a live Matplotlib window plotting the loss curve.

---

## What’s inside?

* **No vectorisation** – every `dot` product is expanded into nested loops so you can see exactly where each gradient term comes from.
* **ReLU + Sigmoid** activations (scalar versions).
* **SGD / Adam** weight updates written “by hand” – a good way to demystify Adam.
* **LivePlotter** helper class that streams loss vs. epoch in real time.
* **Model save/load** via `pickle` so you can checkpoint your weights.

---

## Extending the examples

* Swap in a larger dataset and watch the loss curve evolve.
* Change `hidden_size` or add more hidden layers.
* Replace MSE with a different loss (e.g. binary cross‑entropy) – remember to adjust output activation / delta formula.
* Port the logic to a microcontroller or any environment where NumPy isn’t available.

---

## License

MIT – do whatever you like, just keep the notice. 😊
