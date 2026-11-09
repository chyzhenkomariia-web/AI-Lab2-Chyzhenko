import gradio as gr
import numpy as np

GRID_N = 25  


def get_vecs(img):
    if img is None:
        return np.zeros(GRID_N), np.zeros(GRID_N)
    pixels = np.array(img.convert("L").resize((5, 5)))
    abs_v = (pixels < 128).astype(float).flatten()
    s = np.sum(abs_v)
    return abs_v, (abs_v / s if s > 0 else np.zeros(GRID_N))


# ПЕРЦЕПТРОН
w_perceptron = np.zeros(GRID_N)
bias_p = 0.0


def perceptron_step(img, target, eta, init_w):
    global w_perceptron, bias_p
    if img is None:
        return "Завантажте фото", "", ""
    if np.all(w_perceptron == 0) and init_w != 0:
        w_perceptron = np.full(GRID_N, float(init_w))

    _, norm_v = get_vecs(img)
    net = np.dot(w_perceptron, norm_v) + bias_p
    pred = 1 if net >= 0 else 0
    err = int(target) - pred

    if err != 0:
        w_perceptron += float(eta) * err * norm_v
        bias_p += float(eta) * err
        status = f"Помилка! Ваги оновлено. Target={target}, Pred={pred}"
    else:
        status = f"Вірно! (Pred={pred})"

    return (
        f"Клас: {pred} (net={net:.3f})",
        status,
        str(np.round(w_perceptron, 2).tolist()),
    )


# ХОПФІЛД ТА ХЕММІНГ
def hopfield_hamming(r1, r2, r3, test):
    if any(x is None for x in [r1, r2, r3, test]):
        return "Завантажте 4 фото", "", ""

    refs_abs = [get_vecs(r)[0] for r in [r1, r2, r3]]
    refs_norm = [get_vecs(r)[1] for r in [r1, r2, r3]]
    test_abs, test_norm = get_vecs(test)

    # Хопфілд (бінар,Хебба)
    refs_bip = [np.where(a >= np.median(a), 1, -1) for a in refs_abs]
    W_hop = sum(np.outer(b, b) for b in refs_bip)
    np.fill_diagonal(W_hop, 0)

    test_bip = np.where(test_abs >= np.median(test_abs), 1, -1)
    state = test_bip.copy()
    for _ in range(3):
        state = np.sign(np.dot(W_hop, state))
    hop_class = np.argmax([np.dot(state, b) for b in refs_bip]) + 1

    # Хеммінг (скаляр, maxnet)
    W_ham = np.array(refs_norm)
    ham_outputs = np.dot(W_ham, test_norm)
    ham_class = np.argmax(ham_outputs) + 1

    return (
        f"Хопфілд: КЛАС {hop_class}\nХеммінг: КЛАС {ham_class}",
        f"Матриця Хопфілда (25x25):\n{W_hop[:4,:4]}...",
        f"Матриця Хеммінга (3x25):\n{np.round(W_ham, 2)}",
    )



with gr.Blocks(title="Лаб 2 ШІ") as app:
    gr.Markdown("## Завдання 1. Перцептрон Розенблатта")
    with gr.Row():
        p_img = gr.Image(type="pil", label="Образ", sources=["upload"])
        with gr.Column():
            p_w = gr.Number(value=0, label="Init W")
            p_eta = gr.Number(value=0.1, label="Eta (крок)")
            p_target = gr.Radio([0, 1], label="Учитель (Target)", value=1)
            btn_p = gr.Button("Класифікувати та навчити", variant="primary")
    out_p1, out_p2, out_p3 = (
        gr.Textbox(label="Результат"),
        gr.Textbox(label="Стан"),
        gr.Textbox(label="Ваги W"),
    )
    btn_p.click(
        perceptron_step,
        [p_img, p_target, p_eta, p_w],
        [out_p1, out_p2, out_p3],
    )

    gr.Markdown("## Завдання 2. Хопфілд та Хеммінг")
    with gr.Row():
        r1 = gr.Image(type="pil", label="Еталон 1", sources=["upload"])
        r2 = gr.Image(type="pil", label="Еталон 2", sources=["upload"])
        r3 = gr.Image(type="pil", label="Еталон 3", sources=["upload"])
    t_img = gr.Image(type="pil", label="Тестовий образ", sources=["upload"])
    btn_h = gr.Button("Класифікувати", variant="primary")
    o_h1, o_h2, o_h3 = (
        gr.Textbox(label="Висновок"),
        gr.Textbox(label="Матриця Хопфілда"),
        gr.Textbox(label="Матриця Хеммінга"),
    )
    btn_h.click(hopfield_hamming, [r1, r2, r3, t_img], [o_h1, o_h2, o_h3])

if __name__ == "__main__":
    app.launch()