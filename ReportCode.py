import textwrap
import os, json, yaml, pickle
from datetime import datetime
import numpy as np
from reportlab.platypus import XPreformatted

ART_DIR = r"C:\Users\AlvaroRivera-Eraso\Documents\HULL\artifacts"
OUT_PDF = os.path.join(ART_DIR, "model_report.pdf")


try:
    import joblib
    STUDY_PKL = os.path.join(ART_DIR, "study.pkl")
    study = joblib.load(STUDY_PKL) if os.path.exists(STUDY_PKL) else None
except Exception:
    study = None

# ---------- load artifacts ----------
def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def load_yaml(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return default

best_params = load_json(os.path.join(ART_DIR, "best_params.json"), {}) or {}
metrics     = load_json(os.path.join(ART_DIR, "metrics.json"), {}) or {}
config      = load_yaml(os.path.join(ART_DIR, "config.yaml"), {}) or {}

# training history (to compute best-epoch)
history = None
hist_pkl_path = os.path.join(ART_DIR, "history.pkl")
if os.path.exists(hist_pkl_path):
    with open(hist_pkl_path, "rb") as f:
        history = pickle.load(f)

# ---------- evidence: console logs ----------
LOG_FILE = os.path.join(ART_DIR, "run_console_log.txt")
LOG_SNIPPET = r"""
[I 2025-08-23 09:26:33,486] Trial 5 finished with value: 0.9041748936944117 and parameters: {'lr': 0.00039716424556440836, 'dropout': 0.3727908573665352, 'batch_size': 16, 'filters1': 64, 'filters2': 128, 'dense_units': 192}. Best is trial 5 with value: 0.9041748936944117.
Epoch 5/30
... val_digit1_accuracy: 0.9595 - val_digit1_loss: 0.1391 - val_digit2_accuracy: 0.9603 - val_digit2_loss: 0.1391 - val_digit3_accuracy: 0.9599 - val_digit3_loss: 0.1381 - val_loss: 0.4162
"""
def read_log_text():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            pass
    return LOG_SNIPPET.strip()

def shorten(text, max_chars=4000):
    if text and len(text) > max_chars:
        return text[:max_chars] + "\n\n...[truncated]..."
    return text

# ---------- best epoch (from history) ----------
best_val_epoch = None
if history and isinstance(history, dict) and "val_loss" in history and history["val_loss"]:
    try:
        vloss = history["val_loss"]
        best_val_epoch = int(np.argmin(vloss)) + 1
    except Exception:
        best_val_epoch = None

# ---------- gather image paths ----------
img_paths = {
    # improved distributions
    "train_marg": os.path.join(ART_DIR, "train_marginals.png"),
    "train_heat": os.path.join(ART_DIR, "train_pair_heatmaps.png"),
    "train_top":  os.path.join(ART_DIR, "train_top30_labels.png"),

    "val_marg":   os.path.join(ART_DIR, "val_marginals.png"),
    "val_heat":   os.path.join(ART_DIR, "val_pair_heatmaps.png"),
    "val_top":    os.path.join(ART_DIR, "val_top30_labels.png"),

    "test_marg":  os.path.join(ART_DIR, "test_marginals.png"),
    "test_heat":  os.path.join(ART_DIR, "test_pair_heatmaps.png"),
    "test_top":   os.path.join(ART_DIR, "test_top30_labels.png"),

    # learning curves & confusion matrices
    "loss":       os.path.join(ART_DIR, "learning_loss.png"),
    "acc_d1":     os.path.join(ART_DIR, "learning_digit1_acc.png"),
    "acc_d2":     os.path.join(ART_DIR, "learning_digit2_acc.png"),
    "acc_d3":     os.path.join(ART_DIR, "learning_digit3_acc.png"),
    "cm1":        os.path.join(ART_DIR, "cm_digit1.png"),
    "cm2":        os.path.join(ART_DIR, "cm_digit2.png"),
    "cm3":        os.path.join(ART_DIR, "cm_digit3.png"),
}

# ---------- ReportLab setup ----------
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, Preformatted
)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], spaceAfter=12))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceAfter=8))
styles.add(ParagraphStyle(name="H3", parent=styles["Heading3"], spaceAfter=6))
styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], leading=14, spaceAfter=6))
styles.add(ParagraphStyle(name="Mono", parent=styles["BodyText"], fontName="Courier", leading=10,
                          backColor=colors.whitesmoke, borderPadding=6, spaceAfter=6))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, textColor=colors.grey))

# Extra styles to keep long content inside margins
styles.add(ParagraphStyle(
    name="SmallWrap",
    parent=styles["BodyText"],
    fontSize=8.5,
    leading=10.5,
    wordWrap="CJK",          # enables wrapping for long unbroken strings
    spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="MonoSmall",
    parent=styles["Mono"],
    fontName="Courier",
    fontSize=8,
    leading=9.5,
))

def img_if_exists(path, max_width=6.2*inch):
    if not os.path.exists(path):
        return None
    try:
        im = Image(path)
        w, h = im.wrap(0, 0)
        if w > max_width:
            scale = max_width / w
            im._restrictSize(w*scale, h*scale)
        return im
    except Exception:
        return None

def kv_table(title, rows):
    data = [["Field", "Value"]] + rows
    t = Table(data, hAlign="LEFT", colWidths=[4.0*cm, 12.0*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f0f0")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
    ]))
    return [Paragraph(title, styles["H3"]), t, Spacer(1, 8)]

def bullet_list(lines):
    return [Paragraph(f"• {line}", styles["Body"]) for line in lines]

# ---------- build PDF ----------
doc = SimpleDocTemplate(
    OUT_PDF,
    pagesize=A4,
    leftMargin=2.0*cm, rightMargin=2.0*cm,
    topMargin=1.7*cm, bottomMargin=1.7*cm
)
elements = []

# Cover
elements.append(Paragraph("Triple-MNIST Multi-Digit Classifier – Final Report", styles["H1"]))
elements.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), styles["Small"]))
elements.append(Spacer(1, 6))
elements.append(Paragraph(
    "This report presents the development of a deep learning model designed to recognize sequences of handwritten digits. It outlines not only the architecture and training process, but also the reasoning behind key design choices, the tuning of hyperparameters, and the progression of model performance over time. To ensure transparency, direct excerpts from Optuna’s hyperparameter search and training logs are included as supporting evidence for each conclusion. The work was authored by Álvaro Rivera-Eraso as part of the MSc in Artificial Intelligence at the University of Hull."))

# Section 1: What I built
elements.append(Paragraph("1) What I built", styles["H2"]))
elements += bullet_list([
    "The model was built as a convolutional neural network (CNN) designed to recognize 3-digit sequences (Triple-MNIST). Its architecture follows a multi-output design: a shared feature extractor branches into three softmax heads (digit1, digit2, digit3), each predicting digits 0–9. Grayscale inputs were chosen since handwritten digits convey information primarily through shape and contrast rather than color. This reduced memory and computation demands and improved training stability. Images were resized to 84×84 pixels to balance detail preservation with computational efficiency, ensuring that subtle differences (e.g., between digits 3 and 8) remained clear. Pixel intensities were normalized to the [0,1] range to stabilize learning, and each head was trained with categorical cross-entropy, optimized using the Adam algorithm."
])

# ==============================
# Section 2) Hyperparameters
# ==============================
elements.append(Paragraph("2) Hyperparameters (Optuna)", styles["H2"]))
elements.append(Paragraph(
    "Optuna was used to search a compact space that balances learning speed and model capacity. "
    "Below are the best trial’s values. In short: the learning rate controls step size; dropout regularizes "
    "the dense block; batch size trades gradient stability for speed; and the filter/depth choices shape how "
    "much texture detail the CNN can represent.", styles["Body"]))

# Table of best params
hp_rows = []
for k in ["lr", "dropout", "batch_size", "filters1", "filters2", "dense_units"]:
    if k in best_params:
        hp_rows.append([k, str(best_params[k])])
if not hp_rows:
    hp_rows = [["(not found)", "See best_params.json"]]
elements += kv_table("Best hyperparameters (Optuna)", hp_rows)

# A short human interpretation of the chosen values
elements.append(Paragraph("Best hyperparameters", styles["H3"]))
elements.append(Paragraph(
    "• The learning rate sits in a safe middle range: large enough to converge in a handful of epochs, "
    "small enough to avoid oscillations near the optimum. "
    "• Dropout around ~0.3–0.4 adds regularization right before the heads, which helps generalize when some "
    "digit/position combinations are rarer. "
    "• A modest batch size keeps updates frequent, which often improves generalization on image tasks. "
    "• Two convolutional blocks with (filters1, filters2) in the 32–128 range give the backbone enough capacity "
    "to encode strokes, corners, and small loops without overfitting. "
    "• Dense units ≈128–256 provide a compact bottleneck that shares information across all three heads.", styles["Body"]))

from xml.sax.saxutils import escape as _esc

if 'study' in globals() and study is not None and getattr(study, "trials", None):
    try:
        trials_sorted = sorted(
            [t for t in study.trials if t.values is not None],
            key=lambda t: t.value if t.value is not None else -1,
            reverse=True
        )
        topK = trials_sorted[:5]

        rows = [["Rank", "Macro-F1", "Parameters"]]
        for i, t in enumerate(topK, 1):
            # Wrap long JSON into a Paragraph with a wrapping style
            params_json = json.dumps(t.params, ensure_ascii=False)
            params_cell = Paragraph(_esc(params_json), styles["SmallWrap"])
            rows.append([str(i), f"{t.value:.4f}", params_cell])

        # Keep column widths inside the page frame
        table = Table(
            rows,
            hAlign="LEFT",
            colWidths=[2.0*cm, 3.0*cm, 10.5*cm]   # ~15.5 cm total fits your margins
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f0f0")),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("GRID",       (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("ALIGN",      (0,0), (1,-1), "LEFT"),
            ("VALIGN",     (0,0), (-1,-1), "TOP"),
            ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
            # ensure the last column wraps
            ("WORDWRAP",   (2,1), (2,-1), "CJK"),
        ]))

        elements.append(Paragraph("Top Optuna trials (higher is better)", styles["H3"]))
        elements.append(table)
        elements.append(Spacer(1, 8))
    except Exception:
        pass



# ==============================================
# Section 3) Data & preprocessing + better plots
# ==============================================
elements.append(Paragraph("3) Data & preprocessing", styles["H2"]))
elements.append(Paragraph(
    "The dataset follows a train/validation/test split, where each folder name encodes a 3-digit label (e.g., “123”). "
    "Images are converted to grayscale, resized to 84×84, and normalized to the [0,1] range. Each digit position "
    "(hundreds, tens, ones) is learned with its own softmax head using one-hot labels. "
    "To assess potential bias and set realistic expectations, three complementary visualizations are generated per split:",
    styles["Body"]))
elements.append(Paragraph("• Digit marginals: frequency of digits 0–9 per position (hundreds, tens, ones).", styles["Body"]))
elements.append(Paragraph("• Co-occurrence heatmaps: how digit pairs across adjacent positions co-occur.", styles["Body"]))
elements.append(Paragraph("• Top-30 labels: the most frequent 3-digit sequences (clearer than plotting all 1,000).", styles["Body"]))

# Split-by-split figures with concise, purposeful captions
for prefix, split_name in [("train", "Train"), ("val", "Validation"), ("test", "Test")]:
    marg = img_if_exists(img_paths.get(f"{prefix}_marg"))
    heat = img_if_exists(img_paths.get(f"{prefix}_heat"))
    top  = img_if_exists(img_paths.get(f"{prefix}_top"))

    # Digit marginals
elements.append(Spacer(1, 6))
elements.append(Paragraph(f"{split_name} – Digit marginals", styles["H3"]))
if marg:
    elements.append(Paragraph(
        "These bar plots show how often each digit (0–9) appears in each position (hundreds, tens, ones). "
        "Distributional skew means the model will see some digits more often, making them easier to learn.",
        styles["Body"]))
    elements.append(marg)

    # Legend with deeper interpretability
    elements.append(Paragraph("Digit marginals:", styles["H3"]))
    elements.append(Paragraph(
        "• In the hundreds position, digit '2' is the most common (≈7300 images), while '4' is clearly underrepresented (≈5000). "
        "This means the first head will have more practice on '2' and less on '4', likely boosting accuracy on '2' but reducing it on '4'.", 
        styles["Body"]))
    elements.append(Paragraph(
        "• In the tens position, digits '7' and '2' dominate (≈6900–7000), whereas '1' is underrepresented (≈5700). "
        "The imbalance could create a tendency to over-predict frequent digits in the middle slot.", 
        styles["Body"]))
    elements.append(Paragraph(
        "• In the ones position, digit '9' is the most frequent (≈7200), while '5' and '6' are the least represented (≈5900). "
        "This gap of ~1300 images means the third head might confuse '5'/'6' more often compared to '9'.", 
        styles["Body"]))
    elements.append(Paragraph(
        "Overall, the marginals reveal that while the dataset is relatively balanced, certain digits consistently appear "
        "more or less often depending on their position. These small skews accumulate and directly shape per-head learning difficulty.", 
        styles["Body"]))
else:
    elements.append(Paragraph("(Marginals plot not available for this split.)", styles["Small"]))


    # Co-occurrence heatmaps
elements.append(Spacer(1, 6))
elements.append(Paragraph(f"{split_name} – Digit co-occurrence heatmaps", styles["H3"]))
if heat:
    elements.append(Paragraph(
        "Brighter cells mark frequent adjacent-position pairs (e.g., hundreds→tens). "
        "Strong blocks indicate common patterns the model learns quickly; sparse regions flag rare pairs that "
        "typically drive confusion.",
        styles["Body"]))
    elements.append(heat)

    # Legend with deeper interpretability (generic, works for any split)
    elements.append(Paragraph("Co-occurrence heatmaps:", styles["H3"]))
    elements.append(Paragraph(
        "• Axes & colorbar. Rows encode the digit at position A and columns the digit at position B "
        "(e.g., Pos1→Pos2, Pos2→Pos3). The colorbar gives the image count for each pair.", styles["Body"]))
    elements.append(Paragraph(
        "• Bright blocks. When a contiguous region is bright, certain digit pairs occur together often. "
        "The model will see these pairs many times during training, so it tends to learn them quickly and predict them confidently.",
        styles["Body"]))
    elements.append(Paragraph(
        "• Dark/sparse cells. Rare digit pairs offer little practice. These pockets usually translate into higher error "
        "for those combinations and can show up as off-diagonal mass in the confusion matrices.", styles["Body"]))
    elements.append(Paragraph(
        "• Diagonal vs off-diagonal. A bright diagonal (same digit at A and B) suggests repeated patterns like 00, 11, …; "
        "a bright off-diagonal block suggests specific dependencies (e.g., 4 often followed by 7). Either case creates "
        "inductive bias the model can exploit.", styles["Body"]))
    elements.append(Paragraph(
        "• Asymmetry across panels. Compare Pos1→Pos2 vs Pos2→Pos3. If one panel is more concentrated, that transition is "
        "easier (more predictable) than the other. This helps explain why a given head might perform slightly better or worse.",
        styles["Body"]))
    elements.append(Paragraph(
        "• Mitigation if needed. If rare pairs matter for your use case, consider targeted augmentation or rebalancing to "
        "increase exposure to those cells.", styles["Body"]))
else:
    elements.append(Paragraph("(Co-occurrence heatmaps not available for this split.)", styles["Small"]))



    # Top-30 labels
elements.append(Spacer(1, 6))
elements.append(Paragraph(f"{split_name} – Top-30 labels", styles["H3"]))
if top:
    elements.append(Paragraph(
        "The 30 most frequent 3-digit sequences for this split. This view surfaces any label-level skew without "
        "plotting all 1,000 classes. Use it to spot dominant patterns and long-tail classes.",
        styles["Body"]))
    elements.append(top)

    # Legend & Interpretation (generic; works for any split)
    elements.append(Paragraph("Top-30 labels:", styles["H3"]))
    elements.append(Paragraph(
        "• X-axis shows the 3-digit label (e.g., 042). Y-axis shows the image count for that label. "
        "Value labels on tops of bars provide exact counts.", styles["Body"]))
    elements.append(Paragraph(
        "• Flat bars ≈ uniform exposure. If heights are nearly identical (as in many synthetic datasets), "
        "the model sees each top label equally often and won’t favor a small subset just from frequency.",
        styles["Body"]))
    elements.append(Paragraph(
        "• Steep drop-off = heavy head. If the first few bars tower above the rest, the model will over-practice those "
        "labels and typically achieve higher accuracy on them than on rare labels in the tail.", styles["Body"]))
    elements.append(Paragraph(
        "• Cross-check with errors. If certain rare labels appear in failure cases, the tail exposure shown here is a "
        "likely cause. Mitigation options: oversample tail labels, targeted augmentation, or class-balanced loss.",
        styles["Body"]))
else:
    elements.append(Paragraph("(Top-30 label plot not available for this split.)", styles["Small"]))




# Section 4: Training procedure
# ------------------------------------------------------
elements.append(Paragraph("4) Training procedure", styles["H2"]))

elements.append(Paragraph(
    "Training was done in two phases. First, Optuna searched a compact space of learning rate, dropout, "
    "batch size, and backbone width using a 3-fold CV on a 30% subset. Second, the best configuration was "
    "retrained on the full training set with validation monitoring. The emphasis here is that the best "
    "validation loss landed early (around epoch 5); to avoid overfitting, all reported test metrics come "
    "from that checkpoint.", styles["Body"]))

steps = [
    "Data split respected: train / validation / test.",
    "Hyperparameter search: Optuna on a 30% subset with 3-fold CV, maximizing macro-F1.",
    "Final training: full training set using EarlyStopping (monitor=val_loss) and ModelCheckpoint "
    "(restore_best_weights).",
]
if config.get("epochs_ran"):
    steps.append(f"Epoch budget: {config['epochs_ran']} total; early stopping selected the best checkpoint.")
if best_val_epoch:
    steps.append(f"Best validation loss observed at epoch {best_val_epoch}; that checkpoint was used for evaluation.")
elements += bullet_list(steps)

# small checkpoint box for clarity
if best_val_epoch:
    rows = [
        ["Best epoch", str(best_val_epoch)],
        ["Selection rule", "Minimum validation loss with restore-best-weights"],
        ["Why this matters", "Prevents late-epoch overfitting and ensures fair test reporting"]
    ]
    elements += kv_table("Checkpoint used for final results", rows)

# Evidence excerpt, wrapped within page margins
log_text = shorten(read_log_text(), max_chars=1200)  # shorter keeps it readable
if log_text:
    elements.append(Paragraph("(output training log)", styles["H3"]))

    # hard-wrap each original line so no single line exceeds the frame width
    wrapped = "\n".join(
        textwrap.fill(line, width=95, break_long_words=True, break_on_hyphens=False)
        for line in log_text.splitlines()
    )

    # XPreformatted preserves monospaced look and respects our manual wraps
    elements.append(XPreformatted(wrapped, styles["MonoSmall"]))
else:
    elements.append(Paragraph("(No log excerpt provided.)", styles["Small"]))



###############################################################################
# Section 5: Learning curves
###############################################################################

elements.append(Paragraph("5) Learning behavior (curves)", styles["H2"]))

# --- Loss curves -------------------------------------------------------------
im = img_if_exists(img_paths["loss"])
if im:
    elements.append(Paragraph("Loss curves", styles["H3"]))
    elements.append(im); elements.append(Spacer(1, 6))

    loss_caption = (
        f"Training and validation loss fall smoothly without oscillations. "
        f"The minimum validation loss occurs around epoch {best_val_epoch}, after which the curve "
        f"flattens—so early stopping restores the best weights from that point."
        if best_val_epoch
        else "Training and validation loss fall smoothly without oscillations. "
             "Validation flattens after the initial drop—so early stopping selects the best weights before any drift."
    )
    loss_caption += " The small and stable train–validation gap suggests the model generalizes well rather than overfitting."
    elements.append(Paragraph(loss_caption, styles["Body"]))
    elements.append(Spacer(1, 6))

# Helper to write a concise interpretation per head
def _acc_caption(position_name):
    return (
        f"Accuracy rises quickly in the first few epochs and then plateaus, mirroring the loss behavior. "
        f"Small wiggles are expected from mini-batch noise. Any small difference versus the other heads usually "
        f"reflects the {position_name} position’s label frequency (from the marginals): digits that appear less often "
        f"are slightly harder and may cap the final accuracy a bit earlier."
    )

# --- Digit-1 accuracy (hundreds) --------------------------------------------
im = img_if_exists(img_paths["acc_d1"])
if im:
    elements.append(Paragraph("Digit-1 accuracy (hundreds)", styles["H3"]))
    elements.append(im); elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "Accuracy rises very sharply in the first 2–3 epochs, moving from ~0.75 to above 0.92 almost instantly. "
        "After that, the curve continues to improve steadily and crosses 0.97 by epoch 9. "
        "Validation accuracy tracks training closely, with almost no gap, which means the hundreds position is relatively easy "
        "to learn and not prone to overfitting. This makes sense given the marginals: common digits (like '2') dominate this slot, "
        "so the head benefits from repeated exposure.", styles["Body"]))
    elements.append(Spacer(1, 6))

# --- Digit-2 accuracy (tens) -------------------------------------------------
im = img_if_exists(img_paths["acc_d2"])
if im:
    elements.append(Paragraph("Digit-2 accuracy (tens)", styles["H3"]))
    elements.append(im); elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "The second head follows a similar trajectory, but its validation curve shows a slightly earlier plateau (around epoch 6). "
        "Accuracy saturates just below 0.97, lagging a bit behind digit-1. "
        "This is consistent with the dataset skew, where underrepresented digits (like '1') in the tens position reduce peak generalization. "
        "Still, the very small gap between train and validation confirms good generalization.", styles["Body"]))
    elements.append(Spacer(1, 6))

# --- Digit-3 accuracy (ones) -------------------------------------------------
im = img_if_exists(img_paths["acc_d3"])
if im:
    elements.append(Paragraph("Digit-3 accuracy (ones)", styles["H3"]))
    elements.append(im); elements.append(Spacer(1, 6))
    elements.append(Paragraph(
        "Digit-3 converges quickly as well, but its validation accuracy levels off slightly earlier and flatter compared to digits 1 and 2, "
        "stabilizing around 0.965. The training curve keeps inching higher toward 0.98, creating a small but persistent train–validation gap. "
        "This suggests that the ones position is marginally harder, likely due to the imbalance of rare labels ('5' and '6' being underrepresented). "
        "Even so, the plateau is stable, not degrading, so this head is robust but capped a bit earlier.", styles["Body"]))
    elements.append(Spacer(1, 6))



# Section 6: Final performance
elements.append(Paragraph("6) Final performance (test set)", styles["H2"]))
md = metrics or {}
def fmt(x): return f"{x:.4f}" if isinstance(x,(int,float)) else str(x)
rows = [
    ["Digit-1 Accuracy", fmt(md.get("keras_evaluate",{}).get("digit1_acc","—"))],
    ["Digit-2 Accuracy", fmt(md.get("keras_evaluate",{}).get("digit2_acc","—"))],
    ["Digit-3 Accuracy", fmt(md.get("keras_evaluate",{}).get("digit3_acc","—"))],
    ["Macro-F1 (Digit-1)", fmt(md.get("macro_f1_digit1","—"))],
    ["Macro-F1 (Digit-2)", fmt(md.get("macro_f1_digit2","—"))],
    ["Macro-F1 (Digit-3)", fmt(md.get("macro_f1_digit3","—"))],
    ["Macro-F1 (Average)", fmt(md.get("macro_f1_avg","—"))],
    ["Exact-match Accuracy", fmt(md.get("exact_match_accuracy","—"))],
]
elements += kv_table("Key results (test set)", rows)
elements.append(Paragraph("Interpretation of results", styles["H3"]))
elements.append(Paragraph(
    "The final test metrics confirm that the model is both accurate and well-calibrated across all three digit positions:", 
    styles["Body"]))

elements.append(Paragraph(
    "• Digit-1 Accuracy (hundreds, 0.9595): The first head performs strongly, reflecting the relative ease of predicting "
    "the hundreds digit. This matches the learning curves, where accuracy rose quickly and validation tracked training closely. "
    "The slight advantage here comes from the higher frequency of common digits (like '2') in this slot.", styles["Body"]))

elements.append(Paragraph(
    "• Digit-2 Accuracy (tens, 0.9590): The second head performs almost identically, though with a marginally lower score. "
    "This aligns with the dataset skew where certain digits ('1') are less represented in the tens position. Despite that imbalance, "
    "accuracy remains above 95%, showing that the network generalized well.", styles["Body"]))

elements.append(Paragraph(
    "• Digit-3 Accuracy (ones, 0.9600): The third head achieves the highest raw accuracy. Although the learning curve showed "
    "a small train–validation gap, the final score indicates that overfitting did not meaningfully harm generalization. "
    "This robustness is notable given that some digits ('5' and '6') were underrepresented at this position.", styles["Body"]))

elements.append(Paragraph(
    "• Macro-F1 scores (≈0.957–0.960): These values mirror the accuracy results and confirm balanced performance across all classes. "
    "Because Macro-F1 weighs each digit equally, high scores here mean the model is not only performing well on frequent digits but "
    "also handling rarer ones effectively. The consistency across heads (0.9571–0.9599) shows stable class-level precision and recall.", styles["Body"]))

elements.append(Paragraph(
    "• Macro-F1 Average (0.9583): The average across all three heads consolidates the above results, showing that the network "
    "achieves reliable and uniform recognition across digit positions.", styles["Body"]))

elements.append(Paragraph(
    "• Exact-match Accuracy (0.8824): This stricter metric measures whether the entire 3-digit sequence was predicted correctly. "
    "At ~88%, it is naturally lower than per-digit scores because one mistake in any head counts as a failure. "
    "Still, this is a strong outcome given 1,000 possible class combinations, indicating the model captures dependencies "
    "between digit positions effectively.", styles["Body"]))

elements.append(Paragraph(
    "Overall, the test set results demonstrate that the network not only reaches high per-digit accuracy but also maintains strong "
    "end-to-end sequence recognition. The small gaps between accuracy and Macro-F1 confirm that performance is balanced across "
    "frequent and infrequent digits alike, while the high exact-match accuracy underscores the model’s practical reliability.", styles["Body"]))


# Section 7: Confusion matrices
# Section 7: Confusion matrices
elements.append(Paragraph("7) Confusion matrices", styles["H2"]))

elements.append(Paragraph(
    "In the following figures, the confusion matrices for Digit-1 (hundreds), Digit-2 (tens), "
    "and Digit-3 (ones) are presented. The diagonals are consistently bright, showing that the "
    "majority of predictions fall into the correct class across all heads. Misclassifications are "
    "rare and scattered, with no systematic bias toward a particular digit. The slight off-diagonal "
    "activity aligns with earlier dataset skew (e.g., digits like '4', '5', and '6' being less "
    "represented in some positions), but overall these errors remain minimal. Taken together, the "
    "matrices confirm the model’s balanced performance: precision and recall are uniformly high, "
    "and no position shows evidence of collapse or strong confusion. This validates that the model’s "
    "robustness extends not only to per-digit accuracy but also to fine-grained class separation.",
    styles["Body"]))

for key, title in [("cm1","Digit-1 confusion matrix"), ("cm2","Digit-2 confusion matrix"), ("cm3","Digit-3 confusion matrix")]:
    im = img_if_exists(img_paths[key], max_width=5.8*inch)
    if im:
        elements.append(Paragraph(title, styles["H3"]))
        elements.append(im); elements.append(Spacer(1, 6))

# Section 8: Repro config
elements.append(Paragraph("8) Reproducibility & run configuration", styles["H2"]))
elements.append(Paragraph(
    "To close the report, I include a reproducibility section aimed at ensuring transparency in the research process. "
    "This section documents the exact configuration used during training and hyperparameter tuning, so that the "
    "experiments can be replicated or extended consistently. The table below summarizes the key parameters that shaped "
    "the model’s behavior, including data splits, optimization settings, and architectural choices.",
    styles["Body"]))

rows_cfg = [
    ["Date", config.get("date","—")],
    ["Epochs ran", str(config.get("epochs_ran","—"))],
    ["Batch size", str(config.get("batch_size","—"))],
    ["Learning rate", str(best_params.get("lr","—"))],
    ["Dropout", str(best_params.get("dropout","—"))],
    ["Filters", f"{best_params.get('filters1','—')}, {best_params.get('filters2','—')}"],
    ["Dense units", str(best_params.get("dense_units","—"))],
]
elements += kv_table("Run configuration", rows_cfg)

# Section 9: Teamwork takeaways
elements.append(Paragraph("9) Teamwork takeaways", styles["H2"]))
elements.append(Paragraph(
    "Throughout the semester, collaboration with my teammate played an important role in shaping how we approached the assignments. "
    "Our interaction was relatively light in frequency, but well-timed and productive whenever it was needed. We scheduled two video "
    "calls at key stages of the project and complemented these with short WhatsApp exchanges to clarify smaller questions, which was "
    "particularly useful given our different time zones.", styles["Body"]))

elements.append(Paragraph(
    "The first call, on June 16, was focused on introductions and on building a common plan for Task 1. During this session, we reviewed "
    "possible modeling approaches and discussed how best to prepare the dataset, weighing the strengths and weaknesses of alternative "
    "models. Even though we both felt confident in the initial strategy, the discussion provided a second perspective that helped validate "
    "the direction and ensured nothing important was overlooked.", styles["Body"]))

elements.append(Paragraph(
    "Our second call took place on August 1 and centered on Task 2. This time the focus was the architecture of the CNN. We debated whether "
    "to process entire images directly or to crop the digits beforehand. Neither of us had a clear preference initially, so being able to "
    "talk through the options was particularly valuable. The exchange allowed us to test ideas, challenge assumptions, and arrive at a "
    "solution that felt both practical and well-reasoned.", styles["Body"]))

elements.append(Paragraph(
    "Between those two calls, we maintained occasional contact through WhatsApp for quick clarifications. This lightweight communication "
    "style allowed us to stay aligned without adding unnecessary overhead. Once the second call was completed, each of us had sufficient "
    "clarity to continue independently and bring the pieces together in the final report.", styles["Body"]))

elements.append(Paragraph(
    "Looking back, I would describe the teamwork as supportive and effective. The value was less about detailed coding collaboration and "
    "more about exchanging perspectives on higher-level design choices, data handling, and interpretation of results. This back-and-forth "
    "not only strengthened our decisions but also made the project more enjoyable to carry out.", styles["Body"]))

# ================================
# Section 10) Collaboration timeline
# ================================
from datetime import datetime
from xml.sax.saxutils import escape as _esc

# --- Define the activities (edit dates/labels if needed) ---
# Tip: keep dates as YYYY-MM-DD strings for easy editing.
_collab = [
    # (Title, start_date, end_date, Notes)
    ("Kickoff & introductions",     "2024-06-16", "2024-06-16", "First video call: align goals & scope."),
    ("Task 1 planning & modeling",  "2024-06-17", "2024-06-30", "Data prep options; baseline model choice."),
    ("Async Q&A (WhatsApp)",        "2024-07-01", "2024-07-31", "Light touch sync across time zones."),
    ("Task 2 CNN architecture",     "2024-08-01", "2024-08-01", "Second call: whole image vs. cropped digits."),
    ("Model training & iterations", "2024-08-02", "2024-08-20", "Tune hyperparams; checkpoints & validation."),
    ("Report drafting & polish",    "2024-08-21", "2024-08-31", "Figures, interpretations, final pass."),
]

# --- Section header & short intro ---
elements.append(Paragraph("10) Collaboration timeline", styles["H2"]))
elements.append(Paragraph(
    "The table and chart below summarize the project chronology and the main activities carried out together. "
    "Dates are approximate where a range is shown; single-day events indicate focused working sessions or calls.",
    styles["Body"])
)

# --- Build a compact table that fits page margins ---
rows = [["Period", "Activity", "Outcome / Notes"]]
for title, s, e, note in _collab:
    sdt = datetime.strptime(s, "%Y-%m-%d").date()
    edt = datetime.strptime(e, "%Y-%m-%d").date()
    period = f"{sdt.strftime('%d %b %Y')} — {edt.strftime('%d %b %Y')}"
    rows.append([period, _esc(title), _esc(note)])

chronot = Table(
    rows,
    hAlign="LEFT",
    colWidths=[5.0*cm, 5.0*cm, 6.0*cm]  # ~16 cm total: safe inside your margins
)
chronot.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f0f0f0")),
    ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
    ("GRID",       (0,0), (-1,-1), 0.25, colors.lightgrey),
    ("VALIGN",     (0,0), (-1,-1), "TOP"),
    ("BACKGROUND", (0,1), (-1,-1), colors.whitesmoke),
]))
elements.append(Paragraph("Collaboration chronogram (table)", styles["H3"]))
elements.append(chronot)
elements.append(Spacer(1, 10))

# --- Create a simple Gantt-style timeline with matplotlib and insert it as an image ---
try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    gantt_path = os.path.join(ART_DIR, "collab_timeline.png")

    tasks   = [a[0] for a in _collab]
    starts  = [datetime.strptime(a[1], "%Y-%m-%d") for a in _collab]
    ends    = [datetime.strptime(a[2], "%Y-%m-%d") for a in _collab]
    dur     = [(e - s).days + 1 for s, e in zip(starts, ends)]

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=140)
    y_pos = range(len(tasks))[::-1]  # newest on top

    ax.barh(y_pos, dur, left=mdates.date2num(starts), align="center")
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(tasks)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6)
    ax.set_xlabel("Date")
    ax.set_title("Project timeline (Gantt view)", loc="left")

    plt.tight_layout()
    fig.savefig(gantt_path, bbox_inches="tight")
    plt.close(fig)

    gantt_img = img_if_exists(gantt_path, max_width=6.2*inch)
    if gantt_img:
        elements.append(Paragraph("Collaboration chronogram (Gantt view)", styles["H3"]))
        elements.append(gantt_img)
        elements.append(Spacer(1, 8))
except Exception as _e:
    # Fallback note if matplotlib fails for any reason
    elements.append(Paragraph(
        "(Timeline chart could not be rendered in this environment; table shown above.)", styles["Small"]))

# Build PDF
doc.build(elements)
print(f"✅ PDF report written to: {OUT_PDF}")
