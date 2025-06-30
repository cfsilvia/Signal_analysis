import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

# === Load Signal ===
def load_signal(filepath, column_name='Left_side'):
    df = pd.read_excel(filepath)
    signal = df[column_name].values
    timestamps = df.index.values  # change to df['Timestamp'].values if you have a timestamp column
    return signal, timestamps

# === Initialize Plot ===
def init_plot(signal):
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(signal, label="Signal")
    ax.set_title("Click twice to define region. Press number key to assign label.")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig, ax

# === Event Handlers ===
def onclick(event):
    if event.inaxes != ax or event.button != 1:
        return
    x = int(event.xdata)
    clicks.append(x)
    print(f"Clicked at index: {x}")
    v = ax.axvline(x=x, color='red', linestyle='--')
    vlines.append(v)
    fig.canvas.draw()
    
def delete_last_region():
    if not motif_regions:
        print("Nothing to delete.")
        return

    # Remove region from history
    start, end, label = motif_regions.pop()
    labels[start:end] = 0
    print(f"Deleted region {start}–{end} (label {label})")

    # Remove last 2 vertical lines and label text
    for _ in range(2):  # remove start and end lines
        if vlines:
            v = vlines.pop()
            v.remove()
    if texts:
        t = texts.pop()
        t.remove()

    fig.canvas.draw()


def onkey(event):
    if event.key.isdigit() and len(clicks) >= 2:
        start, end = sorted(clicks[-2:])
        label = int(event.key)
        labels[start:end] = label
        motif_regions.append((start, end, label))
        print(f"Region {start}–{end} labeled as {label}")

        # Draw label
        t = ax.text((start + end)//2, np.max(signal)*0.95, f"{label}",
                    color='blue', ha='center')
        texts.append(t)

        # Clear old lines
        for line in vlines:
            line.remove()
        vlines.clear()
        clicks.clear()
        fig.canvas.draw()

    elif event.key == 'd':
        delete_last_region()

# === Save Outputs ===
def save_labels_np(labels, out_path="labels.npy"):
    np.save(out_path, labels)
    print(f"Saved labels to {out_path}")

def save_regions_csv(motif_regions, signal, timestamps, out_path="motif_regions_detailed.csv"):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Region", "Index", "Timestamp", "SignalValue", "Label"])
        region_id = 1
        for start, end, label in motif_regions:
            for i in range(start, end):
                writer.writerow([region_id, i, timestamps[i], signal[i], label])
            region_id += 1
    print(f"Saved region details to {out_path}")

# === Main Program ===
if __name__ == "__main__":
    # Load data
    signal, timestamps = load_signal("F:/BlindMole_tracking_Juna/2025/BMR10/BMR10/output/BMR10_with_landmarks_left_ToPlot.xlsx")

    # Init labels and state
    labels = np.zeros_like(signal, dtype=int)
    motif_regions = []
    clicks = []
    vlines = []
    texts = []

    # Init plot
    fig, ax = init_plot(signal)

    # Connect interactivity
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)

    # Show plot
    plt.show()

    # Save results
    save_labels_np(labels)
    save_regions_csv(motif_regions, signal, timestamps)
