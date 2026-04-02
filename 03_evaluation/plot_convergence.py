"""
Grafica curvas de convergencia desde nvlog.json de FastPitch.
Uso: python plot_convergence.py --log /path/to/nvlog.json
"""
import json
import matplotlib.pyplot as plt
import argparse

def plot_convergence(log_path, output_path='convergencia.png'):
    train_mel = {}
    val_mel = {}

    with open(log_path) as f:
        for line in f:
            if '"type": "LOG"' not in line:
                continue
            try:
                data = json.loads(line.replace('DLLL ', ''))
                step = data.get('step')
                if not isinstance(step, list) or len(step) == 0:
                    continue
                epoch = step[0]
                if not isinstance(epoch, int) or epoch < 1 or epoch > 500:
                    continue
                d = data.get('data', {})
                if 'train_avg_mel_loss' in d:
                    train_mel[epoch] = d['train_avg_mel_loss']
                if 'val_mel_loss' in d:
                    val_mel[epoch] = d['val_mel_loss']
            except:
                continue

    epochs_t = sorted(train_mel.keys())
    epochs_v = sorted(val_mel.keys())

    plt.figure(figsize=(12, 5))
    plt.plot(epochs_t, [train_mel[e] for e in epochs_t],
             label='Train Mel Loss', color='blue', linewidth=1.5)
    plt.plot(epochs_v, [val_mel[e] for e in epochs_v],
             label='Val Mel Loss', color='orange', linewidth=1.5)
    plt.xlabel('Época')
    plt.ylabel('Mel Loss (MSE)')
    plt.title('Convergencia del Modelo Base FastPitch — Español (500 épocas)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Guardada: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--output', default='convergencia.png')
    args = parser.parse_args()
    plot_convergence(args.log, args.output)
