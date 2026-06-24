from ctgan import CTGAN
import pandas as pd
import threading
import time
import sys
import io

class TensorVeilGenerator:
    def __init__(self, epochs=50):
        self.epochs = epochs
        self.model = CTGAN(epochs=epochs, verbose=True)

    def train(self, data, categorical_columns, progress_bar=None, status_text=None):
        training_done = threading.Event()
        training_error = [None]
        current_epoch = [0]

        class EpochTracker(io.TextIOBase):
            def write(self, text):
                if "Epoch" in text or "epoch" in text:
                    current_epoch[0] += 1
                sys.__stdout__.write(text)
                return len(text)

        def run():
            try:
                old_stdout = sys.stdout
                sys.stdout = EpochTracker()
                self.model.fit(data, categorical_columns)
                sys.stdout = old_stdout
            except Exception as e:
                training_error[0] = e
                print(f"[TensorVeil] Training error: {e}")
            finally:
                training_done.set()

        thread = threading.Thread(target=run)
        thread.start()

        if progress_bar and status_text:
            while not training_done.is_set():
                epoch = current_epoch[0]
                real_progress = min(epoch / self.epochs, 0.99)
                progress_bar.progress(real_progress)
                status_text.text(f"Training... Epoch {epoch}/{self.epochs}")
                time.sleep(0.5)

        thread.join()

        if training_error[0] is not None:
            raise training_error[0]

        if progress_bar and status_text:
            progress_bar.progress(1.0)
            status_text.text("Training Complete!")
            if not self.model.loss_values.empty:
                last = self.model.loss_values.iloc[-1]
                g_loss = last['Generator Loss']
                d_loss = last['Discriminator Loss']
                status_text.text(
                    f"✅ Training complete — "
                    f"G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}"
                )

    def get_loss_history(self):
        return self.model.loss_values

    def generate(self, count):
        synthetic_data = self.model.sample(count)
        numeric_cols = synthetic_data.select_dtypes(include=['float']).columns
        for col in numeric_cols:
            synthetic_data[col] = synthetic_data[col].round(2)
        return synthetic_data