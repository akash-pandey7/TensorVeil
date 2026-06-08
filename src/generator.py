from ctgan import CTGAN
from joblib import parallel_backend
import pandas as pd

class TensorVeilGenerator:
    def __init__(self, epochs=50):
        self.epochs = epochs
        self.model = CTGAN(epochs=epochs, verbose=False)

    def train(self, data, categorical_columns, progress_bar=None, status_text=None):
        import threading
        import time

        training_done = threading.Event()
        training_error = [None]

        def run():
            try:
                with parallel_backend('threading'):
                    self.model.fit(data, categorical_columns)
            except Exception as e:
                training_error[0] = e
            finally:
                training_done.set()

        thread = threading.Thread(target=run)
        thread.start()

        if progress_bar and status_text:
            estimated_seconds = self.epochs * 0.8
            elapsed = 0
            interval = 0.5

            while not training_done.is_set():
                elapsed += interval
                fake_progress = min(elapsed / estimated_seconds, 0.95)
                progress_bar.progress(fake_progress)
                status_text.text(
                    f"Training... (~{max(0, int(estimated_seconds - elapsed))}s remaining)"
                )
                time.sleep(interval)

        thread.join()

        if training_error[0] is not None:
            raise training_error[0]

        if progress_bar and status_text:
            progress_bar.progress(1.0)
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