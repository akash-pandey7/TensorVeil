from ctgan import CTGAN
import threading
import time
import numpy as np

class TensorVeilGenerator:
    def __init__(self, epochs=750, generator_dim=(256, 256), discriminator_dim=(256, 256), pac=10, batch_size=500):
        self.epochs = epochs
        self.model = CTGAN(epochs=epochs, generator_dim=generator_dim, discriminator_dim=discriminator_dim, pac=pac, batch_size=batch_size, verbose=True)
        self._column_decimals = {}
        self.auto_added_categorical_columns = []

    @staticmethod
    def _infer_decimal_precision(series, max_decimals=6):
        """
        Find the smallest number of decimal places that reproduces this
        column's real values (within float tolerance), capped at max_decimals.
        """
        values = series.dropna().to_numpy(dtype=float)
        if values.size == 0:
            return 2
        for decimals in range(0, max_decimals + 1):
            if np.allclose(values, np.round(values, decimals), rtol=0, atol=1e-9):
                return decimals
        return max_decimals

    def train(self, data, categorical_columns, progress_bar=None, status_text=None):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self._column_decimals = {col: self._infer_decimal_precision(data[col]) for col in numeric_cols}
        non_numeric_cols = set(data.columns) - set(numeric_cols)
        self.auto_added_categorical_columns = sorted(c for c in non_numeric_cols if c not in categorical_columns)
        if self.auto_added_categorical_columns:
            categorical_columns = list(categorical_columns) + self.auto_added_categorical_columns
            print(f"[TensorVeil] Auto-added non-numeric column(s) to categorical_columns before training (would otherwise crash CTGAN): {self.auto_added_categorical_columns}")

        training_done = threading.Event()
        training_error = [None]

        def run():
            try:
                self.model.fit(data, categorical_columns)
            except Exception as e:
                training_error[0] = e
                print(f"[TensorVeil] Training error: {e}")
            finally:
                training_done.set()

        thread = threading.Thread(target=run)
        thread.start()

        if progress_bar and status_text:
            while not training_done.is_set():
                epoch = len(self.model.loss_values) if hasattr(self.model, 'loss_values') and self.model.loss_values is not None else 0
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
            decimals = self._column_decimals.get(col, 2)
            synthetic_data[col] = synthetic_data[col].round(decimals)
        return synthetic_data