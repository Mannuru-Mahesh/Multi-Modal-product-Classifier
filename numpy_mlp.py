"""
numpy_mlp.py
--------------
A MULTI-MODAL NEURAL NETWORK implemented from scratch with NumPy
(no PyTorch/TensorFlow needed). This is the "deep learning"
architecture at the heart of this project.

ARCHITECTURE (mirrors real multi-modal models like CLIP, but with
hand-engineered features instead of learned CNN/transformer
embeddings):

    image_features ──> [Image Branch: Dense -> ReLU] ──┐
                                                          ├─> [Fusion: Concat] -> [Dense -> ReLU] -> [Dense -> Softmax] -> class probabilities
    text_features  ──> [Text Branch:  Dense -> ReLU] ──┘

Each branch first projects its modality into a shared-size hidden
representation ("embedding"), then the two embeddings are
concatenated and passed through further layers to the final
classification head. This "two towers + fusion head" pattern is
the same one used by real multi-modal systems.

We implement forward pass, backward pass (manual backpropagation),
and mini-batch gradient descent training — so every matrix
multiplication and gradient is visible and explainable.
"""

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def softmax(x):
    # subtract max for numerical stability
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def one_hot(y, n_classes):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1
    return out


class MultiModalMLP:
    """
    Two-branch (image + text) multi-layer perceptron with a fusion
    layer and softmax classification head.

    Parameters
    ----------
    img_dim, txt_dim : int
        Input feature sizes for each modality.
    branch_hidden : int
        Hidden size for each modality's projection branch.
    fusion_hidden : int
        Hidden size of the fusion layer (after concatenation).
    n_classes : int
        Number of output classes.
    """

    def __init__(self, img_dim, txt_dim, branch_hidden=32, fusion_hidden=32,
                 n_classes=5, seed=42, l2=0.001):
        rng = np.random.default_rng(seed)
        self.l2 = l2  # L2 regularization strength (weight decay)

        def init(in_d, out_d):
            # He initialization — good default for ReLU networks
            return rng.normal(0, np.sqrt(2.0 / in_d), size=(in_d, out_d))

        # --- Image branch ---
        self.W_img = init(img_dim, branch_hidden)
        self.b_img = np.zeros(branch_hidden)

        # --- Text branch ---
        self.W_txt = init(txt_dim, branch_hidden)
        self.b_txt = np.zeros(branch_hidden)

        # --- Fusion layer (input = concat of both branch outputs) ---
        fusion_in = branch_hidden * 2
        self.W_fuse = init(fusion_in, fusion_hidden)
        self.b_fuse = np.zeros(fusion_hidden)

        # --- Output layer ---
        self.W_out = init(fusion_hidden, n_classes)
        self.b_out = np.zeros(n_classes)

        self.n_classes = n_classes

    # -------------------------------------------------------------
    def forward(self, X_img, X_txt):
        """Run a forward pass. Caches intermediate values for backprop."""
        # Image branch
        self.z_img = X_img @ self.W_img + self.b_img
        self.a_img = relu(self.z_img)

        # Text branch
        self.z_txt = X_txt @ self.W_txt + self.b_txt
        self.a_txt = relu(self.z_txt)

        # Fusion: concatenate both branch outputs
        self.fused_input = np.concatenate([self.a_img, self.a_txt], axis=1)
        self.z_fuse = self.fused_input @ self.W_fuse + self.b_fuse
        self.a_fuse = relu(self.z_fuse)

        # Output layer -> softmax probabilities
        self.z_out = self.a_fuse @ self.W_out + self.b_out
        self.probs = softmax(self.z_out)

        # cache inputs for backward pass
        self.X_img, self.X_txt = X_img, X_txt
        return self.probs

    # -------------------------------------------------------------
    def compute_loss(self, y_true):
        """Categorical cross-entropy loss + L2 weight penalty."""
        n = len(y_true)
        y_onehot = one_hot(y_true, self.n_classes)
        # clip for numerical stability (avoid log(0))
        log_probs = np.log(np.clip(self.probs, 1e-12, 1.0))
        ce_loss = -np.sum(y_onehot * log_probs) / n

        # L2 regularization (weight decay) — penalizes large weights
        # to reduce overfitting
        l2_term = self.l2 * sum(
            np.sum(W ** 2) for W in
            [self.W_img, self.W_txt, self.W_fuse, self.W_out]
        )
        return ce_loss + l2_term, y_onehot

    # -------------------------------------------------------------
    def backward(self, y_true, lr=0.01):
        """
        Manual backpropagation: compute gradients of the loss with
        respect to every weight/bias, then apply a gradient descent
        update step (in-place).
        """
        n = len(y_true)
        _, y_onehot = self.compute_loss(y_true)

        # --- Output layer gradient ---
        # dL/dz_out for softmax + cross-entropy simplifies to (probs - y_onehot) / n
        dz_out = (self.probs - y_onehot) / n
        dW_out = self.a_fuse.T @ dz_out + 2 * self.l2 * self.W_out
        db_out = dz_out.sum(axis=0)

        # --- Fusion layer gradient ---
        da_fuse = dz_out @ self.W_out.T
        dz_fuse = da_fuse * relu_grad(self.z_fuse)
        dW_fuse = self.fused_input.T @ dz_fuse + 2 * self.l2 * self.W_fuse
        db_fuse = dz_fuse.sum(axis=0)

        # --- Split gradient back to each branch ---
        d_fused_input = dz_fuse @ self.W_fuse.T
        branch_hidden = self.a_img.shape[1]
        d_a_img = d_fused_input[:, :branch_hidden]
        d_a_txt = d_fused_input[:, branch_hidden:]

        # --- Image branch gradient ---
        dz_img = d_a_img * relu_grad(self.z_img)
        dW_img = self.X_img.T @ dz_img + 2 * self.l2 * self.W_img
        db_img = dz_img.sum(axis=0)

        # --- Text branch gradient ---
        dz_txt = d_a_txt * relu_grad(self.z_txt)
        dW_txt = self.X_txt.T @ dz_txt + 2 * self.l2 * self.W_txt
        db_txt = dz_txt.sum(axis=0)

        # --- Gradient descent update ---
        self.W_out  -= lr * dW_out;  self.b_out  -= lr * db_out
        self.W_fuse -= lr * dW_fuse; self.b_fuse -= lr * db_fuse
        self.W_img  -= lr * dW_img;  self.b_img  -= lr * db_img
        self.W_txt  -= lr * dW_txt;  self.b_txt  -= lr * db_txt

    # -------------------------------------------------------------
    def predict(self, X_img, X_txt):
        probs = self.forward(X_img, X_txt)
        return np.argmax(probs, axis=1)

    def _get_weights(self):
        """Return a deep copy of all weights/biases (for checkpointing)."""
        return {
            'W_img': self.W_img.copy(), 'b_img': self.b_img.copy(),
            'W_txt': self.W_txt.copy(), 'b_txt': self.b_txt.copy(),
            'W_fuse': self.W_fuse.copy(), 'b_fuse': self.b_fuse.copy(),
            'W_out': self.W_out.copy(), 'b_out': self.b_out.copy(),
        }

    def _set_weights(self, weights):
        for k, v in weights.items():
            setattr(self, k, v)

    # -------------------------------------------------------------
    def fit(self, X_img, X_txt, y, epochs=200, batch_size=32, lr=0.05,
            X_img_val=None, X_txt_val=None, y_val=None, verbose=True,
            early_stopping_patience=20):
        """
        Train with mini-batch gradient descent.

        If validation data is provided, EARLY STOPPING is used:
        we track validation accuracy each epoch, checkpoint the
        best-so-far weights, and stop if validation accuracy
        hasn't improved for `early_stopping_patience` epochs —
        then restore the best checkpoint. This prevents the model
        from overfitting the training set (memorizing noise) at
        the cost of validation performance.

        Returns history dict with train/val loss and accuracy per epoch.
        """
        n = len(y)
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        rng = np.random.default_rng(0)

        best_val_acc = -1
        best_weights = None
        epochs_no_improve = 0
        stopped_epoch = epochs - 1

        for epoch in range(epochs):
            # Shuffle each epoch
            idx = rng.permutation(n)
            X_img_s, X_txt_s, y_s = X_img[idx], X_txt[idx], y[idx]

            for start in range(0, n, batch_size):
                end = start + batch_size
                xb_img = X_img_s[start:end]
                xb_txt = X_txt_s[start:end]
                yb = y_s[start:end]

                self.forward(xb_img, xb_txt)
                self.backward(yb, lr=lr)

            # --- end of epoch: record metrics ---
            train_probs = self.forward(X_img, X_txt)
            train_loss, _ = self.compute_loss(y)
            train_acc = (np.argmax(train_probs, axis=1) == y).mean()
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if X_img_val is not None:
                val_probs = self.forward(X_img_val, X_txt_val)
                val_loss, _ = self.compute_loss(y_val)
                val_acc = (np.argmax(val_probs, axis=1) == y_val).mean()
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # --- Early stopping bookkeeping ---
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_weights = self._get_weights()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= early_stopping_patience:
                    stopped_epoch = epoch
                    if verbose:
                        print(f"Early stopping at epoch {epoch} "
                              f"(no val improvement for {early_stopping_patience} epochs). "
                              f"Best val_acc={best_val_acc:.3f}")
                    break

            if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch:4d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f}"
                if X_img_val is not None:
                    msg += f" | val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
                print(msg)

        # Restore the best-performing weights (lowest val overfitting)
        if best_weights is not None:
            self._set_weights(best_weights)
            if verbose:
                print(f"Restored best weights from validation (val_acc={best_val_acc:.3f}, "
                      f"epoch {stopped_epoch - epochs_no_improve})")

        return history

    # -------------------------------------------------------------
    def save(self, path):
        np.savez(path,
                 W_img=self.W_img, b_img=self.b_img,
                 W_txt=self.W_txt, b_txt=self.b_txt,
                 W_fuse=self.W_fuse, b_fuse=self.b_fuse,
                 W_out=self.W_out, b_out=self.b_out,
                 n_classes=self.n_classes)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        model = cls(
            img_dim=data['W_img'].shape[0],
            txt_dim=data['W_txt'].shape[0],
            branch_hidden=data['W_img'].shape[1],
            fusion_hidden=data['W_fuse'].shape[1],
            n_classes=int(data['n_classes'])
        )
        model.W_img, model.b_img = data['W_img'], data['b_img']
        model.W_txt, model.b_txt = data['W_txt'], data['b_txt']
        model.W_fuse, model.b_fuse = data['W_fuse'], data['b_fuse']
        model.W_out, model.b_out = data['W_out'], data['b_out']
        return model
