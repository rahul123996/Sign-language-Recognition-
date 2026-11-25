import os
import sys
import types
import h5py
import numpy as np
from PIL import Image

class ASLModelPredictor:
    def __init__(self, h5_path):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Model file not found: {h5_path}")
        
        with h5py.File(h5_path, 'r') as f:
            self.w_c1 = f['model_weights/conv2d_1/conv2d_1/kernel:0'][:]
            self.b_c1 = f['model_weights/conv2d_1/conv2d_1/bias:0'][:]
            self.w_c2 = f['model_weights/conv2d_2/conv2d_2/kernel:0'][:]
            self.b_c2 = f['model_weights/conv2d_2/conv2d_2/bias:0'][:]
            self.w_d1 = f['model_weights/dense_1/dense_1/kernel:0'][:]
            self.b_d1 = f['model_weights/dense_1/dense_1/bias:0'][:]
            self.w_d2 = f['model_weights/dense_2/dense_2/kernel:0'][:]
            self.b_d2 = f['model_weights/dense_2/dense_2/bias:0'][:]

    def _conv2d_valid(self, x, w, b):
        H, W, Cin = x.shape
        kh, kw, _, Cout = w.shape
        out_h = H - kh + 1
        out_w = W - kw + 1
        shape = (out_h, out_w, kh, kw, Cin)
        strides = (x.strides[0], x.strides[1], x.strides[0], x.strides[1], x.strides[2])
        cols = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        cols = cols.reshape(out_h * out_w, kh * kw * Cin)
        w_flat = w.reshape(kh * kw * Cin, Cout)
        out = np.dot(cols, w_flat) + b
        return np.maximum(out.reshape(out_h, out_w, Cout), 0.0)

    def _maxpool2d(self, x, pool=2):
        H, W, C = x.shape
        out_h, out_w = H // pool, W // pool
        return x[:out_h * pool, :out_w * pool].reshape(out_h, pool, out_w, pool, C).max(axis=(1, 3))

    def predict(self, img):
        if isinstance(img, (list, tuple)):
            img = np.array(img, dtype=np.float32)
        if img.ndim == 4:
            x = img[0].astype(np.float32)
        else:
            x = img.astype(np.float32)
        
        # Ensure 64x64x3
        if x.shape[0] != 64 or x.shape[1] != 64:
            pil_temp = Image.fromarray(x.astype(np.uint8)).resize((64, 64))
            x = np.array(pil_temp, dtype=np.float32)
            if x.ndim == 2:
                x = np.stack([x]*3, axis=-1)

        # Rescale if needed (if pixel values are 0-255)
        if np.max(x) > 1.0:
            x_norm = x / 255.0
        else:
            x_norm = x

        # Forward pass
        c1 = self._conv2d_valid(x_norm, self.w_c1, self.b_c1)
        p1 = self._maxpool2d(c1, 2)
        c2 = self._conv2d_valid(p1, self.w_c2, self.b_c2)
        p2 = self._maxpool2d(c2, 2)
        flat = p2.reshape(-1)
        d1 = np.maximum(np.dot(flat, self.w_d1) + self.b_d1, 0.0)
        logits = np.dot(d1, self.w_d2) + self.b_d2
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        
        # Return probability array of shape (1, 26)
        # Set highest confident class to 1.0 if probability is sufficient
        # to ensure compatibility with 'if result[0][i] == 1' logic
        result = np.zeros((1, 26), dtype=np.float32)
        max_idx = np.argmax(probs)
        if probs[max_idx] >= 0.4:
            result[0, max_idx] = 1.0
        return result

    def predict_detailed(self, img):
        if isinstance(img, (list, tuple)):
            img = np.array(img, dtype=np.float32)
        if img.ndim == 4:
            x = img[0].astype(np.float32)
        else:
            x = img.astype(np.float32)
        
        if x.shape[0] != 64 or x.shape[1] != 64:
            pil_temp = Image.fromarray(x.astype(np.uint8)).resize((64, 64))
            x = np.array(pil_temp, dtype=np.float32)
            if x.ndim == 2:
                x = np.stack([x]*3, axis=-1)

        if np.max(x) > 1.0:
            x_norm = x / 255.0
        else:
            x_norm = x

        c1 = self._conv2d_valid(x_norm, self.w_c1, self.b_c1)
        p1 = self._maxpool2d(c1, 2)
        c2 = self._conv2d_valid(p1, self.w_c2, self.b_c2)
        p2 = self._maxpool2d(c2, 2)
        flat = p2.reshape(-1)
        d1 = np.maximum(np.dot(flat, self.w_d1) + self.b_d1, 0.0)
        logits = np.dot(d1, self.w_d2) + self.b_d2
        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        
        max_idx = int(np.argmax(probs))
        confidence = float(probs[max_idx])
        char = chr(ord('A') + max_idx) if confidence >= 0.35 else ''
        return char, confidence, probs


def load_model(filepath, *args, **kwargs):
    return ASLModelPredictor(filepath)


class ImageModule:
    @staticmethod
    def load_img(path, target_size=None):
        img = Image.open(path).convert('RGB')
        if target_size is not None:
            img = img.resize((target_size[0], target_size[1]))
        return img

    @staticmethod
    def img_to_array(img):
        return np.array(img, dtype=np.float32)

image = ImageModule


def install_keras_shim():
    """Installs shim modules into sys.modules so imports like
    'from keras.models import load_model' and
    'from keras.preprocessing import image'
    succeed seamlessly."""
    try:
        import keras
        # If keras can load without DLL errors, keep it
        _ = keras.models.load_model
    except Exception:
        # Create keras shim package
        keras_mod = types.ModuleType('keras')
        keras_models = types.ModuleType('keras.models')
        keras_models.load_model = load_model
        keras_preprocessing = types.ModuleType('keras.preprocessing')
        keras_preprocessing.image = image
        
        keras_mod.models = keras_models
        keras_mod.preprocessing = keras_preprocessing

        sys.modules['keras'] = keras_mod
        sys.modules['keras.models'] = keras_models
        sys.modules['keras.preprocessing'] = keras_preprocessing
        sys.modules['keras.preprocessing.image'] = image


# Auto-install shim if keras cannot be imported cleanly
install_keras_shim()
