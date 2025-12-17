
import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf

st.title("Melanoma Cancer Detection Demo")

# --- Model loader (load existing .h5/.keras from models/) ---
st.header("Model (.h5 or .keras) — load from `models/`")

# If there's a pre-trained model in models/, allow loading it
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)
available = [f for f in os.listdir(models_dir) if f.lower().endswith((".h5", ".keras"))]
if available:
	st.subheader("Load existing model from `models/` folder")
	choice = st.selectbox("Select a model file to load", options=["-- choose --"] + available)
	if choice and choice != "-- choose --":
		if st.button("Load selected model"):
			path = os.path.join(models_dir, choice)
			try:
				loaded = tf.keras.models.load_model(path)
				st.session_state['model'] = loaded
				st.success(f"Loaded model `{choice}` into session")
			except Exception as e:
				st.error(f"Failed to load model `{choice}`: {e}")
else:
	st.info("No .h5/.keras models found in `models/`. Copy a model into that folder to load it.")


# --- Image input: camera or file upload ---
st.header("Image Input")
st.write("Take a picture with your camera or upload an image file.")

col1, col2 = st.columns(2)

with col1:
	enable_cam = st.checkbox("Enable camera")
	camera_img = st.camera_input("Take a picture", disabled=not enable_cam)

with col2:
	uploaded_image = st.file_uploader("Or upload an image file", type=["png", "jpg", "jpeg"])

image = None
if camera_img is not None:
	try:
		image = Image.open(camera_img)
	except Exception as e:
		st.error(f"Failed to open camera image: {e}")
elif uploaded_image is not None:
	try:
		image = Image.open(uploaded_image)
	except Exception as e:
		st.error(f"Failed to open uploaded image: {e}")

if image is not None:
	st.image(image, caption="Selected image", use_column_width=True)
	# Optionally save the image for later processing
	save_images_dir = "input_images"
	os.makedirs(save_images_dir, exist_ok=True)
	# choose a filename
	fname = getattr(camera_img, "name", None) or getattr(uploaded_image, "name", None) or "capture.png"
	save_path = os.path.join(save_images_dir, fname)
	try:
		image.save(save_path)
		st.write(f"Image saved to `{save_path}`")
	except Exception:
		# non-fatal
		pass

	# If a model is loaded in this session, offer to run inference
	model = st.session_state.get('model', None)
	if model is None:
		st.info("No model loaded. Load a .h5/.keras model from the `models/` folder to run inference.")
	else:
		if st.button("Run inference on this image"):
			try:
				# determine model input size
				try:
					input_shape = model.input_shape
				except Exception:
					input_shape = None

				if input_shape and len(input_shape) >= 3:
					# input_shape is typically (None, H, W, C) or (None, C, H, W)
					if input_shape[0] is None:
						shape = input_shape[1:]
					else:
						shape = input_shape
					# prefer H,W at positions 0,1 if channels last
					if len(shape) == 3:
						h, w = int(shape[0]), int(shape[1])
					else:
						h, w = 224, 224
				else:
					h, w = 224, 224

				img = image.convert('RGB').resize((w, h))
				arr = np.array(img).astype('float32') / 255.0
				batch = np.expand_dims(arr, axis=0)

				preds = model.predict(batch)
				preds = np.asarray(preds)

				if preds.ndim == 2 and preds.shape[0] == 1:
					probs = preds[0]
					class_idx = int(np.argmax(probs))
					confidence = float(probs[class_idx])
					if preds >= 0.5:
							st.write("The model predicts a high likelihood of malignant melanoma. Please consult a medical professional for an accurate diagnosis.")
					else:
							st.write("The model predicts a low likelihood of malignant melanoma. However, this does not rule out the possibility of skin cancer. Always consult a medical professional for an accurate diagnosis.")
					st.markdown(f"**Confidence:** {confidence * 100:.1f}%")
					with st.expander("Raw probabilities"):
						st.write(probs.tolist())
				elif preds.ndim == 1:
					value = float(preds[0])
					st.markdown(f"**Model output:** {value * 100:.1f}%")
				else:
					st.write({"predictions_shape": preds.shape, "predictions": preds.tolist()})
			except Exception as e:
				st.error(f"Error during inference: {e}")

else:
	st.info("No image selected yet. Use the camera or upload a file.")
