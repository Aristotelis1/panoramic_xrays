import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Ensure you have these imports in your project
# pip install streamlit torch torchvision pillow matplotlib


class ImagePreprocessor:
    def __init__(self, image_size=(384, 768)):
        """
        Preprocessor for input images to match the model's expected input

        Args:
            image_size (tuple): Target image size for resizing
        """
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def preprocess_image(self, image):
        """
        Preprocess the uploaded image

        Args:
            image (PIL.Image): Input image

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Convert to grayscale and apply transformations
        processed_image = self.transform(image)
        return processed_image


def load_model(model_path):
    """
    Load the trained model

    Args:
        model_path (str): Path to the model weights

    Returns:
        torch.nn.Module: Loaded model
    """
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def run_inference(model, image_tensor):
    """
    Run model inference on the input image

    Args:
        model (torch.nn.Module): Loaded model
        image_tensor (torch.Tensor): Preprocessed image tensor

    Returns:
        torch.Tensor: Probability mask
    """
    with torch.no_grad():
        # Unsqueeze to add batch dimension
        probabilities = F.sigmoid(model(image_tensor.unsqueeze(0))).squeeze(0)
    return probabilities


def visualize_results(original_image, probability_mask):
    """
    Create a visualization of the original image and prediction mask

    Args:
        original_image (PIL.Image): Original input image
        probability_mask (torch.Tensor): Model's probability mask

    Returns:
        matplotlib figure
    """
    # Print debug information
    # st.write(f"Probability mask shape: {probability_mask.shape}")
    # st.write(f"Probability mask dtype: {probability_mask.dtype}")

    # Handle different possible tensor shapes
    if probability_mask.dim() == 3:
        # If shape is (C, H, W), select the first channel or take the mean
        if probability_mask.shape[0] == 1:
            prob_mask_np = probability_mask[0].numpy()
        else:
            prob_mask_np = probability_mask.mean(dim=0).numpy()
    elif probability_mask.dim() == 2:
        # If shape is already (H, W)
        prob_mask_np = probability_mask.numpy()
    else:
        raise ValueError(f"Unexpected probability mask shape: {probability_mask.shape}")

    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 20))

    # Original image
    # Convert to grayscale numpy array if it's not already
    if isinstance(original_image, Image.Image):
        original_image_np = np.array(original_image.convert('L'))
    elif isinstance(original_image, np.ndarray):
        original_image_np = original_image
    else:
        raise ValueError(f"Unexpected original image type: {type(original_image)}")

    ax1.imshow(original_image_np, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Probability mask
    ax2.imshow(prob_mask_np, cmap='viridis')
    ax2.set_title('Prediction Mask')
    ax2.axis('off')

    plt.tight_layout()
    return fig


def main():
    """
    Main Streamlit app
    """
    st.title('Caries Detection Model')

    # Model path (update this to your actual model path)
    MODEL_PATH = 'UNetEfficientnetB0-best.pth'

    # Create preprocessor
    preprocessor = ImagePreprocessor()

    # Load the model
    model = load_model(MODEL_PATH)

    if model is None:
        st.error("Could not load the model. Please check the model path and try again.")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a dental X-ray image",
        type=['png', 'jpg', 'jpeg', 'dcm', 'tiff']
    )

    if uploaded_file is not None:
        # Read the image
        original_image = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(original_image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        image_tensor = preprocessor.preprocess_image(original_image)

        # Run inference
        st.write("Running inference...")
        probabilities = run_inference(model, image_tensor)
        print("probabilities: ", probabilities.shape)
        # Visualize results
        fig = visualize_results(original_image, probabilities)

        # Display the results
        st.pyplot(fig)

        # Optional: Additional information about the prediction
        st.write("Prediction Mask Interpretation:")
        st.write("- Brighter areas indicate higher probability of caries")
        st.write("- The mask shows potential caries regions in the dental X-ray")


if __name__ == '__main__':
    main()
