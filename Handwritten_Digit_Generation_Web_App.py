import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Generator",
    page_icon="🔢",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Define the Generator class (same as in training script)
class Generator(nn.Module):
    def __init__(self, noise_dim, num_classes, img_size):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Label embedding
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        # Generator layers
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        # Embed labels and concatenate with noise
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat([noise, label_embedding], dim=1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

@st.cache_resource
def load_model():
    """Load the trained generator model"""
    # Force CPU usage for Streamlit Cloud compatibility
    device = torch.device('cpu')
    
    try:
        # Load model checkpoint from root directory with CPU mapping
        checkpoint = torch.load('conditional_gan_generator.pth', map_location='cpu')
        
        # Handle potential key variations in saved model
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            # Fallback config if not saved properly
            config = {
                'noise_dim': 100,
                'num_classes': 10,
                'img_size': 28
            }
            st.warning("Using default config - model may not work perfectly")
        
        # Initialize generator
        generator = Generator(
            noise_dim=config['noise_dim'],
            num_classes=config['num_classes'],
            img_size=config['img_size']
        ).to(device)
        
        # Load trained weights
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)  # Fallback if saved differently
            
        generator.eval()
        
        return generator, device, config
        
    except FileNotFoundError:
        st.error("⚠️ Model file 'conditional_gan_generator.pth' not found!")
        st.info("Please make sure the trained model file is uploaded to your repository root directory.")
        st.stop()
    except RuntimeError as e:
        st.error(f"❌ Model loading error: {str(e)}")
        st.info("This might be a model compatibility issue. Make sure the model was saved correctly.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        st.info("Please check if the model file is correctly formatted and not corrupted.")
        st.stop()

def generate_digit_images(generator, device, config, digit, num_images=5):
    """Generate multiple images of a specific digit"""
    with torch.no_grad():
        # Create noise vectors
        noise = torch.randn(num_images, config['noise_dim']).to(device)
        
        # Create labels for the specified digit
        labels = torch.full((num_images,), digit, dtype=torch.long).to(device)
        
        # Generate images
        generated_imgs = generator(noise, labels)
        
        # Convert to numpy and denormalize
        imgs_np = generated_imgs.cpu().numpy()
        imgs_np = (imgs_np + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
        imgs_np = np.clip(imgs_np, 0, 1)
        
        return imgs_np

def create_image_grid(images, digit):
    """Create a grid of images similar to MNIST format"""
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))
    fig.suptitle(f'Generated Images of Digit {digit}', fontsize=16, fontweight='bold')
    
    for i in range(5):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Image {i+1}', fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

# Main Streamlit app
def main():
    # Title and description
    st.title("🔢 Handwritten Digit Image Generator")
    st.markdown("---")
    
    st.markdown("""
    **Generate synthetic MNIST-like images using a trained Conditional GAN model.**
    
    This app uses a Conditional Generative Adversarial Network (cGAN) trained from scratch 
    to generate handwritten digit images. Simply select a digit and watch the AI create 
    5 unique variations of that digit!
    
    🎯 **Features:**
    - Generate any digit from 0-9
    - Creates 5 unique variations per generation
    - Trained on MNIST dataset from scratch
    - Real-time generation using PyTorch
    """)
    
    # Load model
    with st.spinner("Loading trained model..."):
        generator, device, config = load_model()
    
    st.success("✅ Model loaded successfully!")
    st.markdown("---")
    
    # User input section
    st.subheader("Choose a digit to generate (0-9):")
    
    # Create digit selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        selected_digit = st.selectbox(
            "Select digit:",
            options=list(range(10)),
            index=0,
            help="Choose which digit you want to generate images for"
        )
    
    with col2:
        st.write(f"**Selected digit: {selected_digit}**")
        st.write("Click the button below to generate 5 new images")
    
    # Generate button
    generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
    with generate_col2:
        generate_button = st.button("🎲 Generate Images", type="primary", use_container_width=True)
    
    if generate_button:
        with st.spinner(f"🎨 Generating 5 images of digit {selected_digit}..."):
            try:
                # Generate images
                generated_images = generate_digit_images(
                    generator, device, config, selected_digit, num_images=5
                )
                
                # Create and display image grid
                image_grid = create_image_grid(generated_images, selected_digit)
                
                st.markdown("---")
                st.subheader(f"✨ Generated Images of Digit {selected_digit}")
                st.image(image_grid, use_container_width=True)
                
                # Display individual images
                st.markdown("### 🖼️ Individual Images:")
                cols = st.columns(5)
                for i in range(5):
                    with cols[i]:
                        # Convert single image to PIL format
                        img_array = (generated_images[i].squeeze() * 255).astype(np.uint8)
                        pil_img = Image.fromarray(img_array, mode='L')
                        st.image(pil_img, caption=f"Sample {i+1}", use_container_width=True)
                
                st.success(f"✅ Successfully generated 5 unique images of digit {selected_digit}!")
                
                # Add regeneration hint
                st.info("💡 Click 'Generate Images' again to create completely new variations!")
                
            except Exception as e:
                st.error(f"❌ Error generating images: {str(e)}")
                st.info("Please try again or contact support if the problem persists.")
    
    # Additional information
    st.markdown("---")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        with st.expander("ℹ️ About this Model"):
            st.markdown(f"""
            **Model Architecture:** Conditional GAN
            - **Generator**: Fully connected neural network
            - **Input**: Random noise + digit label
            - **Output**: 28×28 grayscale images
            - **Training**: MNIST dataset (60,000 images)
            - **Framework**: PyTorch
            
            **Specifications:**
            - Noise Dimension: {config.get('noise_dim', 'N/A')}
            - Classes: {config.get('num_classes', 'N/A')} digits (0-9)
            - Image Size: {config.get('img_size', 'N/A')}×{config.get('img_size', 'N/A')} pixels
            """)
    
    with col_info2:
        with st.expander("🚀 How to Use"):
            st.markdown("""
            **Quick Start:**
            1. Select any digit (0-9) from the dropdown
            2. Click "Generate Images" button
            3. View 5 unique AI-generated variations
            4. Generate again for new samples!
            
            **Tips:**
            - Each generation creates completely new images
            - Images are generated in real-time
            - Try different digits to see model performance
            - All images are 28×28 pixels (MNIST standard)
            """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 10px;'>
            🤖 Powered by Conditional GAN | Built with Streamlit & PyTorch
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
