# 🔢 Handwritten Digit Generator Web App

A web application that generates synthetic handwritten digit images using a Conditional Generative Adversarial Network (cGAN) trained from scratch on the MNIST dataset.

## 🌟 Features

- **Real-time Generation**: Generate handwritten digits instantly
- **Conditional Control**: Choose specific digits (0-9) to generate
- **Multiple Variations**: Creates 5 unique images per generation
- **MNIST-style Output**: 28×28 grayscale images similar to the famous MNIST dataset
- **Web Interface**: User-friendly Streamlit interface

## 🚀 Live Demo

[**Try the app here!**](https://your-app-url.streamlit.app) *(Replace with your actual Streamlit URL)*

## 🏗️ Architecture

### Model Details
- **Type**: Conditional Generative Adversarial Network (cGAN)
- **Framework**: PyTorch
- **Dataset**: MNIST (60,000 training images)
- **Training**: Trained from scratch (no pre-trained weights)
- **Input**: Random noise vector + digit label (0-9)
- **Output**: 28×28 grayscale images

### Network Architecture
- **Generator**: Multi-layer perceptron with batch normalization
- **Discriminator**: Multi-layer perceptron with dropout
- **Loss Function**: Binary cross-entropy
- **Optimizer**: Adam optimizer

## 📁 Project Structure

```
├── Handwritten_Digit_Generation_Web_App.py  # Main Streamlit application
├── conditional_gan_generator.pth             # Trained model weights
├── Requirements.txt                          # Python dependencies
└── README.md                                # Project documentation
```

## 🛠️ Installation & Usage

### Option 1: Use the Live Web App
Simply visit the [live demo](https://your-app-url.streamlit.app) - no installation required!

### Option 2: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Problem-3-Handwritten-Digit-Generation-Web-App.git
   cd Problem-3-Handwritten-Digit-Generation-Web-App
   ```

2. **Install dependencies**
   ```bash
   pip install -r Requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run Handwritten_Digit_Generation_Web_App.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

## 🎯 How to Use

1. **Select a digit** (0-9) from the dropdown menu
2. **Click "Generate Images"** to create 5 unique variations
3. **View the results** in both grid and individual formats
4. **Generate again** for completely new variations!

## 🧠 Model Training

The model was trained using:
- **Environment**: Google Colab with T4 GPU
- **Epochs**: 50 epochs
- **Training Time**: ~2-3 hours
- **Batch Size**: 128
- **Learning Rate**: 0.0002

## 📊 Performance

The model successfully generates recognizable handwritten digits with good variety and quality. Each generation produces unique variations while maintaining the characteristics of the selected digit.

## 🛡️ Requirements

- Python 3.7+
- PyTorch 2.0+
- Streamlit 1.28+
- See `Requirements.txt` for complete list

## 📝 Technical Details

### Generator Network
- Input: 100-dim noise + 10-dim label embedding
- Hidden layers: 256 → 512 → 1024 neurons
- Output: 784 neurons (28×28 pixels)
- Activation: ReLU + Tanh output

### Discriminator Network
- Input: 784 pixels + 10-dim label embedding
- Hidden layers: 512 → 256 neurons
- Output: 1 neuron (real/fake classification)
- Activation: LeakyReLU + Sigmoid output

## 🤝 Contributing

Feel free to fork this repository and submit pull requests for any improvements!

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun
- PyTorch team for the framework
- Streamlit for the web app framework

---

**Made with ❤️ using PyTorch and Streamlit**
