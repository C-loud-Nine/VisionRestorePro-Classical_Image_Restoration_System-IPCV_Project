"""
Professional Image Restoration Streamlit App - Enhanced Version
Author: Enhanced for Shafi
Version: 4.8 - Fixed Analysis Display & Improved Deblurring
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64
import time
import os
import math
import random
import warnings
from typing import Optional, Tuple, Dict, Any, List
from scipy import ndimage
from scipy.signal import fftconvolve, find_peaks
from skimage import img_as_float, color, data
from skimage.transform import radon, resize
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet
from skimage import restoration as skrest
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
import pandas as pd
import datetime

# Configure page
st.set_page_config(
    page_title="VisionRestore Pro",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/visionrestorepro',
        'Report a bug': 'https://github.com/yourusername/visionrestorepro/issues',
        'About': "# VisionRestore Pro\nAdvanced image restoration with intelligent processing and adaptive enhancement."
    }
)

# Custom CSS for modern dark/light theme with proper contrast
st.markdown("""
<style>
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --secondary: #7c3aed;
        --accent: #06d6a0;
        --success: #10b981;
        --warning: #f59e0b;
        --error: #ef4444;
        --info: #3b82f6;
        
        /* Light theme */
        --light-bg: #ffffff;
        --light-card-bg: #f8fafc;
        --light-text: #1e293b;
        --light-text-light: #475569;
        --light-text-muted: #64748b;
        --light-border: #e2e8f0;
        --light-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --light-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        
        /* Dark theme */
        --dark-bg: #0f172a;
        --dark-card-bg: #1e293b;
        --dark-text: #f1f5f9;
        --dark-text-light: #cbd5e1;
        --dark-text-muted: #94a3b8;
        --dark-border: #334155;
        --dark-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.2);
        --dark-shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -2px rgba(0, 0, 0, 0.2);
    }

    [data-theme="light"] {
        --background: var(--light-bg);
        --card-bg: var(--light-card-bg);
        --text: var(--light-text);
        --text-light: var(--light-text-light);
        --text-muted: var(--light-text-muted);
        --border: var(--light-border);
        --shadow: var(--light-shadow);
        --shadow-lg: var(--light-shadow-lg);
    }

    [data-theme="dark"] {
        --background: var(--dark-bg);
        --card-bg: var(--dark-card-bg);
        --text: var(--dark-text);
        --text-light: var(--dark-text-light);
        --text-muted: var(--dark-text-muted);
        --border: var(--dark-border);
        --shadow: var(--dark-shadow);
        --shadow-lg: var(--dark-shadow-lg);
    }

    /* Global styles */
    .main {
        background-color: var(--background);
        color: var(--text);
        transition: all 0.3s ease;
    }

    .stApp {
        background-color: var(--background);
    }

    /* Header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }

    .app-subtitle {
        font-size: 1.1rem;
        color: var(--text-light);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Card styling */
    .card {
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        border: 1px solid var(--border);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }

    .card:hover {
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }

    .metric-card {
        background: linear-gradient(135deg, var(--card-bg), var(--background));
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border-left: 4px solid var(--primary);
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text);
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.9rem;
        color: var(--text-light);
        font-weight: 500;
    }

    .improvement-positive {
        color: var(--success);
        font-weight: 700;
    }

    .improvement-negative {
        color: var(--error);
        font-weight: 700;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-header::before {
        content: "";
        display: block;
        width: 4px;
        height: 1.5rem;
        background: linear-gradient(to bottom, var(--primary), var(--secondary));
        border-radius: 2px;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        border: none;
    }

    .stButton > button:first-child {
        background-color: var(--primary);
        color: white;
    }

    .stButton > button:first-child:hover {
        background-color: var(--primary-dark);
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, var(--card-bg), var(--background));
        border-right: 1px solid var(--border);
    }

    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary);
        text-align: center;
    }

    /* Parameter group styling */
    .param-group {
        background: linear-gradient(135deg, var(--card-bg), var(--background));
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }

    .param-group:hover {
        box-shadow: var(--shadow-lg);
        border-color: var(--primary);
    }

    .param-group-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--border);
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background-color: var(--primary);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--card-bg);
        padding: 8px;
        border-radius: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        background-color: transparent;
        color: var(--text-light);
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--primary) !important;
        color: white !important;
    }

    /* File uploader */
    .stFileUploader > div > div {
        border: 2px dashed var(--border);
        border-radius: 12px;
        background-color: var(--card-bg);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
    }

    /* Image containers */
    .image-container {
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
        border: 1px solid var(--border);
        text-align: center;
    }

    .image-title {
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.5rem;
        text-align: center;
    }

    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 20px;
        margin: 0.25rem;
    }

    .badge-primary {
        background-color: var(--primary);
        color: white;
    }

    .badge-success {
        background-color: var(--success);
        color: white;
    }

    .badge-warning {
        background-color: var(--warning);
        color: white;
    }

    /* Status indicators */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        margin: 0.25rem 0;
        width: 100%;
        justify-content: center;
    }

    .status-success {
        background-color: rgba(16, 185, 129, 0.1);
        color: var(--success);
        border: 1px solid var(--success);
    }

    .status-warning {
        background-color: rgba(245, 158, 11, 0.1);
        color: var(--warning);
        border: 1px solid var(--warning);
    }

    .status-info {
        background-color: rgba(59, 130, 246, 0.1);
        color: var(--info);
        border: 1px solid var(--info);
    }

    /* Professional sidebar sections */
    .sidebar-section {
        background: linear-gradient(135deg, var(--card-bg), var(--background));
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
    }

    .sidebar-section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border);
    }

    /* Progress bars */
    .stProgress > div > div > div {
        background-color: var(--primary);
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
        }
        
        .card {
            padding: 1rem;
        }
    }

    /* Animation classes */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--card-bg);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }

    /* Fix contrast for all text elements */
    .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader {
        color: var(--text) !important;
    }

    .stSlider label, .stCheckbox label, .stRadio label {
        color: var(--text) !important;
    }

    .stSelectbox label, .stTextInput label {
        color: var(--text) !important;
    }

    /* Ensure proper contrast in all streamlit elements */
    div[data-testid="stVerticalBlock"] {
        color: var(--text);
    }

    /* Kernel drawing styles */
    .kernel-grid {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        gap: 2px;
        margin: 1rem 0;
        max-width: 300px;
    }

    .kernel-cell {
        width: 35px;
        height: 35px;
        border: 2px solid var(--border);
        border-radius: 4px;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        background-color: var(--card-bg);
    }

    .kernel-cell:hover {
        border-color: var(--primary);
        transform: scale(1.1);
    }

    .kernel-cell.active {
        background-color: var(--primary);
        color: white;
        border-color: var(--primary);
    }

    .degradation-history {
        background: var(--card-bg);
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid var(--border);
        max-height: 200px;
        overflow-y: auto;
    }

    .history-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 4px;
        background: var(--background);
        border-left: 3px solid var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# JavaScript for theme toggle
st.markdown("""
<script>
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);
}

// Load saved theme or default to light
const savedTheme = localStorage.getItem('theme') || 'light';
setTheme(savedTheme);

// Listen for theme changes from Streamlit
window.addEventListener('message', function(event) {
    if (event.data.type === 'streamlit:setTheme') {
        const theme = event.data.theme;
        setTheme(theme);
    }
});
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'restoration_results' not in st.session_state:
    st.session_state.restoration_results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'ground_truth' not in st.session_state:
    st.session_state.ground_truth = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None
if 'texture_info' not in st.session_state:
    st.session_state.texture_info = None
if 'current_theme' not in st.session_state:
    st.session_state.current_theme = 'light'
if 'show_psf_analysis' not in st.session_state:
    st.session_state.show_psf_analysis = False
if 'degraded_image' not in st.session_state:
    st.session_state.degraded_image = None
if 'clean_image' not in st.session_state:
    st.session_state.clean_image = None
if 'degradation_steps' not in st.session_state:
    st.session_state.degradation_steps = []
if 'custom_kernel' not in st.session_state:
    st.session_state.custom_kernel = np.zeros((7, 7), dtype=np.float32)
if 'current_preview' not in st.session_state:
    st.session_state.current_preview = None

# Initialize random seed
np.random.seed(int(time.time()) % 1000)
random.seed(int(time.time()) % 1000)

# BM3D availability
try:
    from bm3d import bm3d
    HAS_BM3D = True
except ImportError:
    HAS_BM3D = False

# PyWavelets for wavelet transforms
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

# =============================================================================
# ENHANCED RESTORATION FUNCTIONS WITH IMPROVED DEBLURRING
# =============================================================================

def fft_conv(img, kernel):
    """Fast convolution using FFT for computational efficiency."""
    if img.ndim == 3:
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            out[..., c] = fftconvolve(img[..., c], kernel, mode='same')
        return out
    else:
        return fftconvolve(img, kernel, mode='same')

def motion_blur_psf(length: float, angle_deg: float) -> np.ndarray:
    """Generate motion blur Point Spread Function (PSF)."""
    length = max(1.0, float(length))
    size = max(3, int(math.ceil(length)) | 1)
    psf = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    theta = math.radians(angle_deg)
    
    x_coords = np.linspace(-length/2.0, length/2.0, size)
    for x in x_coords:
        xi = int(round(center + x * math.cos(theta)))
        yi = int(round(center + x * math.sin(theta)))
        if 0 <= xi < size and 0 <= yi < size:
            psf[yi, xi] += 1.0
    
    psf_sum = psf.sum()
    if psf_sum == 0:
        psf[center, center] = 1.0
    else:
        psf /= psf_sum
    return psf

def disk_psf(radius: float) -> np.ndarray:
    """Generate disk PSF"""
    r = max(1.0, float(radius))
    size = max(3, int(math.ceil(2*r+1)) | 1)
    Y, X = np.ogrid[:size, :size]
    cy, cx = size//2, size//2
    mask = ((Y-cy)**2 + (X-cx)**2) <= r*r
    psf = mask.astype(float)
    psf /= psf.sum()
    return psf

def gaussian_psf(sigma: float) -> np.ndarray:
    """Generate Gaussian PSF"""
    s = max(0.5, float(sigma))
    size = max(3, int(math.ceil(s*6)) | 1)
    ax = np.arange(-size//2 + 1., size//2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0*s*s))
    kernel /= kernel.sum()
    return kernel

def circular_mask(h, w, dtype=np.float32):
    """Create circular mask to reduce boundary artifacts"""
    Y, X = np.ogrid[:h, :w]
    cy, cx = h/2.0, w/2.0
    r = min(h, w)/2.0
    mask = ((Y-cy)**2 + (X-cx)**2) <= (r**2)
    return mask.astype(dtype)

def denoise_luminance(img_rgb: np.ndarray, sigma: float, use_bm3d: bool = True):
    """Advanced denoising applied only to luminance channel."""
    img_u8 = (img_rgb * 255).astype('uint8')
    ycrcb = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb).astype('float32') / 255.0
    y = ycrcb[..., 0]
    
    if HAS_BM3D and use_bm3d:
        try:
            den_y = bm3d(y, sigma_psd=sigma/255.0)
            den_y = np.clip(den_y, 0, 1)
        except Exception:
            den_y = denoise_wavelet(y, sigma=sigma/255.0, convert2y=False, method='VisuShrink', mode='soft')
            den_y = denoise_tv_chambolle(den_y, weight=0.01)
    else:
        den_y = denoise_wavelet(y, sigma=sigma/255.0, convert2y=False, method='VisuShrink', mode='soft')
        den_y = denoise_tv_chambolle(den_y, weight=0.01)
    
    ycrcb[..., 0] = den_y
    den_rgb = cv2.cvtColor((ycrcb * 255).astype('uint8'), cv2.COLOR_YCrCb2RGB)
    return den_rgb.astype('float32') / 255.0, den_y

def estimate_snr(y_channel: np.ndarray):
    """Estimate Signal-to-Noise Ratio using robust statistics."""
    med = np.median(y_channel)
    mad = np.median(np.abs(y_channel - med)) + 1e-9
    noise_std_est = 1.4826 * mad
    sig_var = np.var(y_channel)
    noise_var = noise_std_est**2
    snr = sig_var / (noise_var + 1e-12)
    return max(0.01, float(snr))

def adaptive_wiener_balance(y_channel: np.ndarray, base=0.01):
    """Adapt Wiener regularization parameter based on estimated SNR."""
    snr = estimate_snr(y_channel)
    bal = base * (1.0 / (1.0 + 0.1 * snr))
    return float(np.clip(bal, 1e-5, 0.2))

def estimate_motion_psf_robust(image_gray: np.ndarray, max_len=120):
    """Robust PSF estimation combining Radon transform and cepstrum analysis."""
    try:
        img = image_gray.astype(np.float32).copy()
        if img.max() > 1.0 or img.min() < 0.0:
            img = (img - img.min()) / max(1e-9, img.max() - img.min())
        
        h, w = img.shape
        mask = circular_mask(h, w)
        img_masked = img * mask
        
        # Angle estimation using Radon transform
        theta = np.linspace(0., 180., 90, endpoint=False)
        R = radon(img_masked, theta=theta, circle=True)
        angle = float(theta[int(np.argmax(R.var(axis=0)))])
        
        # Length estimation using cepstrum analysis
        rot_img = ndimage.rotate(img_masked, -angle, reshape=False, mode='reflect')
        mag_spectrum = np.abs(np.fft.fft2(rot_img)) + 1e-8
        cepstrum = np.real(np.fft.ifft2(np.log(mag_spectrum)))
        
        center_row = cepstrum[cepstrum.shape[0]//2, :]
        peaks, _ = find_peaks(np.abs(center_row), distance=2, 
                            height=(np.mean(np.abs(center_row)) * 1.05))
        
        if len(peaks) == 0:
            length = min(max_len, 6)
        else:
            center = len(center_row)//2
            peak = peaks[np.argmin(np.abs(peaks - center))]
            length = abs(peak - center)
        
        length = max(1.0, min(length, max_len))
        psf = motion_blur_psf(length, angle)
        
        return psf, length, angle
        
    except Exception as e:
        warnings.warn(f"PSF estimation failed: {e}, using default PSF")
        return motion_blur_psf(1, 0), 1.0, 0.0

def rl_tv_deconvolution(y_obs, psf, init=None, max_iters=60, tv_interval=6, tv_weight=0.01):
    """Richardson-Lucy deconvolution with Total Variation regularization."""
    if init is None:
        current = y_obs.copy()
    else:
        current = init.copy()
    
    psf_flipped = psf[::-1, ::-1]
    
    for iteration in range(max_iters):
        conv_current = fft_conv(current, psf)
        conv_current = np.clip(conv_current, 1e-12, 1.0)
        relative_blur = y_obs / conv_current
        error_est = fft_conv(relative_blur, psf_flipped)
        current = current * error_est
        current = np.clip(current, 0, 1)
        
        if (iteration + 1) % tv_interval == 0:
            current = denoise_tv_chambolle(current, weight=tv_weight)
    
    return current

def wiener_deconvolution_rgb(img_rgb: np.ndarray, psf: np.ndarray, balance: float = 0.01) -> np.ndarray:
    """Apply Wiener deconvolution to RGB image."""
    img_u8 = (img_rgb * 255).astype('uint8')
    ycrcb = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb).astype('float32') / 255.0
    y_channel = ycrcb[..., 0]
    
    # Apply Wiener deconvolution to Y channel only
    deblurred_y = skrest.wiener(y_channel, psf, balance)
    deblurred_y = np.clip(deblurred_y, 0, 1)
    
    # Convert back to RGB
    ycrcb[..., 0] = deblurred_y
    deblurred_rgb = cv2.cvtColor((ycrcb * 255).astype('uint8'), cv2.COLOR_YCrCb2RGB)
    return deblurred_rgb.astype('float32') / 255.0

def analyze_texture_simple(img_rgb: np.ndarray):
    """Simple texture analysis using gradient magnitude."""
    gray = cv2.cvtColor((img_rgb * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    gray = gray.astype('float32') / 255.0
    
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    texture_energy = np.mean(grad_mag)
    
    # Adaptive parameter recommendations
    if texture_energy > 0.1:
        tv_weight = 0.004
        sharpening_factor = 0.7
        noise_sigma = 8.0
    elif texture_energy < 0.03:
        tv_weight = 0.02
        sharpening_factor = 1.3
        noise_sigma = 15.0
    else:
        tv_weight = 0.01
        sharpening_factor = 1.0
        noise_sigma = 12.0
    
    return {
        'texture_energy': texture_energy,
        'recommended_tv_weight': tv_weight,
        'sharpening_factor': sharpening_factor,
        'recommended_noise_sigma': noise_sigma
    }

def detect_degradation_type(img_rgb: np.ndarray) -> Dict[str, Any]:
    """Detect the type and severity of degradation in the image."""
    gray = cv2.cvtColor((img_rgb * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    gray = gray.astype('float32') / 255.0
    
    # Analyze noise
    noise_analysis = analyze_noise_pattern(gray)
    
    # Analyze blur
    blur_analysis = analyze_blur_severity(gray)
    
    # Detect salt & pepper noise
    sp_detection = detect_salt_pepper_noise(gray)
    
    return {
        'noise_type': noise_analysis['noise_type'],
        'noise_level': noise_analysis['noise_level'],
        'blur_severity': blur_analysis['blur_severity'],
        'blur_detected': blur_analysis['blur_detected'],
        'salt_pepper_present': sp_detection['salt_pepper_present'],
        'sp_ratio': sp_detection['sp_ratio'],
        'overall_degradation': max(noise_analysis['noise_level'], blur_analysis['blur_severity'])
    }

def analyze_noise_pattern(gray_img: np.ndarray) -> Dict[str, Any]:
    """Analyze noise pattern to determine type and level."""
    # Calculate noise variance using multiple methods
    laplacian_var = cv2.Laplacian((gray_img * 255).astype('uint8'), cv2.CV_64F).var()
    
    # Simple noise estimation using median absolute deviation
    med = np.median(gray_img)
    mad = np.median(np.abs(gray_img - med))
    noise_std = mad / 0.6745
    
    # Determine noise type
    if noise_std < 0.01:
        noise_type = "low"
    elif noise_std < 0.03:
        noise_type = "medium"
    else:
        noise_type = "high"
    
    return {
        'noise_type': noise_type,
        'noise_level': min(1.0, noise_std * 10),
        'laplacian_var': laplacian_var,
        'noise_std': noise_std
    }

def analyze_blur_severity(gray_img: np.ndarray) -> Dict[str, Any]:
    """Analyze blur severity using frequency domain analysis - more sensitive version."""
    # Calculate blur metric using Laplacian variance
    laplacian = cv2.Laplacian((gray_img * 255).astype('uint8'), cv2.CV_64F)
    blur_metric = laplacian.var()
    
    # More sensitive blur detection - lower threshold
    blur_detected = blur_metric < 500  # Increased threshold from 100 to 500
    
    # More nuanced blur severity calculation
    if blur_metric > 1000:
        blur_severity = 0.0  # No blur
    elif blur_metric > 500:
        blur_severity = 0.2  # Very mild blur
    elif blur_metric > 200:
        blur_severity = 0.4  # Mild blur
    elif blur_metric > 100:
        blur_severity = 0.6  # Moderate blur
    elif blur_metric > 50:
        blur_severity = 0.8  # Strong blur
    else:
        blur_severity = 1.0  # Very strong blur
    
    return {
        'blur_detected': blur_detected,
        'blur_severity': blur_severity,
        'blur_metric': blur_metric
    }

def detect_salt_pepper_noise(gray_img: np.ndarray) -> Dict[str, Any]:
    """Detect salt and pepper noise in the image."""
    # Count extreme values (near 0 and near 1)
    near_black = np.sum(gray_img < 0.05) / gray_img.size
    near_white = np.sum(gray_img > 0.95) / gray_img.size
    
    salt_pepper_present = (near_black > 0.01) or (near_white > 0.01)
    sp_ratio = near_black + near_white
    
    return {
        'salt_pepper_present': salt_pepper_present,
        'sp_ratio': sp_ratio,
        'black_ratio': near_black,
        'white_ratio': near_white
    }

def adaptive_denoising(img_rgb: np.ndarray, degradation_info: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    """Apply adaptive denoising based on detected noise type."""
    
    if degradation_info['salt_pepper_present']:
        # Use median filtering for salt & pepper noise
        img_u8 = (img_rgb * 255).astype('uint8')
        denoised = cv2.medianBlur(img_u8, 3)
        denoised = denoised.astype('float32') / 255.0
    else:
        # Use BM3D or wavelet for Gaussian noise
        if HAS_BM3D and params['USE_BM3D']:
            try:
                denoised = bm3d_rgb_denoising(img_rgb, degradation_info['noise_level'])
            except Exception:
                denoised = wavelet_rgb_denoising(img_rgb, degradation_info['noise_level'])
        else:
            denoised = wavelet_rgb_denoising(img_rgb, degradation_info['noise_level'])
    
    return denoised

def bm3d_rgb_denoising(img_rgb: np.ndarray, noise_level: float) -> np.ndarray:
    """BM3D denoising for RGB images."""
    if not HAS_BM3D:
        return wavelet_rgb_denoising(img_rgb, noise_level)
    
    # Convert to YCbCr for luminance channel processing
    img_u8 = (img_rgb * 255).astype('uint8')
    ycrcb = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb).astype('float32') / 255.0
    
    # Denoise Y channel only
    y_channel = ycrcb[..., 0]
    sigma_psd = noise_level * 0.1  # Adjust based on noise level
    
    try:
        denoised_y = bm3d(y_channel, sigma_psd=sigma_psd)
        denoised_y = np.clip(denoised_y, 0, 1)
        
        ycrcb[..., 0] = denoised_y
        denoised_rgb = cv2.cvtColor((ycrcb * 255).astype('uint8'), cv2.COLOR_YCrCb2RGB)
        return denoised_rgb.astype('float32') / 255.0
    except Exception:
        return wavelet_rgb_denoising(img_rgb, noise_level)

def wavelet_rgb_denoising(img_rgb: np.ndarray, noise_level: float) -> np.ndarray:
    """Wavelet denoising for RGB images."""
    denoised = np.zeros_like(img_rgb)
    
    for channel in range(3):
        channel_denoised = denoise_wavelet(
            img_rgb[..., channel], 
            sigma=noise_level * 0.2,
            method='BayesShrink' if HAS_PYWT else 'VisuShrink',
            mode='soft'
        )
        denoised[..., channel] = np.clip(channel_denoised, 0, 1)
    
    return denoised

# Replace the adaptive_deblurring function with this improved version:
def adaptive_deblurring(img_rgb: np.ndarray, degradation_info: Dict[str, Any], params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply adaptive deblurring with Wiener as base in ALL cases."""
    
    # Convert to YCbCr for luminance processing
    img_u8 = (img_rgb * 255).astype('uint8')
    ycrcb = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YCrCb).astype('float32') / 255.0
    y_channel = ycrcb[..., 0]
    
    # Estimate PSF - always estimate even for mild blur
    psf, psf_len, psf_ang = estimate_motion_psf_robust(y_channel, max_len=80)
    
    # ALWAYS compute Wiener deconvolution as base with adaptive parameters
    # Adjust balance based on blur severity - higher severity = lower balance (more aggressive)
    base_balance = params['WIENER_BAL']
    if degradation_info['blur_severity'] < 0.2:
        # Very mild blur - use conservative deconvolution
        wiener_balance = base_balance * 2.0  # More conservative
    elif degradation_info['blur_severity'] < 0.5:
        # Moderate blur - use standard balance
        wiener_balance = base_balance
    else:
        # Strong blur - use aggressive deconvolution
        wiener_balance = base_balance * 0.5  # More aggressive
    
    wiener_balance = max(0.001, min(0.1, wiener_balance))  # Clamp to valid range
    
    wiener_result = wiener_deconvolution_rgb(img_rgb, psf, wiener_balance)
    
    # Start with Wiener result as base
    deblurred_rgb = wiener_result
    used_rl = False
    adaptive_iters = 0
    
    # Determine deblur method based on blur severity
    if degradation_info['blur_severity'] < 0.1:
        deblur_method = "Wiener (Very Mild)"
    elif degradation_info['blur_severity'] < 0.3:
        deblur_method = "Wiener (Mild)"
    elif degradation_info['blur_severity'] < 0.6:
        deblur_method = "Wiener (Moderate)"
    else:
        deblur_method = "Wiener (Strong)"
    
    # Only apply RL deconvolution for severe blur cases
    if degradation_info['blur_severity'] > 0.7:  # High threshold for RL
        # Strong blur detected - use RL deconvolution on Wiener result
        adaptive_iters = int(params['RL_ITERS'] * degradation_info['blur_severity'])
        adaptive_iters = min(200, max(30, adaptive_iters))
        
        # Apply RL deconvolution to Y channel of Wiener result
        wiener_ycrcb = cv2.cvtColor((wiener_result * 255).astype('uint8'), cv2.COLOR_RGB2YCrCb).astype('float32') / 255.0
        wiener_y = wiener_ycrcb[..., 0]
        
        deblurred_y = rl_tv_deconvolution(
            wiener_y, psf, 
            max_iters=adaptive_iters,
            tv_interval=params['TV_INTERVAL'],
            tv_weight=params['TV_WEIGHT'] * degradation_info['blur_severity']
        )
        
        # Convert back to RGB
        wiener_ycrcb[..., 0] = deblurred_y
        rl_result = cv2.cvtColor((wiener_ycrcb * 255).astype('uint8'), cv2.COLOR_YCrCb2RGB)
        rl_result = rl_result.astype('float32') / 255.0
        
        # Blend Wiener and RL results based on blur severity
        blend_alpha = min(1.0, (degradation_info['blur_severity'] - 0.7) * 3.0)  # Only blend when severity > 0.7
        deblurred_rgb = blend_alpha * rl_result + (1 - blend_alpha) * wiener_result
        deblurred_rgb = np.clip(deblurred_rgb, 0, 1)
        
        used_rl = True
        deblur_method = f"Wiener + RL (Blend: {blend_alpha:.2f})"
    
    deblur_info = {
        'used_rl': used_rl,
        'deblur_method': deblur_method,
        'adaptive_iters': adaptive_iters,
        'psf': psf,
        'psf_len': psf_len,
        'psf_ang': psf_ang,
        'blur_severity': degradation_info['blur_severity'],
        'wiener_balance': wiener_balance,
        'wiener_stage': wiener_result  # Store Wiener result for analysis display
    }
    
    return deblurred_rgb, deblur_info

# Also modify the restoration pipeline to always apply deblurring:
def restore_image_with_degradation_awareness(img_rgb: np.ndarray, params: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced restoration pipeline with degradation awareness."""
    results = {'input': img_rgb.copy()}
    
    # Step 1: Degradation analysis
    degradation_info = detect_degradation_type(img_rgb)
    results['degradation_info'] = degradation_info
    
    # Step 2: Adaptive denoising based on degradation type
    if degradation_info['noise_level'] > 0.1 or degradation_info['salt_pepper_present']:
        denoised = adaptive_denoising(img_rgb, degradation_info, params)
        results['denoised'] = denoised
    else:
        denoised = img_rgb.copy()
        results['denoised'] = denoised
    
    # Step 3: ALWAYS apply adaptive deblurring (Wiener as base in all cases)
    # Even if blur isn't strongly detected, apply mild Wiener deconvolution
    deblurred, deblur_info = adaptive_deblurring(denoised, degradation_info, params)
    results.update(deblur_info)
    results['deblurred'] = deblurred
    
    # Always store Wiener stage for analysis
    if deblur_info.get('wiener_stage') is not None:
        results['wiener'] = deblur_info['wiener_stage']
    
    # Step 4: Final enhancement
    # Adaptive sharpening based on degradation severity
    sharpening_factor = params['UNSHARP_AMOUNT'] * (1 + degradation_info['overall_degradation'])
    sharpening_factor = min(2.0, sharpening_factor)
    
    blurred = ndimage.gaussian_filter(deblurred, sigma=(params['UNSHARP_SIGMA'], 
                                                       params['UNSHARP_SIGMA'], 0))
    final = np.clip(deblurred + sharpening_factor * (deblurred - blurred), 0, 1)
    results['final'] = final
    results['sharpening_factor'] = sharpening_factor
    
    # Step 5: Post-processing for specific degradations
    if degradation_info['salt_pepper_present']:
        # Additional processing for salt & pepper artifacts
        final = enhance_salt_pepper_recovery(final, degradation_info)
        results['final'] = final
    
    return results

def enhance_salt_pepper_recovery(img_rgb: np.ndarray, degradation_info: Dict[str, Any]) -> np.ndarray:
    """Additional enhancement for salt & pepper noise recovery."""
    # Apply gentle bilateral filtering to smooth remaining artifacts
    img_u8 = (img_rgb * 255).astype('uint8')
    enhanced = cv2.bilateralFilter(img_u8, 5, 75, 75)
    return enhanced.astype('float32') / 255.0

def compute_comprehensive_metrics(gt_rgb: np.ndarray, input_rgb: np.ndarray, restored_rgb: np.ndarray):
    """Compute comprehensive quality metrics between images."""
    if input_rgb.shape[:2] != gt_rgb.shape[:2]:
        input_rgb = cv2.resize(input_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]))
    if restored_rgb.shape[:2] != gt_rgb.shape[:2]:
        restored_rgb = cv2.resize(restored_rgb, (gt_rgb.shape[1], gt_rgb.shape[0]))
    
    def to_y_channel(img):
        ycrcb = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2YCrCb)
        return ycrcb[..., 0].astype('float32') / 255.0
    
    gt_y = to_y_channel(gt_rgb)
    inp_y = to_y_channel(input_rgb)
    res_y = to_y_channel(restored_rgb)
    
    # PSNR
    psnr_input = sk_psnr(gt_y, inp_y, data_range=1.0)
    psnr_restored = sk_psnr(gt_y, res_y, data_range=1.0)
    
    # SSIM
    min_dim = min(gt_y.shape[0], gt_y.shape[1])
    win_size = min(7, min_dim) if min_dim % 2 == 1 else min(7, min_dim - 1)
    win_size = max(3, win_size)
    
    ssim_input = sk_ssim(gt_y, inp_y, data_range=1.0, win_size=win_size)
    ssim_restored = sk_ssim(gt_y, res_y, data_range=1.0, win_size=win_size)
    
    # MSE
    mse_input = np.mean((gt_y - inp_y) ** 2)
    mse_restored = np.mean((gt_y - res_y) ** 2)
    
    # Additional metrics
    from scipy import ndimage as ndi
    
    # Gradient magnitude for sharpness assessment
    def gradient_magnitude(img):
        grad_x = ndi.sobel(img, axis=1)
        grad_y = ndi.sobel(img, axis=0)
        return np.sqrt(grad_x**2 + grad_y**2)
    
    gt_grad = gradient_magnitude(gt_y)
    inp_grad = gradient_magnitude(inp_y)
    res_grad = gradient_magnitude(res_y)
    
    sharpness_input = np.mean(inp_grad)
    sharpness_restored = np.mean(res_grad)
    sharpness_gt = np.mean(gt_grad)
    
    return {
        'PSNR_input': psnr_input,
        'PSNR_final': psnr_restored,
        'SSIM_input': ssim_input,
        'SSIM_final': ssim_restored,
        'MSE_input': mse_input,
        'MSE_final': mse_restored,
        'Sharpness_input': sharpness_input,
        'Sharpness_final': sharpness_restored,
        'Sharpness_gt': sharpness_gt,
        'PSNR_improvement': psnr_restored - psnr_input,
        'SSIM_improvement': ssim_restored - ssim_input,
        'MSE_improvement': mse_input - mse_restored,  # Lower is better
        'Sharpness_improvement': sharpness_restored - sharpness_input
    }

def generate_synthetic_sample(size=(512, 512), sigma_noise=12.0):
    """Generate synthetic test image with ground truth"""
    # Use standard test images
    src_images = [data.astronaut(), data.coffee(), data.chelsea()]
    im = src_images[np.random.randint(0, len(src_images))]
    
    if im.ndim == 2:
        im_rgb = color.gray2rgb(img_as_float(im))
    else:
        im_rgb = img_as_float(im)
    
    h, w = im_rgb.shape[:2]
    th, tw = size
    
    if h < th or w < tw:
        im_res = resize(im_rgb, size, anti_aliasing=True)
    else:
        ch = (h-th)//2
        cw = (w-tw)//2
        im_res = im_rgb[ch:ch+th, cw:cw+tw]
    
    # Add random rotation
    if np.random.random() < 0.3:
        im_res = ndimage.rotate(im_res, np.random.uniform(-5, 5), reshape=False, mode='reflect')
    
    # Create blur - REDUCED BLUR STRENGTH
    blur_type = np.random.random()
    if blur_type < 0.6:
        length = np.random.uniform(5, 10)  # Reduced from 8-15
        angle = np.random.uniform(-30, 30)
        psf = motion_blur_psf(length, angle)
    elif blur_type < 0.8:
        radius = np.random.uniform(1, 3)  # Reduced from 2-4
        psf = disk_psf(radius)
    else:
        sigma = np.random.uniform(0.5, 1.5)  # Reduced from 1-2
        psf = gaussian_psf(sigma)
    
    blurred = fft_conv(im_res, psf)
    noisy = np.clip(blurred + np.random.normal(0.0, sigma_noise/255.0, blurred.shape), 0, 1)
    
    return noisy, im_res, psf

def create_psf_analysis_plot(psf, psf_len, psf_ang):
    """Create PSF analysis visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # PSF heatmap
    im1 = ax1.imshow(psf, cmap='viridis', interpolation='nearest')
    ax1.set_title(f'Estimated PSF\nLength: {psf_len:.1f}px, Angle: {psf_ang:.1f}°', fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Center profil
    # e
    center_row = psf[psf.shape[0]//2, :]
    ax2.plot(center_row, 'b-', linewidth=2, marker='o', markersize=3)
    ax2.fill_between(range(len(center_row)), center_row, alpha=0.3)
    ax2.set_title('Horizontal Center Profile')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Intensity')
    ax2.grid(True, alpha=0.3)
    
    # 3D surface plot
    x = np.arange(psf.shape[1])
    y = np.arange(psf.shape[0])
    X, Y = np.meshgrid(x, y)
    contour = ax3.contourf(X, Y, psf, levels=20, cmap='viridis')
    ax3.set_title('PSF Contour Plot')
    ax3.axis('off')
    plt.colorbar(contour, ax=ax3, fraction=0.046, pad=0.04)
    
    # Statistics
    ax4.axis('off')
    stats_text = (
        f"PSF Statistics:\n"
        f"• Size: {psf.shape}\n"
        f"• Sum: {psf.sum():.4f}\n"
        f"• Max: {psf.max():.4f}\n"
        f"• Min: {psf.min():.4f}\n"
        f"• Non-zero pixels: {np.count_nonzero(psf > 1e-6)}"
    )
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    plt.tight_layout()
    return fig

def create_restoration_pipeline_plot(results, metrics=None):
    """Create restoration pipeline visualization with ALL stages"""
    # Define all possible stages in processing order
    stages = {
        'Input': results.get('input'),
        'Denoised': results.get('denoised'),
        'Wiener Deconv': results.get('wiener'),  # This will always be populated now
        'Final': results.get('final')
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (stage_name, img) in enumerate(stages.items()):
        if img is not None:
            axes[idx].imshow(np.clip(img, 0, 1))
            axes[idx].set_title(stage_name, fontweight='bold', fontsize=12)
            axes[idx].axis('off')
        else:
            # Show placeholder for missing stages
            axes[idx].text(0.5, 0.5, f"{stage_name}\nNot Available", 
                          ha='center', va='center', transform=axes[idx].transAxes,
                          fontsize=12, fontweight='bold')
            axes[idx].set_facecolor('#f0f0f0')
            axes[idx].axis('off')
    
    # Add metrics if available
    if metrics:
        metrics_text = (
            f"PSNR: {metrics['PSNR_input']:.2f} → {metrics['PSNR_final']:.2f} dB\n"
            f"SSIM: {metrics['SSIM_input']:.4f} → {metrics['SSIM_final']:.4f}"
        )
        if results.get('psf_len') is not None:
            metrics_text += f"\nPSF: {results.get('psf_len', 0):.1f}px, {results.get('psf_ang', 0):.1f}°"
        if results.get('deblur_method'):
            metrics_text += f"\nMethod: {results.get('deblur_method')}"
        
        fig.text(0.02, 0.02, metrics_text, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    plt.suptitle("Restoration Pipeline", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

# =============================================================================
# DEGRADATION FUNCTIONS WITH REDUCED STRENGTH
# =============================================================================

def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Add Gaussian noise to image with reduced strength"""
    noise = np.random.normal(0, sigma/255.0, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0, 1)

def add_salt_pepper_noise(img: np.ndarray, amount: float, salt_prob: float = 0.5) -> np.ndarray:
    """Add salt and pepper noise to image with reduced strength"""
    noisy = img.copy()
    
    # Salt noise (white pixels)
    salt_mask = np.random.random(img.shape[:2]) < (amount * salt_prob)
    if img.ndim == 3:
        noisy[salt_mask] = [1, 1, 1]
    else:
        noisy[salt_mask] = 1
    
    # Pepper noise (black pixels)  
    pepper_mask = np.random.random(img.shape[:2]) < (amount * (1 - salt_prob))
    if img.ndim == 3:
        noisy[pepper_mask] = [0, 0, 0]
    else:
        noisy[pepper_mask] = 0
    
    return noisy

def apply_custom_blur(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply custom blur kernel to image with reduced strength"""
    if kernel.sum() == 0:
        return img.copy()
    
    # Normalize kernel
    kernel = kernel / kernel.sum()
    
    # Apply convolution with reduced strength (partial blur)
    blurred = fft_conv(img, kernel)
    # Mix with original to reduce blur strength
    alpha = 0.6  # Reduced from 1.0 to 0.6 for weaker blur
    result = alpha * blurred + (1 - alpha) * img
    return np.clip(result, 0, 1)

def create_average_kernel(size: int) -> np.ndarray:
    """Create average blur kernel"""
    kernel = np.ones((size, size), dtype=np.float32)
    kernel /= kernel.sum()
    return kernel

def create_motion_blur_kernel(length: int, angle: float) -> np.ndarray:
    """Create motion blur kernel with reduced strength"""
    psf = motion_blur_psf(length, angle)
    return psf

def create_gaussian_blur_kernel(size: int, sigma: float) -> np.ndarray:
    """Create Gaussian blur kernel with reduced strength"""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    kernel /= kernel.sum()
    return kernel

def toggle_kernel_cell(row: int, col: int):
    """Toggle kernel cell state"""
    st.session_state.custom_kernel[row, col] = 1.0 if st.session_state.custom_kernel[row, col] == 0.0 else 0.0

def reset_kernel():
    """Reset custom kernel to all zeros"""
    st.session_state.custom_kernel = np.zeros((7, 7), dtype=np.float32)

def create_kernel_visualization(kernel: np.ndarray) -> plt.Figure:
    """Create visualization for custom kernel"""
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(kernel, cmap='Blues', vmin=0, vmax=1)
    ax.set_title('Custom Kernel')
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.grid(True, alpha=0.3)
    
    # Add values to cells
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f'{kernel[i, j]:.1f}', 
                   ha='center', va='center', 
                   fontweight='bold',
                   color='white' if kernel[i, j] > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig

def create_degradation_visualization(degradation_info: Dict[str, Any]) -> np.ndarray:
    """Create a visualization of the degradation analysis."""
    fig, ax = plt.subplots(figsize=(4, 3))
    
    metrics = [
        ('Noise Level', degradation_info['noise_level']),
        ('Blur Severity', degradation_info['blur_severity']),
        ('Salt & Pepper', degradation_info['sp_ratio'] * 5),  # Scale for visibility
    ]
    
    names = [m[0] for m in metrics]
    values = [m[1] for m in metrics]
    
    bars = ax.bar(names, values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Severity')
    ax.set_title('Degradation Analysis')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Convert matplotlib figure to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

# =============================================================================
# IMAGE UTILITY FUNCTIONS
# =============================================================================

def ensure_image_range(img: np.ndarray) -> np.ndarray:
    """Ensure image is in valid range [0, 1] for display"""
    if img.dtype == np.float32 or img.dtype == np.float64:
        return np.clip(img, 0.0, 1.0)
    elif img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    else:
        return np.clip(img.astype(np.float32), 0.0, 1.0)

def safe_image_display(img: np.ndarray, caption: str = "", width: int = None):
    """Safely display image with proper range handling"""
    display_img = ensure_image_range(img)
    if width:
        st.image(display_img, caption=caption, width=width)
    else:
        st.image(display_img, caption=caption)

# =============================================================================
# STREAMLIT APP MAIN FUNCTION - FIXED ANALYSIS DISPLAY
# =============================================================================

def main():
    # Header with new name and tagline
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">🔮 VisionRestore Pro</h1>', unsafe_allow_html=True)
        st.markdown('<p class="app-subtitle">Intelligent Image Restoration with Custom Degradation Pipeline</p>', unsafe_allow_html=True)
    
    # Sidebar for parameters with improved organization and aesthetics
    st.sidebar.markdown('<div class="sidebar-header">⚙️ Control Panel</div>', unsafe_allow_html=True)





    
    # Processing Mode Section
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section-title">🎯 Processing Mode</div>', unsafe_allow_html=True)
    auto_adapt = st.sidebar.checkbox("Auto-Adapt Parameters", True, 
                                    help="Automatically adjust parameters based on image analysis")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Denoising Parameters
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section-title">🔇 Noise Reduction</div>', unsafe_allow_html=True)
    noise_sigma = st.sidebar.slider("Noise Level", 5.0, 25.0, 12.0, 1.0, 
                                   help="Estimated noise standard deviation")
    use_bm3d = st.sidebar.checkbox("Use BM3D Denoising", HAS_BM3D, disabled=not HAS_BM3D,
                                  help="Use BM3D for superior denoising")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Deblurring Parameters
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section-title">🌀 Deblurring</div>', unsafe_allow_html=True)
    wiener_balance = st.sidebar.slider("Wiener Balance", 0.001, 0.1, 0.01, 0.001,
                                      help="Regularization parameter for Wiener filter")
    rl_iters = st.sidebar.slider("RL Iterations", 30, 200, 100, 10,
                                help="Number of Richardson-Lucy iterations")
    min_psf_len = st.sidebar.slider("Min PSF Length", 0.5, 5.0, 1.5, 0.5,
                                   help="Minimum PSF length for RL deconvolution")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Enhancement Parameters
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section-title">✨ Enhancement</div>', unsafe_allow_html=True)
    tv_weight = st.sidebar.slider("TV Weight", 0.001, 0.05, 0.008, 0.001,
                                 help="Total Variation regularization strength")
    unsharp_amount = st.sidebar.slider("Unsharp Amount", 0.1, 1.0, 0.6, 0.1,
                                      help="Sharpening strength")
    unsharp_sigma = st.sidebar.slider("Unsharp Sigma", 0.5, 3.0, 1.0, 0.1,
                                     help="Sharpening radius")
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Display Options
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section-title">👁️ Display Options</div>', unsafe_allow_html=True)
    show_intermediate = st.sidebar.checkbox("Show Intermediate Steps", True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # System Status
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section-title">🔧 System Status</div>', unsafe_allow_html=True)
    
    # Display BM3D status
    if HAS_BM3D:
        if use_bm3d:
            st.sidebar.markdown('<div class="status-indicator status-success">✅ BM3D Enabled</div>', unsafe_allow_html=True)
        else:
            st.sidebar.markdown('<div class="status-indicator status-info">ℹ️ Wavelet Denoising</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-indicator status-warning">⚠️ BM3D Not Available</div>', unsafe_allow_html=True)
    
    # Processing status
    if st.session_state.restoration_results is not None:
        st.sidebar.markdown('<div class="status-indicator status-success">✅ Restoration Complete</div>', unsafe_allow_html=True)
        if st.session_state.processing_time:
            st.sidebar.markdown(f'<div style="text-align: center; color: var(--text-light); font-size: 0.8rem;">Processing time: {st.session_state.processing_time:.2f}s</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-indicator status-info">⏳ Ready for Processing</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Create parameters dictionary
    params = {
        'NOISE_SIGMA': noise_sigma,
        'WIENER_BAL': wiener_balance,
        'RL_ITERS': rl_iters,
        'TV_INTERVAL': 6,
        'TV_WEIGHT': tv_weight,
        'UNSHARP_AMOUNT': unsharp_amount,
        'UNSHARP_SIGMA': unsharp_sigma,
        'MIN_PSF_LEN': min_psf_len,
        'MAX_PSF_LEN': 80,
        'USE_BM3D': use_bm3d,
        'AUTO_ADAPT': auto_adapt
    }

    # Main content area with improved tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📥 Image Input", "🎨 Degrade Image", "🔄 Restoration", "📊 Analysis", "ℹ️ About"])

    with tab1:
        st.markdown('<div class="section-header">📥 Image Input</div>', unsafe_allow_html=True)
        
        # Two-column layout for input options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Upload Your Image")
            uploaded_file = st.file_uploader(
                "Choose an image for restoration", 
                type=['png', 'jpg', 'jpeg', 'bmp', 'tif'],
                help="Upload an image for restoration processing",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                # Load uploaded image
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                if img_array.ndim == 3:
                    img_rgb = img_array.astype(np.float32) / 255.0
                else:
                    img_rgb = color.gray2rgb(img_array).astype(np.float32) / 255.0
                
                st.session_state.uploaded_image = img_rgb
                st.session_state.clean_image = img_rgb.copy()  # Store as clean reference
                st.session_state.ground_truth = None
                
                # Show image info
                st.success(f"✅ Image loaded: {img_array.shape[1]}×{img_array.shape[0]} pixels")
        
        with col2:
            st.markdown("#### Quick Options")
            
            # Synthetic sample generation
            st.markdown("**Test with Sample**")
            if st.button("🎲 Generate Sample", use_container_width=True):
                with st.spinner("Creating synthetic sample..."):
                    blurred, ground_truth, _ = generate_synthetic_sample(sigma_noise=noise_sigma)
                    st.session_state.uploaded_image = blurred
                    st.session_state.clean_image = ground_truth.copy()  # Store clean version
                    st.session_state.ground_truth = ground_truth
                    st.session_state.restoration_results = None
                    st.success("Synthetic sample generated!")
            
            # Ground truth upload
            st.markdown("**Add Reference**")
            gt_file = st.file_uploader(
                "Upload reference image (optional)",
                type=['png', 'jpg', 'jpeg', 'bmp', 'tif'],
                key="gt_uploader",
                label_visibility="collapsed"
            )
            
            if gt_file is not None and st.session_state.uploaded_image is not None:
                gt_image = Image.open(gt_file)
                gt_array = np.array(gt_image)
                if gt_array.ndim == 3:
                    gt_rgb = gt_array.astype(np.float32) / 255.0
                else:
                    gt_rgb = color.gray2rgb(gt_array).astype(np.float32) / 255.0
                st.session_state.ground_truth = gt_rgb
                st.success("✅ Reference image loaded!")
            
            if st.session_state.uploaded_image is None:
                st.info("👆 Upload an image or generate a sample to get started")
        
        # Display uploaded/generated images
        if st.session_state.uploaded_image is not None:
            st.markdown("---")
            st.markdown("#### Image Preview")
            
            img_height, img_width = st.session_state.uploaded_image.shape[:2]
            display_width = min(600, img_width)
            
            cols = st.columns(2 if st.session_state.ground_truth is not None else 1)
            
            with cols[0]:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.markdown('<div class="image-title">📷 Input Image</div>', unsafe_allow_html=True)
                safe_image_display(st.session_state.uploaded_image, 
                                 f"Input Image ({img_width}×{img_height})",
                                 display_width)
                st.markdown('</div>', unsafe_allow_html=True)
            
            if st.session_state.ground_truth is not None:
                with cols[1]:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.markdown('<div class="image-title">🎯 Reference Image</div>', unsafe_allow_html=True)
                    safe_image_display(st.session_state.ground_truth, 
                                     "Reference (Ground Truth)",
                                     display_width)
                    st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="section-header">🎨 Custom Image Degradation</div>', unsafe_allow_html=True)
        
        if st.session_state.uploaded_image is None:
            st.info("👆 Please upload an image first in the 'Image Input' tab to start degradation.")
        else:
            # Initialize degradation state if needed
            if st.session_state.current_preview is None:
                st.session_state.current_preview = st.session_state.uploaded_image.copy()
                # Store the clean image if not already stored
                if st.session_state.clean_image is None:
                    st.session_state.clean_image = st.session_state.uploaded_image.copy()
                st.session_state.degradation_steps = []
            
            # Display current state
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Current Image State")
                safe_image_display(st.session_state.current_preview, 
                                 "Current Image (After Degradations)")
            
            with col2:
                st.markdown("#### Degradation History")
                if st.session_state.degradation_steps:
                    st.markdown('<div class="degradation-history">', unsafe_allow_html=True)
                    for i, step in enumerate(st.session_state.degradation_steps):
                        st.markdown(f'<div class="history-item">{i+1}. {step}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No degradations applied yet")
            
            st.markdown("---")
            
            # Degradation controls
            st.markdown("#### 🛠️ Degradation Tools")
            
            # Two columns for noise and blur controls
            col_noise, col_blur = st.columns(2)
            
            with col_noise:
                st.markdown("##### 🔊 Add Noise")
                
                noise_type = st.selectbox(
                    "Noise Type",
                    ["Gaussian", "Salt & Pepper"],  # REMOVED: "Mixed"
                    key="noise_type"
                )
                
                if noise_type == "Gaussian":
                    gauss_sigma = st.slider("Gaussian Sigma", 1.0, 30.0, 15.0, 1.0,  # Reduced max from 50.0 to 30.0
                                          help="Standard deviation of Gaussian noise")
                    
                    if st.button("➕ Add Gaussian Noise", use_container_width=True):
                        degraded = add_gaussian_noise(st.session_state.current_preview, gauss_sigma)
                        st.session_state.current_preview = degraded
                        st.session_state.degradation_steps.append(f"Gaussian Noise (σ={gauss_sigma:.1f})")
                        st.success(f"✅ Added Gaussian noise (σ={gauss_sigma:.1f})")
                        st.rerun()
                
                else:  # Salt & Pepper
                    sp_amount = st.slider("Noise Amount", 0.01, 0.1, 0.05, 0.01,  # Reduced max from 0.2 to 0.1
                                        help="Proportion of pixels affected")
                    salt_prob = st.slider("Salt Probability", 0.0, 1.0, 0.5, 0.1,
                                        help="Probability of salt (white) vs pepper (black)")
                    
                    if st.button("➕ Add Salt & Pepper Noise", use_container_width=True):
                        degraded = add_salt_pepper_noise(st.session_state.current_preview, sp_amount, salt_prob)
                        st.session_state.current_preview = degraded
                        st.session_state.degradation_steps.append(f"Salt & Pepper (amount={sp_amount:.2f}, salt={salt_prob:.1f})")
                        st.success(f"✅ Added Salt & Pepper noise")
                        st.rerun()
            
            with col_blur:
                st.markdown("##### 🌀 Add Blur")
                
                blur_type = st.selectbox(
                    "Blur Type",
                    ["Motion Blur", "Gaussian Blur", "Average Blur", "Custom Kernel"],
                    key="blur_type"
                )
                
                if blur_type == "Motion Blur":
                    motion_length = st.slider("Motion Length", 3, 15, 8, 1,  # Reduced max from 30 to 15
                                            help="Length of motion blur")
                    motion_angle = st.slider("Motion Angle", -90, 90, 0, 5)
                    
                    if st.button("➕ Add Motion Blur", use_container_width=True):
                        kernel = create_motion_blur_kernel(motion_length, motion_angle)
                        degraded = apply_custom_blur(st.session_state.current_preview, kernel)
                        st.session_state.current_preview = degraded
                        st.session_state.degradation_steps.append(f"Motion Blur (length={motion_length}, angle={motion_angle}°)")
                        st.success(f"✅ Added motion blur")
                        st.rerun()
                
                elif blur_type == "Gaussian Blur":
                    gauss_size = st.slider("Kernel Size", 3, 9, 5, 2,  # Reduced max from 15 to 9
                                         help="Size of Gaussian kernel")
                    gauss_sigma = st.slider("Gaussian Sigma", 0.5, 2.5, 1.0, 0.1,  # Reduced max from 5.0 to 2.5
                                          help="Standard deviation of Gaussian blur")
                    
                    if st.button("➕ Add Gaussian Blur", use_container_width=True):
                        kernel = create_gaussian_blur_kernel(gauss_size, gauss_sigma)
                        degraded = apply_custom_blur(st.session_state.current_preview, kernel)
                        st.session_state.current_preview = degraded
                        st.session_state.degradation_steps.append(f"Gaussian Blur (size={gauss_size}, σ={gauss_sigma:.1f})")
                        st.success("✅ Added Gaussian blur")
                        st.rerun()
                
                elif blur_type == "Average Blur":
                    avg_size = st.slider("Kernel Size", 3, 9, 5, 2,  # Reduced max from 15 to 9
                                       help="Size of average blur kernel")
                    
                    if st.button("➕ Add Average Blur", use_container_width=True):
                        kernel = create_average_kernel(avg_size)
                        degraded = apply_custom_blur(st.session_state.current_preview, kernel)
                        st.session_state.current_preview = degraded
                        st.session_state.degradation_steps.append(f"Average Blur (size={avg_size})")
                        st.success("✅ Added average blur")
                        st.rerun()
                
                else:  # Custom Kernel
                    st.markdown("**Draw Custom 7×7 Kernel**")
                    
                    # Add sigma control for custom kernel
                    custom_sigma = st.slider("Blur Strength", 0.1, 2.0, 1.0, 0.1,
                                        help="Strength of the custom blur effect")
                    
                    # Kernel visualization
                    fig = create_kernel_visualization(st.session_state.custom_kernel)
                    st.pyplot(fig)
                    
                    # Kernel drawing interface
                    st.markdown("Click cells to toggle ON/OFF:")
                    for i in range(7):
                        cols = st.columns(7)
                        for j in range(7):
                            with cols[j]:
                                is_active = st.session_state.custom_kernel[i, j] > 0
                                label = "█" if is_active else "□"
                                if st.button(label, key=f"kernel_{i}_{j}", use_container_width=True):
                                    toggle_kernel_cell(i, j)
                                    st.rerun()
                    
                    col_reset, col_apply = st.columns(2)
                    with col_reset:
                        if st.button("🔄 Reset Kernel", use_container_width=True):
                            reset_kernel()
                            st.rerun()
                    
                    with col_apply:
                        if st.button("➕ Apply Custom Blur", use_container_width=True):
                            if np.sum(st.session_state.custom_kernel) > 0:
                                # Apply the custom kernel with the specified sigma strength
                                kernel = st.session_state.custom_kernel.copy()
                                # Normalize the kernel
                                if kernel.sum() > 0:
                                    kernel = kernel / kernel.sum()
                                # Apply with strength control
                                blurred = fft_conv(st.session_state.current_preview, kernel)
                                # Mix with original based on sigma strength
                                alpha = min(1.0, custom_sigma * 0.5)  # Scale sigma to alpha
                                degraded = alpha * blurred + (1 - alpha) * st.session_state.current_preview
                                st.session_state.current_preview = np.clip(degraded, 0, 1)
                                active_cells = np.sum(st.session_state.custom_kernel > 0)
                                st.session_state.degradation_steps.append(f"Custom Blur ({active_cells} active cells, σ={custom_sigma:.1f})")
                                st.success("✅ Applied custom blur")
                                st.rerun()
                            else:
                                st.error("❌ Kernel is empty! Please draw a kernel first.")
            
            st.markdown("---")
            
            # Management controls
            st.markdown("#### 🎛️ Management")
            col_manage1, col_manage2, col_manage3 = st.columns(3)
            
            with col_manage1:
                if st.button("🔄 Reset to Clean", use_container_width=True):
                    st.session_state.current_preview = st.session_state.clean_image.copy()
                    st.session_state.degradation_steps = []
                    st.success("✅ Reset to clean image")
                    st.rerun()
            
            with col_manage2:
                if st.button("↩️ Undo Last Step", use_container_width=True):
                    if st.session_state.degradation_steps:
                        # Simple approach: reset to clean and reapply all but last step
                        current_steps = st.session_state.degradation_steps[:-1]
                        temp = st.session_state.clean_image.copy()
                        
                        # Reapply all steps except the last one
                        for step in current_steps:
                            if "Gaussian Noise" in step:
                                sigma = float(step.split("σ=")[1].split(")")[0])
                                temp = add_gaussian_noise(temp, sigma)
                            elif "Salt & Pepper" in step:
                                # Extract parameters from step description
                                if "amount=" in step and "salt=" in step:
                                    amount_str = step.split("amount=")[1].split(",")[0]
                                    salt_str = step.split("salt=")[1].split(")")[0]
                                    amount = float(amount_str)
                                    salt_prob = float(salt_str)
                                    temp = add_salt_pepper_noise(temp, amount, salt_prob)
                            elif "Motion Blur" in step:
                                # For simplicity, just apply a default motion blur
                                temp = apply_custom_blur(temp, create_motion_blur_kernel(10, 0))
                            elif "Gaussian Blur" in step:
                                temp = apply_custom_blur(temp, create_gaussian_blur_kernel(5, 1.5))
                            elif "Average Blur" in step:
                                size = int(step.split("size=")[1].split(")")[0])
                                temp = apply_custom_blur(temp, create_average_kernel(size))
                        
                        st.session_state.current_preview = temp
                        st.session_state.degradation_steps = current_steps
                        st.success("✅ Undid last step")
                        st.rerun()
                    else:
                        st.info("ℹ️ No steps to undo")
            
            with col_manage3:
                if st.button("✅ Set as Input", use_container_width=True, type="primary"):
                    # Store the degraded image for restoration
                    st.session_state.degraded_image = st.session_state.current_preview.copy()
                    
                    # Set as current input for restoration
                    st.session_state.uploaded_image = st.session_state.current_preview.copy()
                    
                    # Ensure we have ground truth for metrics
                    if st.session_state.ground_truth is None and st.session_state.clean_image is not None:
                        st.session_state.ground_truth = st.session_state.clean_image.copy()
                    
                    st.session_state.restoration_results = None
                    st.success("✅ Degraded image set as input for restoration!")
                    st.balloons()

    with tab3:
        st.markdown('<div class="section-header">🔄 Image Restoration</div>', unsafe_allow_html=True)
        
        # FIXED: Use degraded image if available, otherwise use uploaded image
        current_input_image = None
        if st.session_state.get('degraded_image') is not None:
            current_input_image = st.session_state.degraded_image
            st.info("🎨 Using degraded image from Degradation tab as input")
        elif st.session_state.uploaded_image is not None:
            current_input_image = st.session_state.uploaded_image
        else:
            current_input_image = None
        
        if current_input_image is not None:
            # Display current input image info
            img_height, img_width = current_input_image.shape[:2]
            st.markdown(f"**Input Image:** {img_width}×{img_height} pixels")
            
            if st.session_state.get('degraded_image') is not None:
                st.success("✅ Processing custom degraded image")
            
            # Restoration controls
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Start Restoration", type="primary", use_container_width=True):
                    with st.spinner("Performing advanced degradation-aware restoration..."):
                        try:
                            # Perform enhanced restoration
                            start_time = time.time()
                            results = restore_image_with_degradation_awareness(current_input_image, params)
                            processing_time = time.time() - start_time
                            
                            st.session_state.restoration_results = results
                            st.session_state.processing_time = processing_time
                            
                            # Compute metrics if ground truth available
                            ground_truth_ref = st.session_state.ground_truth
                            if ground_truth_ref is None and st.session_state.get('clean_image') is not None:
                                ground_truth_ref = st.session_state.clean_image
                            
                            if ground_truth_ref is not None:
                                metrics = compute_comprehensive_metrics(
                                    ground_truth_ref,
                                    current_input_image,
                                    results['final']
                                )
                                st.session_state.metrics = metrics
                            else:
                                st.session_state.metrics = None
                            
                            st.success(f"✅ Restoration completed in {processing_time:.2f} seconds!")
                            
                        except Exception as e:
                            st.error(f"❌ Restoration failed: {str(e)}")
            
            # Display degradation analysis if available
            if st.session_state.restoration_results and 'degradation_info' in st.session_state.restoration_results:
                degradation_info = st.session_state.restoration_results['degradation_info']
                st.markdown("#### 🔍 Degradation Analysis")
                
                cols = st.columns(4)
                with cols[0]:
                    noise_level = degradation_info['noise_level']
                    noise_status = "🟢 Low" if noise_level < 0.3 else "🟡 Medium" if noise_level < 0.6 else "🔴 High"
                    st.metric("Noise Level", noise_status)
                
                with cols[1]:
                    blur_status = "✅ None" if not degradation_info['blur_detected'] else "🟡 Mild" if degradation_info['blur_severity'] < 0.5 else "🔴 Severe"
                    st.metric("Blur Detection", blur_status)
                
                with cols[2]:
                    sp_status = "✅ No" if not degradation_info['salt_pepper_present'] else f"🔴 Yes ({degradation_info['sp_ratio']:.3f})"
                    st.metric("Salt & Pepper", sp_status)
                
                with cols[3]:
                    overall_status = "🟢 Good" if degradation_info['overall_degradation'] < 0.3 else "🟡 Moderate" if degradation_info['overall_degradation'] < 0.6 else "🔴 Severe"
                    st.metric("Overall State", overall_status)
            
            # Display results if available
            if st.session_state.restoration_results is not None:
                st.markdown("---")
                st.markdown("#### Restoration Results")
                
                # Calculate display sizes
                display_width = min(400, img_width)
                
                # FIXED: Improved logic for third image display
                # Check what reference images are available
                has_ground_truth = st.session_state.ground_truth is not None
                has_clean_image = st.session_state.get('clean_image') is not None
                
                if has_ground_truth or has_clean_image:
                    # We have a reference image, use 3-column layout
                    cols = st.columns(3)
                    
                    with cols[0]:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.markdown('<div class="image-title">📷 Input Image</div>', unsafe_allow_html=True)
                        safe_image_display(current_input_image, width=display_width)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with cols[1]:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.markdown('<div class="image-title">✨ Restored Image</div>', unsafe_allow_html=True)
                        safe_image_display(st.session_state.restoration_results['final'], width=display_width)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with cols[2]:
                        # FIXED: Clear logic for third image
                        if has_ground_truth:
                            ref_image = st.session_state.ground_truth
                            title = "🎯 Ground Truth"
                            description = "Original reference image for comparison"
                        elif has_clean_image:
                            ref_image = st.session_state.clean_image
                            title = "🔄 Clean Original"
                            description = "Original image before degradation"
                        
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.markdown(f'<div class="image-title">{title}</div>', unsafe_allow_html=True)
                        safe_image_display(ref_image, width=display_width)
                        st.markdown(f'<div style="font-size: 0.8rem; color: var(--text-light); margin-top: 0.5rem;">{description}</div>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # No reference available, use 2-column layout and show degradation analysis
                    cols = st.columns(2)
                    
                    with cols[0]:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.markdown('<div class="image-title">📷 Input Image</div>', unsafe_allow_html=True)
                        safe_image_display(current_input_image, width=display_width)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with cols[1]:
                        st.markdown('<div class="image-container">', unsafe_allow_html=True)
                        st.markdown('<div class="image-title">✨ Restored Image</div>', unsafe_allow_html=True)
                        safe_image_display(st.session_state.restoration_results['final'], width=display_width)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show degradation analysis in the third spot
                    st.markdown("#### 🔍 Degradation Visualization")
                    if st.session_state.restoration_results.get('degradation_info'):
                        degradation_map = create_degradation_visualization(
                            st.session_state.restoration_results['degradation_info']
                        )
                        st.image(degradation_map, caption="Degradation Analysis Chart", use_column_width=True)
            
            # Download section
            st.markdown("---")
            st.markdown("#### Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Download Restored Image", use_container_width=True):
                    final_img = (st.session_state.restoration_results['final'] * 255).astype(np.uint8)
                    pil_img = Image.fromarray(final_img)
                    
                    buf = io.BytesIO()
                    pil_img.save(buf, format='PNG')
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="Download PNG",
                        data=byte_im,
                        file_name="restored_image.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            with col2:
                if st.session_state.metrics is not None:
                    if st.button("📊 Download Metrics Report", use_container_width=True):
                        # Create metrics report
                        metrics_df = pd.DataFrame([st.session_state.metrics])
                        csv = metrics_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name="restoration_metrics.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        else:
            st.info("👆 Please upload an image or create a degraded image first.")

    with tab4:
        st.markdown('<div class="section-header">📊 Detailed Analysis</div>', unsafe_allow_html=True)
        
        if st.session_state.restoration_results is not None:
            # PSF Analysis Dropdown
            st.markdown("#### 🔍 PSF Analysis Options")
            st.session_state.show_psf_analysis = st.checkbox(
                "Show Detailed PSF Analysis", 
                value=st.session_state.show_psf_analysis,
                help="Display detailed Point Spread Function analysis plots"
            )
            
            # Metrics display with improved layout
            if st.session_state.metrics is not None:
                st.markdown("#### 📈 Quality Metrics")
                
                metrics = st.session_state.metrics
                
                # Main improvement metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">PSNR Improvement</div>', unsafe_allow_html=True)
                    psnr_imp = metrics['PSNR_improvement']
                    imp_class = "improvement-positive" if psnr_imp > 0 else "improvement-negative"
                    st.markdown(f'<div class="metric-value {imp_class}">{psnr_imp:+.2f} dB</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size: 0.8rem; color: var(--text-light);">({metrics["PSNR_input"]:.2f} → {metrics["PSNR_final"]:.2f} dB)</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">SSIM Improvement</div>', unsafe_allow_html=True)
                    ssim_imp = metrics['SSIM_improvement']
                    imp_class = "improvement-positive" if ssim_imp > 0 else "improvement-negative"
                    st.markdown(f'<div class="metric-value {imp_class}">{ssim_imp:+.4f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size: 0.8rem; color: var(--text-light);">({metrics["SSIM_input"]:.4f} → {metrics["SSIM_final"]:.4f})</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">MSE Improvement</div>', unsafe_allow_html=True)
                    mse_imp = metrics['MSE_improvement']
                    imp_class = "improvement-positive" if mse_imp > 0 else "improvement-negative"
                    st.markdown(f'<div class="metric-value {imp_class}">{mse_imp:+.6f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size: 0.8rem; color: var(--text-light);">({metrics["MSE_input"]:.6f} → {metrics["MSE_final"]:.6f})</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<div class="metric-label">Sharpness Improvement</div>', unsafe_allow_html=True)
                    sharp_imp = metrics['Sharpness_improvement']
                    imp_class = "improvement-positive" if sharp_imp > 0 else "improvement-negative"
                    st.markdown(f'<div class="metric-value {imp_class}">{sharp_imp:+.4f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size: 0.8rem; color: var(--text-light);">({metrics["Sharpness_input"]:.4f} → {metrics["Sharpness_final"]:.4f})</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed metrics table
                st.markdown("#### 📋 Detailed Metrics")
                detailed_metrics = {
                    'Metric': ['PSNR (dB)', 'SSIM', 'MSE', 'Sharpness'],
                    'Input': [
                        f"{metrics['PSNR_input']:.2f}",
                        f"{metrics['SSIM_input']:.4f}",
                        f"{metrics['MSE_input']:.6f}",
                        f"{metrics['Sharpness_input']:.4f}"
                    ],
                    'Restored': [
                        f"{metrics['PSNR_final']:.2f}",
                        f"{metrics['SSIM_final']:.4f}",
                        f"{metrics['MSE_final']:.6f}",
                        f"{metrics['Sharpness_final']:.4f}"
                    ],
                    'Improvement': [
                        f"{metrics['PSNR_improvement']:+.2f}",
                        f"{metrics['SSIM_improvement']:+.4f}",
                        f"{metrics['MSE_improvement']:+.6f}",
                        f"{metrics['Sharpness_improvement']:+.4f}"
                    ]
                }
                
                metrics_df = pd.DataFrame(detailed_metrics)
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            # PSF Analysis (Conditional Display)
            if st.session_state.show_psf_analysis and st.session_state.restoration_results.get('psf') is not None:
                st.markdown("#### 🔍 PSF Analysis")
                psf = st.session_state.restoration_results['psf']
                psf_len = st.session_state.restoration_results['psf_len']
                psf_ang = st.session_state.restoration_results['psf_ang']
                
                fig_psf = create_psf_analysis_plot(psf, psf_len, psf_ang)
                st.pyplot(fig_psf)
            
            # Algorithm Insights
            st.markdown("#### 🧠 Algorithm Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**⚙️ Processing Details**")
                
                results = st.session_state.restoration_results
                details = [
                    f"**PSF Length:** `{results.get('psf_len', 0):.1f} pixels`",
                    f"**PSF Angle:** `{results.get('psf_ang', 0):.1f}°`",
                    f"**Deblur Method:** `{results.get('deblur_method', 'Not Applied')}`"
                ]
                
                if results.get('used_rl', False):
                    details.append(f"**RL Iterations:** `{results.get('adaptive_iters', 0)}`")
                
                if results.get('wiener_balance'):
                    details.append(f"**Wiener Balance:** `{results.get('wiener_balance', 0):.4f}`")
                
                details.append(f"**Processing Time:** `{st.session_state.get('processing_time', 0):.2f} seconds`")
                
                for detail in details:
                    st.markdown(detail)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("**🎯 Adaptive Parameters**")
                
                if st.session_state.restoration_results.get('degradation_info'):
                    deg_info = st.session_state.restoration_results['degradation_info']
                    adaptive_details = [
                        f"**Noise Level:** `{deg_info['noise_level']:.3f}`",
                        f"**Blur Severity:** `{deg_info['blur_severity']:.3f}`",
                        f"**Salt & Pepper:** `{'Yes' if deg_info['salt_pepper_present'] else 'No'}`",
                        f"**Sharpening Factor:** `{results.get('sharpening_factor', 0):.2f}`"
                    ]
                    
                    for detail in adaptive_details:
                        st.markdown(detail)
                else:
                    st.markdown("No degradation information available")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Pipeline Stages - FIXED: Now properly shows all 4 images
            if show_intermediate:
                st.markdown("#### 🔄 Pipeline Stages")
                st.info("This shows the complete restoration pipeline with all intermediate stages")
                fig_pipeline = create_restoration_pipeline_plot(
                    st.session_state.restoration_results, 
                    st.session_state.metrics
                )
                st.pyplot(fig_pipeline)
                st.caption("The pipeline shows: Input → Denoised → Wiener Deconvolution → Final Result")
        
        else:
            st.info("👆 Run restoration first to see detailed analysis.")

    with tab5:
        st.markdown('<div class="section-header">ℹ️ About VisionRestore Pro</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🚀 Advanced Image Restoration Solution
        
        **VisionRestore Pro** is a state-of-the-art application for restoring images using advanced computational photography techniques.
        
        #### 🎯 Key Features:
        
        - **🤖 Adaptive Intelligence**: Automatically adjusts parameters based on image analysis
        - **🔍 Robust Analysis**: Advanced algorithms for image understanding
        - **🔄 Multi-Stage Pipeline**: Comprehensive processing pipeline
        - **📊 Quality Metrics**: Comprehensive quality evaluation
        - **🎨 Modern UI/UX**: Clean, responsive interface with theme support
        - **🎛️ Custom Degradation**: Create custom degraded images for testing
        
        #### 🛠️ Technical Highlights:
        
        - **Texture-Aware Processing**: Adapts to image characteristics
        - **Intelligent Algorithms**: Advanced restoration techniques
        - **Luminance-Channel Processing**: Optimized noise reduction
        - **TV Regularization**: Reduces artifacts while preserving edges
        - **Custom Degradation Pipeline**: Create realistic test scenarios
        
        #### 📈 Performance Optimization:
        
        - **Auto-Parameter Tuning**: Based on image analysis
        - **Adaptive Processing**: Adjusts based on content complexity
        - **Efficient Algorithms**: Optimized for performance
        - **Memory Optimization**: Handles large images efficiently
        """)
        
        # System information
        st.markdown("#### 🔧 System Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("BM3D Available", "✅ Yes" if HAS_BM3D else "❌ No")
        with col2:
            st.metric("OpenCV Version", cv2.__version__)
        with col3:
            st.metric("Streamlit Version", st.__version__)

if __name__ == "__main__":
    main()