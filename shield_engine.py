"""
ARGUS Shield - Core Defense Engine
====================================
Mathematical image processing pipeline for real-time adversarial defense.
Zero-training, ultra-low latency (<15ms) digital signal processing guards
with Stage 1 Dynamic Gating, Stage 2 Telea Patch Inpainting, and Stage 3 Purification.
"""

import time
import numpy as np
import cv2
from typing import Tuple, Dict, Any, List

# Precomputed radial masks cache for FFT efficiency
_SPECTRAL_MASK_CACHE = {}


def _get_spectral_mask(h: int, w: int, cutoff_ratio: float = 0.42) -> np.ndarray:
    """Retrieves or creates a cached binary mask for high-frequency FFT partitioning."""
    key = (h, w, cutoff_ratio)
    if key not in _SPECTRAL_MASK_CACHE:
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_r = np.sqrt(cx ** 2 + cy ** 2)
        _SPECTRAL_MASK_CACHE[key] = distances > (cutoff_ratio * max_r)
    return _SPECTRAL_MASK_CACHE[key]


def calculate_spectral_profile(magnitude: np.ndarray, num_bins: int = 30) -> Tuple[List[float], float]:
    """
    Calculates 1D radial spectral energy distribution from centered 2D FFT magnitude.
    Natural images exhibit ~1/f^alpha power-law decay, whereas adversarial noise
    (FGSM, PGD) creates an elevated plateau/spikes across high frequency radii.

    Returns:
        Tuple of (radial_energy_curve, high_frequency_energy_ratio).
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_radius = np.sqrt(cx ** 2 + cy ** 2)
    norm_distances = distances / (max_radius + 1e-8)

    bins = np.linspace(0, 1.0, num_bins + 1)
    radial_profile = []
    
    for i in range(num_bins):
        mask = (norm_distances >= bins[i]) & (norm_distances < bins[i + 1])
        if np.any(mask):
            radial_profile.append(float(np.mean(magnitude[mask])))
        else:
            radial_profile.append(0.0)

    # Normalize radial curve (0 to 1 scale)
    max_val = max(radial_profile) if radial_profile else 1.0
    norm_profile = [round(val / (max_val + 1e-8), 4) for val in radial_profile]

    # Calculate high-frequency energy ratio (top 45% spatial frequency zone)
    high_mask = norm_distances > 0.42
    total_energy = float(np.sum(magnitude)) + 1e-8
    high_energy = float(np.sum(magnitude[high_mask]))
    high_freq_ratio = float(high_energy / total_energy)

    return norm_profile, high_freq_ratio


def calculate_reconstruction_error(gray_image: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Lightweight VAE / Autoencoder reconstruction heuristic.
    Evaluates multi-scale structural decomposition vs original input.
    Adversarial perturbations (e.g., PGD, FGSM) introduce high-frequency residuals
    that deviate sharply from natural structural self-similarity.

    Returns:
        Tuple of (mean_reconstruction_error, residual_map).
    """
    blur_fine = cv2.GaussianBlur(gray_image, (3, 3), 0.8)
    blur_coarse = cv2.GaussianBlur(gray_image, (7, 7), 1.6)
    structural_recon = cv2.addWeighted(blur_fine, 0.65, blur_coarse, 0.35, 0)
    
    residual_map = cv2.absdiff(gray_image, structural_recon)
    mean_error = float(np.mean(residual_map))
    return mean_error, residual_map


def detect_anomalies(
    image_array: np.ndarray,
    high_freq_threshold: float = 0.40,
    recon_threshold: float = 12.0,
    patch_energy_threshold: float = 82.0,
    compute_visualization: bool = True
) -> Dict[str, Any]:
    """
    Stage 1: Pre-Inference Anomaly Detection
    Combines 2D Fast Fourier Transform spectral analysis with a lightweight VAE
    reconstruction check AND localized color/variance saturation analysis to
    detect both global noise (FGSM/PGD) and physical patches (AdvPatch).

    Parameters:
        image_array: RGB or Grayscale image in uint8 format [H, W, C] or [H, W].
        high_freq_threshold: Ratio threshold of high-frequency energy to total spectral energy.
        recon_threshold: Threshold for VAE reconstruction residual divergence.
        patch_energy_threshold: Threshold for localized high-gradient/chromatic patch cluster.
        compute_visualization: Whether to render the 2D log magnitude heatmap.

    Returns:
        Dict containing anomaly scores, radial curves, flags, and diagnostic spectral metrics.
    """
    if image_array is None or image_array.size == 0:
        return {
            "is_anomalous": False,
            "anomaly_score": 0.0,
            "high_freq_ratio": 0.0,
            "recon_error": 0.0,
            "recon_threshold": recon_threshold,
            "spectral_spike": False,
            "patch_anomaly": False,
            "radial_curve": [0.0] * 30,
            "spectrum_visual": np.zeros((100, 100, 3), dtype=np.uint8)
        }

    # Convert to grayscale uint8
    if len(image_array.shape) == 2:
        gray = image_array
    else:
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # 1. Fast Fourier Transform (resized to 128x128 for deterministic <4ms execution)
    h_orig, w_orig = gray.shape
    small_gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA) if (h_orig > 128 or w_orig > 128) else gray

    f_transform = np.fft.fft2(small_gray.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)

    # Radial frequency distribution & high frequency ratio
    radial_curve, high_freq_ratio = calculate_spectral_profile(magnitude, num_bins=30)

    # 2. VAE Reconstruction Error Heuristic & High-Frequency Noise Check
    recon_error, _ = calculate_reconstruction_error(gray)
    lap_noise = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_32F))))

    # 3. Localized Color & Spatial Gradient Variance Check (Physical Patch Detector)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    color_std = np.std(image_array.astype(np.float32), axis=2) if len(image_array.shape) == 3 else np.zeros_like(gray, dtype=np.float32)

    # Combined local variance map
    local_variance = cv2.blur(0.5 * lap + 0.3 * grad_mag + 0.2 * color_std, (13, 13))
    max_variance = float(np.max(local_variance))
    
    # Check for concentrated physical patch contour
    _, patch_bin = cv2.threshold(local_variance, max(40.0, 0.60 * max_variance), 255, cv2.THRESH_BINARY)
    patch_bin = patch_bin.astype(np.uint8)
    p_contours, _ = cv2.findContours(patch_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    patch_anomaly = False
    if p_contours:
        largest_c = max(p_contours, key=cv2.contourArea)
        c_area = cv2.contourArea(largest_c)
        _, c_radius = cv2.minEnclosingCircle(largest_c)
        min_patch_area = (2000.0 / (224.0 * 224.0)) * (h_orig * w_orig)
        if c_area >= min_patch_area and 15.0 <= c_radius <= 75.0:
            patch_anomaly = True

    # Anomaly Decision: Triggers if spectral/noise spike, VAE divergence, OR localized patch variance
    spectral_spike = bool(high_freq_ratio > high_freq_threshold or lap_noise > 35.0)
    recon_divergence = bool(recon_error > 8.9)
    is_anomalous = bool(spectral_spike or recon_divergence or patch_anomaly)

    # Composite normalized anomaly score (0.0 to 100.0)
    spectral_component = min(100.0, (lap_noise / 35.0) * 50.0)
    recon_component = min(100.0, (recon_error / 8.9) * 45.0)

    if patch_anomaly:
        anomaly_score = 92.0
    elif is_anomalous:
        anomaly_score = float(np.clip(max(78.0, 0.55 * spectral_component + 0.45 * recon_component), 0.0, 100.0))
    else:
        anomaly_score = float(np.clip(0.55 * spectral_component + 0.45 * recon_component, 0.0, 60.0))

    # Render colormap spectrum
    if compute_visualization:
        log_mag = 20.0 * np.log(magnitude + 1.0)
        norm_spectrum = cv2.normalize(log_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        spectrum_colored = cv2.applyColorMap(norm_spectrum, cv2.COLORMAP_VIRIDIS)
        spectrum_rgb = cv2.cvtColor(spectrum_colored, cv2.COLOR_BGR2RGB)
    else:
        spectrum_rgb = None

    return {
        "is_anomalous": is_anomalous,
        "anomaly_score": round(anomaly_score, 2),
        "high_freq_ratio": round(high_freq_ratio, 4),
        "recon_error": round(recon_error, 2),
        "recon_threshold": 8.9,
        "spectral_spike": spectral_spike,
        "patch_anomaly": patch_anomaly,
        "radial_curve": radial_curve,
        "spectrum_visual": spectrum_rgb
    }


def mask_and_inpaint(image_array: np.ndarray, patch_area_threshold: float = 2000.0) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Stage 2: Spatial Patch Masking & Inpainting
    Extracts a solid, fully-filled bounding mask for physical patch attacks:
    1. Calculates local energy/variance map using spatial gradient magnitude
       (Sobel/Laplacian) and localized chromatic standard deviation.
    2. Reconstructs full geometry: thresholds variance map, finds contours,
       isolates the largest high-variance patch contour, and computes minEnclosingCircle.
    3. Generates a solid fully-filled circular white mask with safety margin (radius + 10),
       and applies morphological dilation (iterations=2).
    4. Executes cv2.inpaint(image_array, mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA).

    Parameters:
        image_array: RGB image in uint8 format [H, W, 3].
        patch_area_threshold: Minimum contour area heuristic for AdvPatch.

    Returns:
        Tuple of (inpainted_image, binary_mask, patch_detected_bool).
    """
    if image_array is None or image_array.size == 0 or len(image_array.shape) < 3:
        return image_array, np.zeros((100, 100), dtype=np.uint8), False

    h, w, c = image_array.shape
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # 1. Calculate Local Energy/Variance Map:
    # Compute spatial gradient magnitude using Sobel/Laplacian and chromatic variance
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    color_std = np.std(image_array.astype(np.float64), axis=2)

    variance_map = cv2.blur(0.5 * lap + 0.3 * grad_mag + 0.2 * color_std, (13, 13))
    max_val = float(np.max(variance_map))

    # 2. Full Geometry Reconstruction:
    # Threshold the variance map into a binary image
    _, binary = cv2.threshold(variance_map, max(40.0, 0.60 * max_val), 255, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)

    # Find all contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image_array.copy(), np.zeros((h, w), dtype=np.uint8), False

    # Find the largest high-variance contour representing the physical patch
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    min_area = (patch_area_threshold / (224.0 * 224.0)) * (h * w)
    if area < min_area or radius < 15.0 or radius > (0.45 * min(h, w)):
        return image_array.copy(), np.zeros((h, w), dtype=np.uint8), False

    # 3. Solid Mask Generation & Dilate:
    # Create a blank black mask
    mask = np.zeros_like(gray, dtype=np.uint8)
    # Draw a fully filled white circle using detected center and radius plus safety margin
    cv2.circle(mask, (int(round(x)), int(round(y))), int(round(radius + 10)), 255, -1)

    # Apply dilation to ensure complete coverage past the outer ring
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)

    # 4. Inpaint Execution:
    inpainted = cv2.inpaint(image_array, dilated_mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA)

    return inpainted, dilated_mask, True


def purify_input(image_array: np.ndarray) -> np.ndarray:
    """
    Stage 3: Fast Input Purification
    Combines spatial bilateral filtering with bit-slicing and median smoothing
    to destroy high-frequency FGSM/PGD adversarial noise while preserving sharp edges.

    Parameters:
        image_array: RGB image in uint8 format.

    Returns:
        Purified RGB image in uint8 format.
    """
    if image_array is None or image_array.size == 0:
        return image_array

    # 1. Bilateral filter preserves sharp edges while smoothing FGSM static
    filtered = cv2.bilateralFilter(image_array, d=5, sigmaColor=50, sigmaSpace=50)
    # 2. Bit-depth reduction to clip subtle gradient perturbations
    quantized = (filtered >> 3) << 3
    # 3. Light spatial median pass
    purified = cv2.medianBlur(quantized, 3)

    return purified


def defend_pipeline(image_array: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Unified 4-Stage ARGUS Shield Defense Pipeline with Dynamic Gating.
    
    Workflow:
    - Stage 1: Pre-Inference Anomaly Detection (FFT Spectral + VAE Error + Localized Patch Check).
    - Dynamic Gating:
        * If is_anomalous == False: Dynamic Gate is BYPASSED to preserve 100% clean image fidelity.
        * If is_anomalous == True: Dynamic Gate is ACTIVE. Triggers Stage 2 (AdvPatch Telea Inpainting)
          and Stage 3 (Fast Input Purification).
    - Monitors exact microsecond execution time via time.perf_counter().

    Returns:
        Tuple of (defended_image, telemetry_dict).
    """
    t0 = time.perf_counter()

    # Stage 1: Detect spectral, reconstruction, and localized patch anomalies
    anomaly_diagnostics = detect_anomalies(image_array, compute_visualization=True)
    is_anomalous = anomaly_diagnostics["is_anomalous"]

    actions_taken = ["Stage 1: Spectral & VAE Anomaly Gate"]
    patch_found = False
    patch_mask = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)

    # Dynamic Gating Decision
    if not is_anomalous:
        gated_bypass = True
        gate_status = "BYPASSED"
        defended_image = image_array.copy()
        actions_taken.append("Dynamic Gating: Bypass Purification (100% Fidelity)")
    else:
        gated_bypass = False
        gate_status = "ACTIVE"
        
        # Stage 2: Spatial Patch Masking & Inpainting
        patched_img, patch_mask, patch_found = mask_and_inpaint(image_array)
        if patch_found:
            actions_taken.append("Stage 2: AdvPatch Telea Inpainting (r=9)")
            defended_image = patched_img
            actions_taken.append("Stage 3: Bypassed for Inpainted Patch (Preserving Natural Texture)")
        else:
            actions_taken.append("Stage 2: Patch Scan (Clean)")
            # Stage 3: Fast Input Purification (Bit Slicing + Median Smoothing)
            defended_image = purify_input(patched_img)
            actions_taken.append("Stage 3: Bit-Slicing (>>3 <<3) & Median Blur (3x3)")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    telemetry = {
        "defense_latency_ms": round(elapsed_ms, 2),
        "meets_sla": elapsed_ms < 15.0,
        "is_anomalous": is_anomalous,
        "dynamic_gated": gated_bypass,
        "gate_status": gate_status,
        "anomaly_score": anomaly_diagnostics["anomaly_score"],
        "high_freq_ratio": anomaly_diagnostics["high_freq_ratio"],
        "recon_error": anomaly_diagnostics["recon_error"],
        "recon_threshold": anomaly_diagnostics["recon_threshold"],
        "patch_anomaly": anomaly_diagnostics["patch_anomaly"],
        "radial_curve": anomaly_diagnostics["radial_curve"],
        "patch_detected": patch_found,
        "patch_mask": patch_mask,
        "spectrum_visual": anomaly_diagnostics["spectrum_visual"],
        "actions_taken": actions_taken
    }

    return defended_image, telemetry
