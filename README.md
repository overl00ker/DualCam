<div align="center">

<img src="icon.png" alt="DualCam logo" width="120" />

# DualCam — Dual-Channel Imaging Polarimeter

**Real-time detection of polarized laser radiation against an unpolarized background: two cameras behind one polarizing beam-splitter cube, and their balanced difference D = CAM1 − k·CAM2.**

Built with C++17 · Qt 6 · OpenCV 4/5 — runs on Windows 11 desktops and Raspberry Pi 5.

<sub>Developed for the scientific and metrology work of the Institute of Physics, National Academy of Sciences of Ukraine (IF NASU).</sub>

</div>

---

## Table of contents

- [What it does](#what-it-does)
- [Physical principle](#physical-principle)
- [Signal path: every formula and what it means](#signal-path-every-formula-and-what-it-means)
- [Error budget](#error-budget)
- [Not implemented yet](#not-implemented-yet)
- [Screenshots & figures](#screenshots--figures)
- [The interface, panel by panel](#the-interface-panel-by-panel)
- [Saving data: snapshots and EXIF](#saving-data-snapshots-and-exif)
- [Implementation notes](#implementation-notes)
- [Dependencies](#dependencies)
- [Building from source](#building-from-source)
- [Running on a Raspberry Pi 5](#running-on-a-raspberry-pi-5)
- [Hotkeys & presets](#hotkeys--presets)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [License & credits](#license--credits)

---

## What it does

DualCam is the software half of a two-channel imaging polarimeter. One lens forms the image, a polarizing beam-splitter cube (PBS) splits it into two orthogonal linear polarizations, and two sensors record both images at the same moment. Unpolarized light — daylight, lamps, almost everything in a scene — is split equally between the sensors. Linearly polarized light, such as a laser beam, goes mostly into one of them. Once the two images are balanced and subtracted, the scene cancels and the polarized source remains.

The program captures both channels, suppresses noise, registers the images to a fraction of a pixel, equalizes their sharpness and gain, forms the difference and saves every frame with a complete record of how it was obtained. For every step, the sections below give the formula, the reason for its form and its physical meaning.

Symbols used throughout: $\tilde I_1, \tilde I_2$ — the two recorded images in DN (digital numbers, 0…255); $k$ — the channel balance; $D$ — the difference; $B$ — background brightness; $I_L$, $p$, $\psi$ — intensity, degree and azimuth of linear polarization of the source.

---

## Physical principle

### Stokes description of the light

The polarization state of partially polarized light is described by the Stokes vector, whose components are intensities measured behind ideal analyzers:

$$
S_0 = I_0 + I_{90}, \qquad S_1 = I_0 - I_{90}, \qquad S_2 = I_{45} - I_{135}, \qquad S_3 = I_{\circlearrowright} - I_{\circlearrowleft}.
$$

The scene is modelled as an unpolarized background $B(\mathbf r)$ plus a source of intensity $I_L(\mathbf r)$ with degree of linear polarization $p$ and azimuth $\psi$:

$$
S_0 = B + I_L, \qquad S_1 = p\, I_L \cos 2\psi .
$$

The background contributes only to $S_0$; the source contributes to both.

### The cube is a pair of crossed analyzers

The two ports of a PBS act as orthogonal linear analyzers with Mueller matrices

$$
\mathbf M_{\parallel,\perp} = \frac12
\begin{pmatrix}
1 & \pm 1 & 0 & 0\\
\pm 1 & 1 & 0 & 0\\
0 & 0 & 0 & 0\\
0 & 0 & 0 & 0
\end{pmatrix},
$$

so the intensities reaching the two sensors are $I_{1,2} = \tfrac12 (S_0 \pm S_1)$, and

$$
\Sigma = I_1 + I_2 = S_0, \qquad D = I_1 - I_2 = S_1 .
$$

**Meaning.** $\Sigma$ is an ordinary image of the scene (total flux). $D$ is a polarimetric image: the unpolarized background lands in both channels as $B/2$ and cancels exactly, while the source leaves $p\,I_L\cos 2\psi$. Each channel on its own is still a complete image of the scene; nothing is thrown away, unlike a single camera behind a polarizer.

### From light to digital numbers

The sensors do not record $I_{1,2}$ directly. Each channel has its own gain $g_i$ (port transmission or reflection, quantum efficiency, analogue gain, exposure time), point-spread function $h_i$ (focus), dark level $d_i$ (black offset plus dark current) and noise $n_i$, and the second image is geometrically displaced by a transform $W$:

$$
\tilde I_1 = g_1\,(h_1 \ast I_1) + d_1 + n_1, \qquad
\tilde I_2 = \big[g_2\,(h_2 \ast I_2)\big]\circ W + d_2 + n_2 .
$$

DualCam computes

$$
D = \tilde I_1 - k\,\tilde I_2 .
$$

With $k = g_1/g_2$, matched PSFs and registered geometry this becomes

$$
D = g_1\,p\,I_L\cos 2\psi \;+\; (d_1 - k\,d_2) \;+\; (n_1 - k\,n_2):
$$

the polarized signal, a constant dark offset and noise. Every processing step below exists to remove or limit one of the other terms: $W$ (registration), $h_1 \ne h_2$ (sharpness matching), $g_1 \ne g_2$ (channel balance $k$), $n_i$ (accumulation, threshold).

### What a two-channel instrument cannot see

- **Only one Stokes component.** The response is proportional to $\cos 2\psi$ and vanishes for a source polarized at $\psi = 45^\circ$ to the cube axes, however strongly polarized it is: then both sensors get equal halves, exactly like unpolarized light. For a random azimuth the response is below half its maximum in one case out of three, and below one tenth in about 6 % of cases. Remedy: rotate the instrument about its optical axis, or add a third channel at 45° or a phase plate.
- **Polarized background does not cancel.** Skylight is polarized by Rayleigh scattering (strongest at 90° from the sun, typically up to 70–80 % in a clear sky), and specular reflections from water, glass or wet asphalt near Brewster's angle are strongly polarized. They stay in $D$. They are usually extended and weak, whereas a laser spot is compact and bright, so the spatial structure tells them apart.
- **Finite extinction of the cube does not spoil cancellation.** The unpolarized part still enters both channels with fixed weights, and $k$ absorbs those weights. Extinction only scales the useful signal by a polarimetric efficiency $\eta$ slightly below 1.
- **Field of view.** A PBS extinguishes worse for oblique rays, and a wide-angle lens adds peripheral aberrations; the central part of the field is the most reliable.

---

## Signal path: every formula and what it means

Order of processing, as implemented: capture → grey level → mirror flip of CAM2 → motion detection → temporal accumulation → bilateral filter → sharpness metric — all on a worker thread — then, on the main thread: registration → sharpness matching → difference with $k$ → clipping → noise floor → display.

### 1. Capture and grey level

$$
Y = 0.299\,R + 0.587\,G + 0.114\,B
$$

**What it is.** In the grey modes a colour frame is converted to luma by the standard weights (OpenCV `BGR2GRAY`); a monochrome sensor (*native grey*) is used as is. All values are in DN, 0…255.

**Why this form.** At the laser lines the instrument is built for (808 and 1064 nm) the dyes of a Bayer mosaic are almost transparent, so the three pixel groups respond nearly equally. The weighted sum is then close to the raw sensor signal and demosaicing artefacts are suppressed.

**Physical meaning and limits.** Processing is 8-bit. Uniform quantization with step $q$ adds variance $q^2/12$ per sample, so a difference of two independently quantized samples carries $q^2(1+k^2)/12 \approx q^2/6$. The IMX296 sensor digitizes 10 bits; going to 8 bits makes the step four times larger, which is 12 dB more quantization noise. Saturation is set by the background: it fills the pixel's full well whether or not it is later subtracted, so the exposure must be chosen by the background brightness.

### 2. Mirror flip of CAM2

$$
x' = W_{\mathrm{img}} - 1 - x \quad\text{and/or}\quad y' = H_{\mathrm{img}} - 1 - y
$$

**Why.** The reflected port of the cube sees a mirror image (a single reflection reverses handedness). A homography could represent a mirror, but feature descriptors used for registration are not mirror-invariant, so the handedness is fixed first. EXIF: `flipHorizontal2`, `flipVertical2`.

### 3. Motion detection

$$
m = \frac{N_{\mathrm{moving}}}{N}, \qquad \text{motion} \iff m > m_{\mathrm{thr}}
$$

**What it is.** A Gaussian-mixture background model (OpenCV MOG2) keeps, for every pixel, a few Gaussians describing its recent values. A pixel counts as moving if it lies more than $4\sigma$ (squared Mahalanobis distance > 16) from every background component. The model learns at a rate of 0.01 per frame, so it remembers about 100 frames. It runs on CAM2, downscaled to at most 320 px.

**Why.** Temporal accumulation (step 4) assumes a static scene. When the rig or the source moves, the accumulator smears the image into a trail with a time constant of about $N$ frames. When motion is flagged the accumulator is reset and restarts from the current frame.

**Setting.** Motion threshold 1–50 % of the pixels, default 5 % (EXIF `motionThreshold`).

### 4. Temporal accumulation (T-buffer)

$$
y_n = \alpha\,x_n + (1-\alpha)\,y_{n-1}, \qquad \alpha = \frac1N, \qquad H(z) = \frac{\alpha}{1-(1-\alpha)\,z^{-1}} .
$$

**Why this form.** A recursive (exponential) average needs one stored frame and constant work for any depth $N$; a moving average of $N$ frames needs $N$ stored frames.

**Physical meaning.** $H(1) = 1$: a stationary signal passes unattenuated. For white noise of variance $\sigma^2$ the steady-state variance is

$$
\sigma_y^2 = \sigma^2\,\frac{\alpha}{2-\alpha} = \frac{\sigma^2}{2N-1},
$$

so the signal-to-noise ratio improves by $\sqrt{2N-1}$: a factor of 3.9 for the default $N = 8$. The price is lag: after $n$ frames a step change has reached $1-(1-1/N)^n$ of its final value, i.e. the time constant is about $N$ frames (≈ 0.27 s for $N = 8$ at 30 fps).

**What it does not remove.** Only the random terms (shot and read noise) are averaged away. Gain mismatch, misregistration, defocus and pixel-sensitivity non-uniformity repeat identically in every frame and pass through unchanged. Increasing $N$ helps only until the noise falls below those deterministic terms (see [Error budget](#error-budget)). The accumulator is kept in floating point; the output is rounded to 8 bits each frame.

**Setting.** `timeBuffer` 1–30, default 8 (1 = off).

### 5. Bilateral filter (optional)

$$
I'(\mathbf x) = \frac{1}{W(\mathbf x)} \sum_{\mathbf y \in \Omega}
\exp\!\left(-\frac{\lVert\mathbf x-\mathbf y\rVert^2}{2\sigma_s^2}\right)
\exp\!\left(-\frac{\big(I(\mathbf x)-I(\mathbf y)\big)^2}{2\sigma_r^2}\right) I(\mathbf y)
$$

**What it is.** A smoothing whose weights fall off both with distance and with brightness difference. With strength $s$ (1–20) the window diameter is $s$ (rounded up to odd) and $\sigma_s = \sigma_r = 10\,s$. Because $\sigma_s$ is much larger than the window, spatially it is almost a flat $s \times s$ window, and the range term decides: neighbours that differ by much less than $\sigma_r$ are averaged, much larger differences are left out.

**Physical meaning.** $\sigma_r$ is a contrast threshold in DN. At the default $s = 5$ ($\sigma_r = 50$ DN) any structure with a contrast below about 50 DN — including a weak laser spot — is smoothed like a 5 × 5 mean, while strong edges are preserved. The filter is non-linear and acts on each channel separately before subtraction, so wherever the channels differ (the laser is in one of them) it acts differently and leaves its own residual in $D$. For quantitative measurements keep it off or weak. It also runs before the sharpness metric and changes $F$.

**Setting.** `bilateralFilter`, `bilateralStrength`, default 5.

### 6. Sharpness metric

$$
F_i = \operatorname{Var}\!\big[\nabla^2 \tilde I_i\big], \qquad
\nabla^2 \approx \begin{pmatrix} 0 & 1 & 0\\ 1 & -4 & 1\\ 0 & 1 & 0 \end{pmatrix}
$$

computed over the whole frame of each camera.

**Why this form.** In Fourier space the Laplacian multiplies by $-\lvert\boldsymbol\omega\rvert^2$, so

$$
F = \sum_{\boldsymbol\omega \ne 0} \lvert\boldsymbol\omega\rvert^4\, \big\lvert \tilde I(\boldsymbol\omega)\big\rvert^2
$$

is the energy of the finest detail. A Gaussian PSF of width $\sigma$ multiplies the power spectrum by $e^{-\sigma^2\lvert\boldsymbol\omega\rvert^2}$, so for a fixed scene $F$ falls monotonically as a channel defocuses: equal $F$ means equal blur.

**Properties that matter.**

- The dark level does not enter: the Laplacian of a constant is zero.
- $F$ scales with the square of the gain, $F(g\,I) = g^2 F(I)$. To compare sharpness rather than brightness, DualCam compares $F_1$ with $k^2 F_2$.
- Noise sets a floor. White noise $\sigma_n$ alone gives $\operatorname{Var}[\nabla^2 n] = 20\,\sigma_n^2$ (20 is the sum of the squared kernel weights), so in a dark or noisy frame $F$ measures noise, not focus. Accumulation lowers this floor.

**Use.** The Focus panel shows $F_1$ and $F_2$ live. Move the sensor until $F_1 \approx k^2 F_2$: that is mechanical sharpness matching, which is always better than the computational one in step 8. EXIF: `focus.cam1`, `focus.cam2`.

### 7. Geometric registration

$$
\mathbf x_1 \sim \mathbf H\,\mathbf x_2, \qquad \mathbf H \in \mathbb R^{3\times 3}\ \text{(8 degrees of freedom)}
$$

**Why a homography.** Both sensors look through the same lens: every image point is formed by the same bundle of rays leaving the exit pupil, and the two sensors differ only by their position and tilt in image space (plus the mirror of the cube). Two planes cutting the same bundle of rays from one centre are related by a projective transformation, whatever the depth of the scene. A homography is therefore the exact model; what it cannot express (slightly different distortion of the two optical paths) is what remains.

**Why sub-pixel accuracy is required.** A registration error $\delta\mathbf r$ leaves a residual proportional to the brightness gradient:

$$
\delta_{\mathrm{geo}} \approx \nabla B \cdot \delta\mathbf r .
$$

For an edge of contrast $\Delta B$ spread over two pixels, $\lvert\nabla B\rvert \approx \Delta B/2$, so an error of 0.1–0.5 px leaves 5–25 % of the edge contrast in $D$ — comparable to, or larger than, a weak laser. The method does not work without sub-pixel registration.

**Automatic registration** (Calibrate alignment, done once for a mechanical configuration):

1. *Registration signal.* $s = \big\lvert \nabla \big(G_{\sigma=1} \ast \mathrm{CLAHE}(\tilde I)\big)\big\rvert$, normalized to [0, 1]. Gradients, not intensities: the two channels are *supposed* to differ in brightness — that difference is the gain mismatch and the polarimetric signal itself — but the positions of edges are common to both. A brightness-matching optimizer would distort the geometry to explain the photometric difference away. Local histogram equalization (CLAHE, clip 2, 8 × 8 tiles) lets dim and bright regions contribute equally.
2. *Global search.* SIFT features (fallbacks ORB and, with OpenCV 4, AKAZE), Lowe's ratio test 0.75 with a mutual cross-check, and $\mathbf H$ by RANSAC with a 3 px reprojection tolerance. This stage works from any initial offset.
3. *Sub-pixel refinement.* Each inlier is re-located by normalized cross-correlation of a 21 × 21 patch within ±4 px. The correlation peak is refined by a parabola through the three samples around the maximum, $\Delta x = \dfrac{c_{-1} - c_{+1}}{2\,(c_{-1} - 2c_0 + c_{+1})}$, and $\mathbf H$ is re-fitted (RANSAC, 2 px). A keypoint detector alone localizes to about half a pixel, which in $D$ shows up as false edges.
4. *ECC polish* (optional). Maximizes the enhanced correlation coefficient $\rho = \dfrac{\langle \hat s_1,\ \hat s_2\circ W\rangle}{\lVert\hat s_1\rVert\,\lVert\hat s_2\circ W\rVert}$ (zero-mean signals) over all pixels on a three-level pyramid, starting from step 3. Every pixel votes, so the residual drops to a fraction of a pixel. The result is accepted only if the homography stays geometrically sane and the image corners move by less than 25 px.

Each stage is rejected if it fails its check, so the solution degrades to plain SIFT + RANSAC rather than to garbage. A homography is valid only at the resolution at which it was solved; it is discarded when the frame size changes. EXIF: `alignment.warp`, `alignment.warpModel`, `alignment.warpSize`.

**Manual 6-DOF adjustment.**

$$
\mathbf H_{\mathrm{man}} = \mathbf T(t_x, t_y)\;\mathbf S_c(s)\;\mathbf K\,\mathbf R\,\mathbf K^{-1}, \qquad
\mathbf K = \begin{pmatrix} f & 0 & c_x\\ 0 & f & c_y\\ 0 & 0 & 1 \end{pmatrix}, \quad
f = \max(W_{\mathrm{img}}, H_{\mathrm{img}}), \quad \mathbf R = \mathbf R_z \mathbf R_y \mathbf R_x .
$$

$\mathbf K\mathbf R\mathbf K^{-1}$ is the image of a camera rotated about its projection centre: the angles $r_x, r_y$ describe sensor tilt (keystone), $r_z$ rotation about the optical axis. $\mathbf S_c$ scales about the image centre — a difference in magnification from a different sensor distance. $\mathbf T$ is a lateral sensor shift in pixels. $f = \max(W, H)$ corresponds to a field of view of about 53° across the long side; it only sets how strongly a tilt angle turns into keystone. EXIF: `alignment.manualCam1`, `alignment.manualCam2`.

**Overlap mask.** After warping, a pixel is kept only if its whole bilinear footprint came from inside the source frame: an all-255 mask is warped with bilinear interpolation and only exact 255 survives. Interpolating against the black border outside the frame would create a false edge in $D$. Outside the overlap $D = 0$, and the share of such pixels is reported (`diff.noOverlap`).

### 8. Sharpness (PSF) matching

**The residual it removes.** Approximate the two PSFs by Gaussians of widths $\sigma_1, \sigma_2$. A Gaussian satisfies the heat equation, $\partial G/\partial(\sigma^2/2) = \nabla^2 G$, so the difference of two close Gaussians is a scaled Laplacian of Gaussian:

$$
\delta_{\mathrm{df}} \approx \tfrac12\,\Delta\sigma^2\,\nabla^2 B, \qquad \Delta\sigma^2 = \sigma_1^2 - \sigma_2^2 .
$$

A defocus difference therefore draws the outline of every bright object in $D$. This residual carries no polarization information. It vanishes only for $\sigma_1 = \sigma_2$: what matters is equal sharpness of the two channels, not the absolute focus.

**Assembly tolerance.** A longitudinal sensor offset $\delta z$ at f-number $N_f$ enlarges the blur circle radius by about $\delta z/(2N_f)$. With 3.45 µm pixels and $N_f \ge 2.8$, keeping the difference below one pixel requires $\delta z \lesssim 20$ µm — hard to hold with two independently mounted sensors, so some sharpness mismatch is present in any real assembly.

**The correction.** Gaussian blurs add in variance, so blurring the sharper channel with

$$
\sigma_c^2 = \lvert \sigma_1^2 - \sigma_2^2 \rvert
$$

equalizes the PSFs. $\sigma_i$ are unknown, but the ratio of the sharpness metrics measures them. With $r = \max(F_1, k^2F_2)/\min(F_1, k^2F_2)$ and the spectral form of $F$ (step 6), to first order in $\Delta\sigma^2$

$$
r \approx 1 + \Delta\sigma^2\,\langle \lvert\boldsymbol\omega\rvert^2 \rangle_F
\quad\Longrightarrow\quad
\sigma_c \approx \frac{\sqrt{r-1}}{\omega_F},
$$

where $\omega_F^2 = \langle\lvert\boldsymbol\omega\rvert^2\rangle_F$ is the mean square spatial frequency weighted by the scene's Laplacian energy spectrum. DualCam uses

$$
\sigma_c = \min\!\big(6\ \mathrm{px},\ 0.6\sqrt{r-1}\big), \qquad \text{applied only if } r > 1.15 .
$$

The constant 0.6 px corresponds to $\omega_F \approx 1.7$ rad/px; it is empirical and reflects that the Laplacian energy of typical scenes lies close to the Nyquist frequency (π rad/px). The 1.15 threshold is a dead band: $F$ fluctuates with noise from frame to frame, and channels that are already matched should not be smoothed. The 6 px cap protects against a runaway blur when one channel is dark or noise-dominated. The Gaussian kernel is about $3\sigma_c$ wide (odd, 3–31 px), i.e. truncated at about $\pm1.5\sigma_c$, so the effective blur is somewhat narrower than $\sigma_c$.

**The price.** Blurring also spreads a compact source in the sharper channel: its peak in $D$ drops, and with it the signal-to-noise ratio for point sources. Computational matching only compensates a residual mismatch; it does not replace mechanical focusing. The smaller $\lvert\Delta\sigma^2\rvert$ is optically, the less blur is needed.

The Diff readout shows the ratio and the action, e.g. `sharpness F1/(k²F2) 1.32 → CAM1 blurred, σ 0.34 px`. EXIF: `diff.sharpnessMatch` (`blurredCamera`, `sigmaPx`).

### 9. The difference

$$
D(\mathbf x) = \operatorname{round}\!\big(\tilde I_1(\mathbf x) - k\,\tilde I_2(\mathbf x)\big) \quad [\mathrm{DN}]
$$

computed on the grey, registered, sharpness-matched channels and kept as a signed 16-bit value.

**Physical meaning.** With a correct $k$ the background cancels and

$$
D = g_1\,\eta\,p\,I_L\cos 2\psi + (d_1 - k\,d_2) + (n_1 - k\,n_2).
$$

The **sign** tells the orientation: $D > 0$ — the component analysed by CAM1's port dominates, $D < 0$ — the orthogonal one analysed by CAM2's port. The display shows $\lvert D\rvert$; the sign is currently not saved (see [Not implemented yet](#not-implemented-yet)). The dark levels leave a constant offset $d_1 - k\,d_2$ of a few DN, which is not subtracted yet and normally lies below the noise floor.

**Noise of the difference.** $\sigma_D^2 = \sigma_1^2 + k^2\sigma_2^2$. Background photons are Poisson-distributed; each channel collects about $B/2$, so the shot noise of $D$ is $\sigma_{\mathrm{sh}} \approx \sqrt B$ (in electrons). Subtraction removes the mean of the background, not its fluctuations. Read noise adds $\sigma_{\mathrm{rd}}\sqrt{1+k^2}$ and matters only in low light or with very short exposures.

**Why the difference wins.** In one channel a faint spot must stand out against the structure of the scene, which is of the order of $B$ itself. In $D$ the structure is gone and only the background noise $\sqrt B$ remains. The contrast gain is

$$
G \sim \frac{B}{\sqrt B} = \sqrt B,
$$

about two orders of magnitude for a typical well filling of $B \sim 10^4$ electrons.

### 10. Channel balance k

**What k is.** For unpolarized light $I_1 = I_2 = S_0/2$, so

$$
\frac{\tilde I_1}{\tilde I_2} = \frac{g_1 S_0/2 + d_1}{g_2 S_0/2 + d_2} \;\longrightarrow\; \frac{g_1}{g_2} = k \qquad (d_i \ll \text{signal}).
$$

One number collects every radiometric difference between the channels: port transmission and reflection $\tau_i$, quantum efficiency $\eta_i$, analogue gain $G_i$, exposure time $t_i$, vignetting:

$$
k = \frac{\tau_1\,\eta_1\,G_1\,t_1}{\tau_2\,\eta_2\,G_2\,t_2} .
$$

Changing the gain or the shutter of either camera changes $k$ in proportion, so $k$ must be calibrated with the exposure that will be used and the exposure locked afterwards. With automatic exposure each camera adapts on its own and $k$ becomes a random variable. DualCam warns when the exposure differs from the one at calibration.

**How it is measured** (Calibrate k). For each frame pair $j$:

$$
k_j = \operatorname*{median}_{\mathbf x \in P_j} \frac{\tilde I_1(\mathbf x)}{\tilde I_2(\mathbf x)}, \qquad
P_j = \lbrace \mathbf x:\ \text{every 2nd row and column, inside the overlap},\ 24 \le \tilde I_{1,2}(\mathbf x) \le \mathrm{clip}-1 \rbrace,\ \ \lvert P_j\rvert \ge 1000,
$$

then, over 20 consecutive frames,

$$
k = \operatorname*{median}_{j} k_j, \qquad \sigma_k = \operatorname{std}_j k_j, \qquad s = \frac{Q_{75} - Q_{25}}{2k},
$$

where $Q_{25}, Q_{75}$ are the quartiles of the per-pixel ratios (the median of the per-frame values is reported).

**Why each choice.**

- *A ratio*, because the model is multiplicative: $k$ is a gain ratio.
- *The median*, because its breakdown point is 50 %: edges with residual misregistration, polarized specular highlights, sky, hot pixels or a laser left on can occupy up to half the pixels without moving $k$. A least-squares slope $\sum \tilde I_1\tilde I_2 / \sum \tilde I_2^2$ would be dominated by the brightest pixels and pulled by every outlier.
- *At least 24 DN in both channels*, because a ratio of small numbers is noisy (its relative error grows as $\sigma/\tilde I$) and biased by the dark levels $d_i$. *Below the clip level*, because a clipped pixel's ratio is wrong.
- *At least 1000 pixels*: the standard error of a median is $\approx 1.25\,s_r/\sqrt n$ ($s_r$ — per-pixel spread), so 1000 pixels give about 4 % of the per-pixel spread. A 640 × 480 frame gives about 77 000 sample pixels, 1456 × 1088 about 400 000.
- *20 frames.* One frame pair is enough to obtain a value. Mains-powered lamps flicker at 100 Hz and the two cameras are not synchronized, so in a single pair the two exposures can integrate different parts of the flicker cycle and the whole frame is biased. The median over 20 frames (≈ 0.7 s at 30 fps) removes such pairs, and $\sigma_k$ is the uncertainty to quote as $k \pm \sigma_k$. In sunlight there is no flicker and $\sigma_k$ is small.
- *Pixel spread $s$* is how much the ratio varies across the field (pixel-response non-uniformity, different vignetting, residual polarization in the scene). On a uniform unpolarized scene it is the remaining pixel-by-pixel residual, about $s\,\tilde I$. Above 5 % DualCam warns that the target is probably uneven or polarized.

**How accurate k must be.** An error $\delta k$ leaves a residual $\delta k\,\tilde I_2$, proportional to the background — larger in daylight. It must stay below the useful signal:

$$
\lvert\delta k\rvert \lesssim \frac{\eta\,p\,I_L}{B} .
$$

For example, a background of 150 DN with $\sigma_k = 0.001$ leaves 0.15 DN, far below a 15 DN noise floor.

**Calibration target.** A matte, diffuse white surface filling the frame (paper, a ceramic tile): diffuse scattering depolarizes. Laser off. Avoid sky, water, glass and glossy paint. Exposure as for the measurement, background at roughly 50–80 % of full scale.

**Storage.** $k$ is written to the settings file the moment it is measured, restored on the next start and not changed by presets; the mouse wheel changes it only when the field has focus. The line under the field shows the record, and every snapshot carries it in EXIF as `channelBalance`: `k`, `source` (calibrated / manual), `at`, `frames`, `perFrame` (all 20 values), `sigma`, `pixelSpread`, `pixelsPerFrame`, `exposureAtCalibration`, `exposureNow`, `frameSize`, `method`.

### 11. Saturation (clip level)

A clipped pixel reads the maximum of the ADC however much light arrived: the true value is at least the reading.

- **Clipped in both channels:** $D = 0$ by definition. There is no information about the difference, and two saturated images must subtract to zero. The share is reported (`diff.clippedBoth`).
- **Clipped in one channel:** $D$ is computed and shown, but biased. If $\tilde I_1$ is clipped the true $D$ is larger than shown; if $\tilde I_2$ is clipped the true $D$ is smaller (more negative). Either way $\lvert D\rvert$ is a lower bound when the clipped channel is the one carrying the signal. The share is reported (`diff.clippedOne`). If the amplitude of the spot matters, reduce the exposure until its bright channel no longer clips.

**Setting.** Clip level 128–255 DN, default 254 (EXIF `clipLevel`).

### 12. Noise floor (detection threshold)

$$
D(\mathbf x) \leftarrow 0 \quad \text{where} \quad \lvert D(\mathbf x)\rvert \le T
$$

**Physical meaning.** This is the detection decision: a pixel belongs to a source when the difference exceeds $T$. For Gaussian noise of standard deviation $\sigma_D$ and $T = \kappa\,\sigma_D$ the false-alarm probability per pixel is $\operatorname{erfc}(\kappa/\sqrt2)$:

| κ | false alarms per pixel | in a 1.6 Mpx frame |
|---|---|---|
| 3 | 2.7 × 10⁻³ | ≈ 4 000 px |
| 4 | 6.3 × 10⁻⁵ | ≈ 100 px |
| 5 | 5.7 × 10⁻⁷ | ≈ 1 px |

Choose $T \approx 4\text{–}5\,\sigma_D$, with $\sigma_D$ measured on an unpolarized scene (after accumulation $\sigma_D$ is already $\sqrt{2N-1}$ times smaller).

**Limitation.** $T$ is one value for the whole frame, while shot noise grows as the square root of brightness: a fixed threshold gives false alarms in bright areas and loses sensitivity in dark ones. The physically consistent threshold would follow the sum channel, $T(\mathbf r) = \kappa\sqrt{\Sigma(\mathbf r)}$ (not implemented yet).

**Setting.** 0–50 DN, default 15 (EXIF `noiseFloor`, `diff.noiseFloorDN`).

### 13. Display: stretch and colour scale

$$
c = \min\!\big(255,\ \operatorname{round}(\lvert D\rvert \cdot G)\big), \qquad
G = \begin{cases} 255/\max\lvert D\rvert & \text{stretch on}\\ 1 & \text{stretch off}\end{cases}, \qquad
\text{colour} = \mathrm{JET}(c).
$$

**Meaning.** Purely for viewing: linear, and $D = 0$ stays 0. The top of the colour scale is $255/G$ DN; both the factor and the scale are shown on the image. Stretch changes the picture, not the physics, and is not comparable between sessions. EXIF: `diff.displayGain`, `diff.fullScaleDN`, `diff.maxAbsDiffDN`; the colour index of a saved diff image converts back to DN as $\lvert D\rvert \approx c/G$.

### 14. Peak tracking

$$
\mathbf p = \operatorname*{arg\,max}_{\mathbf x}\ \big(\mathrm{box}_{31\times31} \ast \tilde I\big)(\mathbf x), \qquad \text{Peak} = \max_{\mathbf x} \tilde I(\mathbf x)
$$

The 31 × 31 mean locates the centre of the brightest compact spot and ignores single hot pixels. The reported value is the brightest raw pixel of each channel; 255 means that channel is saturated.

### 15. Fusion (sum channel)

$$
F_{\mathrm{fus}} = \tfrac12\big(\tilde I_1 + k\,\tilde I_2 \circ \mathbf H\big)
$$

**Why k is needed here too.** With equal weights the two gains do not cancel and the polarization leaks into the "total flux" image:

$$
\tilde I_1 + \tilde I_2 = \frac{g_1+g_2}{2}\,S_0 + \frac{g_1-g_2}{2}\,S_1 ,
$$

so a polarized source would look brighter or dimmer in the sum depending on its azimuth. With the same $k = g_1/g_2$ as in the difference the $S_1$ term vanishes exactly:

$$
\tilde I_1 + k\,\tilde I_2 = g_1\,S_0 + (d_1 + k\,d_2).
$$

**Meaning.** $F_{\mathrm{fus}}$ is $\Sigma/2$, the ordinary image of the total flux $S_0$ with the polarization removed, expressed in CAM1 units: for an unpolarized scene it equals CAM1. The sum and the difference use one $k$, so $\Sigma$ and $D$ are a consistent pair ($S_0$ and $S_1$). Values above 255 DN (e.g. a source polarized into CAM2 with $k > 1$) clip. The view is labelled with the formula and $k$; EXIF: `fusion`, `fusionFormula`.

### 16. Profile and surface tools

The profile samples the image every ≈ 1 px along the drawn line with bilinear interpolation (luma for colour images); the 3-D surface averages the selected region down to at most 96 px on its long side and plots the grey level as height.

> **Caveat.** A diff snapshot stores the colour map, not $\lvert D\rvert$. The luma of the JET colours is not monotonic in $\lvert D\rvert$ — dark red, the largest $\lvert D\rvert$, is darker than cyan or yellow — so a profile or surface of a diff snapshot does not show $\lvert D\rvert$. Use these tools on camera snapshots.

### 17. Exposure

The recorded signal is proportional to analogue gain × exposure time × scene radiance. Manual gain and shutter (shared or per camera) are applied to libcamera (CSI) cameras; USB cameras keep their own automatic exposure. Exposure must be locked for $k$ to remain valid (step 10). EXIF: `exposure`.

---

## Error budget

Collecting the terms above, the residual of an unpolarized area in $D$ has variance

$$
\sigma_{\mathrm{res}}^2 = (\delta k\,B)^2 + (\nabla B\cdot\delta\mathbf r)^2 + \tfrac14\big(\Delta\sigma^2\,\nabla^2 B\big)^2 + (0.7\,\gamma B)^2 + \frac{B}{2N-1},
$$

and the signal-to-noise ratio of a source is $\mathrm{SNR}_D = \eta\,p\,I_L\cos 2\psi \,/\, \sigma_{\mathrm{res}}$.

| Term | Origin | Where it shows in D | Handled by | What remains |
|---|---|---|---|---|
| $\delta k\,B$ | gain mismatch | everywhere, ∝ brightness | Calibrate k (step 10) | $\sigma_k B$ |
| $\nabla B\cdot\delta\mathbf r$ | misregistration | at edges | homography, sub-pixel, ECC (step 7) | ≈ 0.1 px × gradient |
| $\tfrac12\Delta\sigma^2\nabla^2 B$ | different focus | outlines of bright objects | Focus panel, PSF matching (steps 6, 8) | lower resolution for point sources |
| $0.7\,\gamma B$ | pixel response non-uniformity, $\gamma \approx 0.3$–1 % | fixed pattern ∝ brightness | not corrected yet | all of it |
| $B/(2N-1)$ | shot noise | random | accumulation (step 4), noise floor (step 12) | decreases with $N$ |
| $\sigma_{\mathrm{rd}}\sqrt{1+k^2}$ | read noise | random | accumulation | matters in low light only |
| $d_1 - k\,d_2$ | dark level | constant offset | noise floor | a few DN |
| $q^2(1+k^2)/12$ | 8-bit quantization | random | — | fixed by bit depth |

The pixel-response term comes from independent sensitivity patterns $1+\gamma_j$ of the two sensors: $\gamma\sqrt{1+k^2}\,B/2 \approx 0.7\,\gamma B$. It is of the same order as shot noise ($B = 10^4$ electrons: shot noise 100 e⁻, non-uniformity at $\gamma = 1$ % ≈ 70 e⁻), but it does not average away: after $N = 8$ frames shot noise falls to 26 e⁻ and the fixed pattern dominates by almost a factor of three. Beyond that point a deeper accumulation brings nothing without a flat-field correction. The table also dictates the order of work: lock the exposure, register, match focus mechanically, calibrate $k$ — then accumulate.

---

## Not implemented yet

Parts of the physical model that the program does not apply yet:

- **Dark-frame subtraction** of $d_i$. The IMX296 has optically masked rows that could track the dark level frame by frame.
- **Flat-field correction** of the pixel-response non-uniformity $\gamma_j$ — needed to reach the shot-noise limit.
- **Normalized degree of polarization** $\Pi = \dfrac{\tilde I_1 - k\tilde I_2}{\tilde I_1 + k\tilde I_2} = \eta\,\dfrac{S_1}{S_0}$, independent of illumination, vignetting and exposure, so comparable between sessions. It needs dark-level subtraction first; otherwise $d_1 + d_2$ in the denominator biases it low in dark areas.
- **Spatially varying threshold** $T(\mathbf r) = \kappa\sqrt{\Sigma(\mathbf r)}$ with a constant false-alarm rate.
- **Saving the signed difference** $D$ as data (e.g. 16-bit), not only the colour map of $\lvert D\rvert$.
- **10-bit processing** (12 dB less quantization noise).

---

## Screenshots & figures

> 📸 **This is where your own images go.** The application is highly visual, so a few good screenshots make the README far more convincing. Drop the files into [`docs/img/`](docs/img/) using the names below and they will appear automatically. Suggested captures, roughly in order of usefulness:
>
> | File to add | What it should show | Why it helps |
> |---|---|---|
> | `docs/img/hero.png` | The full main window with the side navigation visible and cameras off (clean placeholder views) | The single most important image — a calm, idle UI reads more clearly than a busy live scene; put it right under the title |
> | `docs/img/dual-mode.png` | Both cameras running side by side (the standard dual-view layout) | Shows the core two-camera workflow |
> | `docs/img/diff-mode.png` | Diff mode active, with the Jet heat map in the result view | Demonstrates the headline feature |
> | `docs/img/visualization-settings.png` | The visualization-settings dialog (noise floor, intensity stretch, colormap controls) | Documents how the difference view is tuned |
>
> If you later add **measurement results** (focus-vs-time curves, before/after denoise comparisons, a difference map next to the raw frames), those are exactly the kind of figures that belong in a "Results" or "Example session" section — tell me and I'll add one.

<!--
Once you add the files, uncomment this block:

<div align="center">
  <img src="docs/img/hero.png" alt="DualCam main window" width="80%" />
</div>
-->

---

## The interface, panel by panel

**Main viewports.** Two camera views (Cam 1 / Cam 2) show the raw or processed streams; a Result view shows the composed output — overlay, alignment, or difference map.

**Side navigation.** A collapsible rail switches between the functional sections: Capture, Pipeline, Diff, Focus, Snapshot, Presets and Gallery.

**Capture.** Source selection (camera index or GStreamer pipeline), independent Start/Stop, the flip checkboxes, and exposure controls.

**Pipeline.** The T-buffer (step 4) and bilateral strength (step 5), the motion threshold (step 3), and the alignment controls (step 7) — the automatic calibration and the manual 6-DOF dialog.

**Diff.** Channel balance `k` with **Calibrate k** and its record (step 10), clip level (step 11), noise floor (step 12), stretch (step 13) and peak tracking (step 14), plus a readout of every transform applied to the intensities: k, noise floor, stretch factor and resulting scale, clipped and no-overlap fractions, sharpness matching.

**Focus.** The large per-camera sharpness values $F_1$, $F_2$ (step 6) and their chart over time, with an adjustable history window.

**Snapshot.** Naming, the choice of which parameters to fold into the filename, and a recent-captures preview.

**Presets.** Save, load and delete named configuration profiles. The channel balance `k` is not part of a preset: it belongs to the rig.

**Floating controls.** Quick-access buttons (stream toggle, snapshot, mode toggle, focus toggle) plus status pills showing live FPS and alignment state. **HK's** opens the Actions & Keybinds menu, **?** the help.

---

## Saving data: snapshots and EXIF

A snapshot is never just a picture — it's a picture plus a record of how it was made.

**Image formats.** You can save the two cameras *combined* into one image, *separately* as two files (`_cam1` / `_cam2`, the **Separate files** option in the Snapshot panel), or save the *difference* map. File names carry a one-second timestamp; a second shot within the same second gets a `_2`, `_3`, … suffix rather than overwriting the first.

**Embedded parameter record.** Every snapshot carries, inside its EXIF `ImageDescription` tag, a JSON record of the full state: capture mode and timestamp, an optional user-supplied name, color mode, flip flags, time-buffer size, motion threshold, fusion state, bilateral filter settings, noise floor, intensity-stretch and peak-tracking flags, the channel balance `k` with its calibration record (`channelBalance`, see step 10) and the clip level (for diff snapshots also the display factor, the DN value at the top of the colour scale, the maximum |D|, the clipped / no-overlap fractions and the sharpness matching), the solved warp matrix and the manual 6-DOF adjustments, per-camera exposure (gain and shutter, and whether they were actually applied), the live sharpness values, the motion-active flag, diff-mode state and the frame count. In short: enough to reproduce the measurement. The record lives in the image file itself, so it cannot be separated from it; the preview window's METADATA panel reads it back, and so do standard tools (`exiftool`, file properties). Non-ASCII text (a Cyrillic snapshot name, say) is stored as JSON `\uXXXX` escapes, since the EXIF text field is ASCII.

**Embedded EXIF.** The EXIF block is written by a small hand-rolled writer (`exif_writer.cpp`): image description (the record above), make/model, software tag, and exposure time and ISO when manual exposure was applied through libcamera.

---

## Implementation notes

- **Threads.** `CameraWorker`, a dedicated `QThread`, grabs both cameras and runs steps 1–6. The main thread does registration, the difference and display. At most one frame pair is in flight: if the interface is still busy, the newest pair is dropped rather than queued, so latency never builds up.
- **Capture on the Pi.** The ISP delivers BGR directly (`libcamerasrc ! video/x-raw,format=BGR ! appsink`); a software `videoconvert` cost a full core per camera and capped 1456 × 1088 at about 20 fps. Motion detection runs on a ≤ 320 px proxy (full-resolution MOG2 took ~40 ms per frame).
- **Registration runs on its own thread**, so a slow solve never stalls capture. The solved matrix is stored in the settings file and restored on the next launch (the **Remember** checkbox).
- **Rendering.** Each frame is downscaled once to the view size and drawn 1:1 by a raster `ImageView`; there is deliberately no `QOpenGLWidget`, which forced GL composition of the whole window on every frame.
- **Resource monitor.** A CPU/temperature pill in the top bar (amber at 70 °C, high load or a frequency cap; red at 80 °C, throttling or under-voltage). Click it for per-core load, app / GUI / camera-thread CPU, ARM clock, fan speed, Raspberry Pi throttle flags, RAM and camera / shown / dropped FPS.

---

## Dependencies

| Component | Requirement |
|---|---|
| Compiler | C++17 (MSVC 2022, or GCC 10+) |
| Build system | CMake 3.16 or newer |
| Qt 6 | `Core`, `Gui`, `Widgets`, `Charts`, `OpenGL`, `OpenGLWidgets` (and `DataVisualization` for the integrated surface viewer) |
| OpenCV 4.x or 5.x | `core`, `imgproc`, `video` (background segmentation), `videoio`, `features2d`/`calib3d` (alignment) |
| Raspberry Pi only | `libcamera`, `gstreamer1.0` |

> On Windows the bundled `CMakeLists.txt` defaults `Qt6_DIR` and `OpenCV_DIR` to local install paths: `D:/Qt/6.11.1/msvc2022_64` + `D:/opencv/build` for MSVC, `D:/Qt/6.11.1/mingw_64` + `D:/opencv/build_mingw/install` for MinGW. Adjust these to your machine, or pass them on the command line as shown below.

---

## Building from source

### Windows 11 (MSVC + Ninja)

1. Install Qt 6 (via the Qt Maintenance Tool) and OpenCV (prebuilt or self-compiled).
2. Open the **Developer Command Prompt for VS 2022**.
3. Configure and build:

```cmd
mkdir build && cd build
cmake -G "Ninja" ^
  -DQt6_DIR="C:/Qt/6.11.1/msvc2022_64/lib/cmake/Qt6" ^
  -DOpenCV_DIR="C:/opencv/build" ^
  -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

On Windows the build automatically runs `windeployqt` and copies the required OpenCV DLLs (and, with MinGW, the compiler runtime) next to the executable, so the build output is ready to run and can be zipped as is for another computer.

### Windows 11 (MinGW, no Visual Studio)

The Qt installer ships MinGW (`D:\Qt\Tools\mingw1310_64`), CMake and Ninja. The OpenCV prebuilt is an MSVC build and cannot be linked with MinGW, so OpenCV is built once from `D:\opencv\sources` into `D:\opencv\build_mingw\install` with the same MinGW (CMake, `-G Ninja`, `BUILD_opencv_world=OFF`, `CMAKE_INSTALL_PREFIX` set to that folder). Then:

```cmd
cmake --preset mingw-release
cmake --build out\build\mingw-release
```

### Linux desktop

```bash
mkdir build && cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

> **Build targets.** Besides the main `DualCam` executable, the project defines `test_exif` (a tiny EXIF round-trip check, run with `ctest`) and a standalone `DualCamViewer`. With `-DDUALCAM_SEPARATE_VIEWER=ON` the main app opens the profile and 3-D surface viewers in that separate process instead of in-process.

---

## Running on a Raspberry Pi 5

**Quick install** on Raspberry Pi OS (Bookworm or Trixie, desktop):

```bash
git clone https://github.com/overl00ker/DualCam.git ~/DualCam
cd ~/DualCam
bash setup-pi.sh
```

[`setup-pi.sh`](setup-pi.sh) installs the packages, builds a Release version into `build/` and adds **DualCam** to the applications menu. Start it from the menu or with `~/DualCam/build/DualCam`. The first build takes a few minutes. To update to the latest version from GitHub later:

```bash
cd ~/DualCam
bash setup-pi.sh --update
```

**What the script installs.** `build-essential cmake ninja-build`, `qt6-base-dev qt6-charts-dev`, Qt Data Visualization — `qt6-datavisualization-dev` on Bookworm (Debian 12), `qt6-datavis3d-dev` on Trixie (Debian 13); the script picks whichever exists — `libopencv-dev`, the GStreamer base and good plugins, and `gstreamer1.0-libcamera` (the `libcamerasrc` element) and `rpicam-apps` if they are missing. To do it by hand, install the same packages and build as for Linux above.

**After installing.** Settings live in `~/.config`, separately from any other computer: calibrate the alignment and `k` on the Pi itself, with its own cameras and the exposure you will measure with.

On the Pi the capture path uses a GStreamer `libcamerasrc` pipeline and expects two `libcamera`-visible cameras; if fewer than two are found the app warns and attempts a fallback.

At startup the app reads the sensor modes from `rpicam-hello --list-cameras` and only offers resolutions the sensors can deliver — the ISP scales down, never up. For the IMX296 global-shutter cameras that means at most 1456×1088 @ 60 fps (offered as the "native" entries); a saved or custom mode above that is shrunk to fit, keeping its aspect ratio.

---

## Hotkeys & presets

Every action in the app — starting cameras, taking a snapshot, toggling diff or focus mode, opening dialogs — is a named command that you can bind to any key combination. Open **Actions & Keybinds** from the **?** button to see and remap them.

Your keybindings, pipeline parameters, the alignment matrix and the channel balance `k` are persisted to the OS configuration location (`AppConfigLocation`) automatically, and named **presets** let you keep several processing configurations and switch between them instantly. Presets do not touch `k`.

---

## Project layout

```
DualCam/
├── main.cpp              # Application entry point
├── mainwindow.h/.cpp     # UI, rendering, analysis, presets, hotkeys,
│                         #   plus the CameraWorker capture/processing thread
├── exif_writer.h/.cpp    # Minimal JPEG EXIF reader/writer
├── resource_monitor.h/.cpp # CPU / temperature / throttling / FPS sampling
├── viewer_*.cpp/.h        # Auxiliary profile / surface viewers (optional target)
├── test_exif.cpp         # EXIF round-trip test target
├── CMakeLists.txt        # Build configuration
├── CMakePresets.json     # CMake preset definitions
├── setup-pi.sh           # One-command install and build on Raspberry Pi OS
├── resources.qrc         # Qt resources (icons)
├── appicon.rc            # Windows icon resource
└── icon.ico / icon.png   # Application icon
```

The bulk of the application lives in `mainwindow.cpp` — it owns the window, the image views, the charts, the snapshot logic and the `CameraWorker` thread that does the actual capture and per-frame processing.

---

## Troubleshooting

**The app builds but can't find Qt or OpenCV at configure time.** Point `Qt6_DIR` and `OpenCV_DIR` at your own install paths, either by editing `CMakeLists.txt` or by passing `-D...` on the `cmake` command line.

**It runs but a camera view stays blank.** Check the source selection in the Capture panel — on Windows the camera index must match a device DirectShow can see; on the Pi the two cameras must be visible to `libcamera`.

**Automatic alignment fails or looks wrong.** Feature-based registration needs texture to lock onto. On low-contrast or near-identical scenes it can fail the internal sanity check; fall back to the manual 6-DOF dialog and align by eye.

**Low frame rate or a hot Pi.** Open the resource monitor (click the CPU pill). If `camera` FPS is below the requested rate while `CAM` thread CPU is near 100 %, the per-frame processing is the limit — the bilateral filter is the most expensive option. `THROT now: throttled` or a temperature above 80 °C means the SoC is being slowed down: improve cooling. `under-voltage` means the power supply is too weak.

**Manual exposure has no effect.** Gain and shutter are passed to the sensors through the libcamera pipeline, so they apply to the Pi's CSI cameras only. USB / DirectShow / V4L2 cameras (the fallback path, e.g. on Windows) keep their own exposure; the status bar says so when you change it.

**The difference of an unpolarized scene is not zero.** Work through the [error budget](#error-budget) in order. Lock the exposure (MAN, same gain and shutter for both cameras), register the channels, then press **Calibrate k** on a matte, evenly lit, unpolarized target. If the line under `k` says *Exposure changed since calibration*, calibrate again. Bright outlines that remain are a focus mismatch (step 8); a fine fixed pattern proportional to brightness is pixel-response non-uniformity. Saturated areas are zero by construction once both cameras clip.

**The difference map is all noise.** Raise the noise floor in the Diff panel (step 12), and make sure the two streams are registered first — an unregistered difference is dominated by registration error, not by the signal you care about.

---

## License & credits

Developed for the scientific and research work of the **Institute of Physics, National Academy of Sciences of Ukraine (IF NASU)**.

Built on the shoulders of the [Qt](https://www.qt.io/) and [OpenCV](https://opencv.org/) projects.
