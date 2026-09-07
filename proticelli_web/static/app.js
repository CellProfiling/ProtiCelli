(() => {
  "use strict";

  const COLORS = {
    microtubules: "#ff4057",
    nucleus: "#3d7eff",
    er: "#ffd23f",
    prediction: "#46f28d",
  };

  const state = {
    config: null,
    input: null,
    layers: new Map(),
    activeLayer: "microtubules",
    currentJob: null,
    jobs: [],
    runs: [],
    currentRun: null,
    renameRunId: null,
    deleteRunId: null,
    selectedPredictionIds: new Set(),
    pollingJobIds: new Set(),
    transform: { zoom: 1, x: 0, y: 0 },
    dragging: null,
    renderPending: false,
    proteinAbort: null,
    uploadSession: null,
    crop: {
      sourceWidth: 0,
      sourceHeight: 0,
      targetWidth: 0,
      targetHeight: 0,
      x: 0,
      y: 0,
      sourceCanvas: null,
      drawing: null,
      dragging: null,
      requestToken: 0,
    },
    compareItems: [],
    compareTransform: { zoom: 1, centerX: 256, centerY: 256 },
    compareDragging: null,
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

  const elements = {
    runtimeStatus: $("#runtimeStatus"),
    runtimeDialog: $("#runtimeDialog"),
    runtimeDetails: $("#runtimeDetails"),
    runtimeGuidance: $("#runtimeGuidance"),
    runtimeCommand: $("#runtimeCommand"),
    activeRunSelect: $("#activeRunSelect"),
    newRunButton: $("#newRunButton"),
    renameRunButton: $("#renameRunButton"),
    newRunDialog: $("#newRunDialog"),
    newRunName: $("#newRunName"),
    newRunUseInput: $("#newRunUseInput"),
    createRunButton: $("#createRunButton"),
    newRunStatus: $("#newRunStatus"),
    renameRunDialog: $("#renameRunDialog"),
    renameRunName: $("#renameRunName"),
    saveRunName: $("#saveRunName"),
    renameRunStatus: $("#renameRunStatus"),
    deleteRunDialog: $("#deleteRunDialog"),
    deleteRunName: $("#deleteRunName"),
    deleteRunSummary: $("#deleteRunSummary"),
    cancelDeleteRun: $("#cancelDeleteRun"),
    confirmDeleteRun: $("#confirmDeleteRun"),
    deleteRunStatus: $("#deleteRunStatus"),
    inputName: $("#inputName"),
    inputMeta: $("#inputMeta"),
    openUpload: $("#openUpload"),
    loadExample: $("#loadExample"),
    emptyExample: $("#emptyExample"),
    uploadDialog: $("#uploadDialog"),
    fileInput: $("#fileInput"),
    dropZone: $("#dropZone"),
    uploadStatus: $("#uploadStatus"),
    channelMapper: $("#channelMapper"),
    mapperSummary: $("#mapperSummary"),
    mappingGrid: $("#mappingGrid"),
    cropSelector: $("#cropSelector"),
    cropCanvasWrap: $("#cropCanvasWrap"),
    cropCanvas: $("#cropCanvas"),
    cropReadout: $("#cropReadout"),
    centerCrop: $("#centerCrop"),
    uploadPixelSize: $("#uploadPixelSize"),
    uploadResample: $("#uploadResample"),
    uploadNormalize: $("#uploadNormalize"),
    normalizationBitDepth: $("#normalizationBitDepth"),
    prepareInput: $("#prepareInput"),
    proteinInput: $("#proteinInput"),
    proteinOptions: $("#proteinOptions"),
    cellLineSelect: $("#cellLineSelect"),
    ensembleInput: $("#ensembleInput"),
    seedInput: $("#seedInput"),
    stepsInput: $("#stepsInput"),
    stepsValue: $("#stepsValue"),
    pixelSizeInput: $("#pixelSizeInput"),
    dtypeSelect: $("#dtypeSelect"),
    generateButton: $("#generateButton"),
    jobProgress: $("#jobProgress"),
    progressMessage: $("#progressMessage"),
    cancelJob: $("#cancelJob"),
    viewerCanvas: $("#viewerCanvas"),
    canvasStage: $("#canvasStage"),
    emptyState: $("#emptyState"),
    viewerTitle: $("#viewerTitle"),
    viewerSubtitle: $("#viewerSubtitle"),
    resetView: $("#resetView"),
    zoomReadout: $("#zoomReadout"),
    openIntensity: $("#openIntensity"),
    closeIntensity: $("#closeIntensity"),
    intensityPanel: $("#intensityPanel"),
    intensityChannel: $("#intensityChannel"),
    channelColor: $("#channelColor"),
    histogramCanvas: $("#histogramCanvas"),
    displayMin: $("#displayMin"),
    displayMax: $("#displayMax"),
    displayMinValue: $("#displayMinValue"),
    displayMaxValue: $("#displayMaxValue"),
    brightness: $("#brightness"),
    brightnessValue: $("#brightnessValue"),
    contrast: $("#contrast"),
    contrastValue: $("#contrastValue"),
    autoIntensity: $("#autoIntensity"),
    resetIntensity: $("#resetIntensity"),
    savePreset: $("#savePreset"),
    channelBar: $("#channelBar"),
    resultBadge: $("#resultBadge"),
    filmstrip: $("#filmstrip"),
    sampleList: $("#sampleList"),
    reliabilityValue: $("#reliabilityValue"),
    reliabilityFill: $("#reliabilityFill"),
    selectionValue: $("#selectionValue"),
    selectedSeed: $("#selectedSeed"),
    pixelType: $("#pixelType"),
    exportButton: $("#exportButton"),
    savePreview: $("#savePreview"),
    engineNote: $("#engineNote"),
    compareFit: $("#compareFit"),
    compareGrid: $("#compareGrid"),
    predictionGallerySection: $("#predictionGallerySection"),
    predictionGallery: $("#predictionGallery"),
    predictionCount: $("#predictionCount"),
    clearPredictionSelection: $("#clearPredictionSelection"),
    galleryComparison: $("#galleryComparison"),
    comparisonTitle: $("#comparisonTitle"),
    returnToGallery: $("#returnToGallery"),
    runsBody: $("#runsBody"),
    refreshRuns: $("#refreshRuns"),
    toast: $("#toast"),
  };

  const compositeCanvas = document.createElement("canvas");
  compositeCanvas.width = 512;
  compositeCanvas.height = 512;
  const compositeContext = compositeCanvas.getContext("2d", { willReadFrequently: true });

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const payload = await response.json();
        message = payload.detail || message;
      } catch (_) {
        // Keep HTTP status when the response is not JSON.
      }
      throw new Error(message);
    }
    return response.json();
  }

  function notify(message, timeout = 3600) {
    elements.toast.textContent = message;
    elements.toast.hidden = false;
    clearTimeout(notify.timer);
    notify.timer = setTimeout(() => { elements.toast.hidden = true; }, timeout);
  }

  function formatValue(value, span = 1) {
    if (!Number.isFinite(value)) return "—";
    if (Math.abs(span) >= 1000 || Math.abs(value) >= 10000) return value.toExponential(2);
    if (Number.isInteger(value) && Math.abs(span) >= 100) return value.toLocaleString();
    if (Math.abs(span) < 0.1) return value.toFixed(4);
    if (Math.abs(span) < 10) return value.toFixed(3);
    return value.toFixed(1);
  }

  function hexToRgb(hex) {
    return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
  }

  function statsToSlider(value, stats) {
    const span = stats.maximum - stats.minimum;
    return span > 0 ? Math.round(((value - stats.minimum) / span) * 255) : 0;
  }

  function sliderToStats(value, stats) {
    return stats.minimum + (Number(value) / 255) * (stats.maximum - stats.minimum);
  }

  function makeLayer({ key, label, color, pixels, stats, visible = true, opacity = 1 }) {
    const preset = loadPreset(key);
    const presetLut = preset?.lut || preset;
    const autoMin = Math.max(0, Math.min(254, statsToSlider(stats.p005, stats)));
    const autoMax = Math.max(autoMin + 1, Math.min(255, statsToSlider(stats.p995, stats)));
    return {
      key, label, color: preset?.color || color, pixels, stats, visible, opacity,
      lut: presetLut || { min: autoMin, max: autoMax, brightness: 100, contrast: 100 },
    };
  }

  function loadPreset(key) {
    try {
      const value = JSON.parse(localStorage.getItem(`proticelli-lut-${key}`));
      const lut = value?.lut || value;
      if (lut && ["min", "max", "brightness", "contrast"].every((field) => Number.isFinite(lut[field]))) return value;
    } catch (_) {
      // Ignore malformed display preferences.
    }
    return null;
  }

  async function loadGrayImage(url) {
    const image = new Image();
    image.decoding = "async";
    image.src = url;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    context.drawImage(image, 0, 0);
    const rgba = context.getImageData(0, 0, canvas.width, canvas.height).data;
    const pixels = new Uint8ClampedArray(canvas.width * canvas.height);
    for (let index = 0, pixel = 0; index < rgba.length; index += 4, pixel += 1) pixels[pixel] = rgba[index];
    return { pixels, width: canvas.width, height: canvas.height };
  }

  function scheduleRender() {
    if (state.renderPending) return;
    state.renderPending = true;
    requestAnimationFrame(() => {
      state.renderPending = false;
      renderComposite();
    });
  }

  function renderComposite() {
    const layers = [...state.layers.values()].filter((layer) => layer.visible && layer.pixels);
    paintComposite(layers, compositeContext);
    drawViewer();
  }

  function paintComposite(layers, context) {
    const output = context.createImageData(512, 512);
    const data = output.data;
    const rgb = layers.map((layer) => hexToRgb(layer.color));
    for (let pixel = 0; pixel < 512 * 512; pixel += 1) {
      let red = 0;
      let green = 0;
      let blue = 0;
      layers.forEach((layer, layerIndex) => {
        const denominator = Math.max(1, layer.lut.max - layer.lut.min);
        let value = (layer.pixels[pixel] - layer.lut.min) / denominator;
        value = ((value - 0.5) * (layer.lut.contrast / 100) + 0.5) * (layer.lut.brightness / 100);
        value = Math.max(0, Math.min(1, value)) * layer.opacity;
        red += rgb[layerIndex][0] * value;
        green += rgb[layerIndex][1] * value;
        blue += rgb[layerIndex][2] * value;
      });
      const offset = pixel * 4;
      data[offset] = Math.min(255, red);
      data[offset + 1] = Math.min(255, green);
      data[offset + 2] = Math.min(255, blue);
      data[offset + 3] = 255;
    }
    context.putImageData(output, 0, 0);
  }

  function drawViewer() {
    const canvas = elements.viewerCanvas;
    const context = canvas.getContext("2d");
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    const targetWidth = Math.max(1, Math.round(width * ratio));
    const targetHeight = Math.max(1, Math.round(height * ratio));
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    if (!state.layers.size) return;
    context.imageSmoothingEnabled = state.transform.zoom < 2.5;
    context.drawImage(
      compositeCanvas,
      state.transform.x,
      state.transform.y,
      512 * state.transform.zoom,
      512 * state.transform.zoom,
    );
    const baseZoom = Math.min(width / 512, height / 512) * 0.92;
    elements.zoomReadout.textContent = `${Math.round((state.transform.zoom / baseZoom) * 100)}%`;
  }

  function fitView() {
    const width = elements.viewerCanvas.clientWidth;
    const height = elements.viewerCanvas.clientHeight;
    const zoom = Math.min(width / 512, height / 512) * 0.92;
    state.transform.zoom = zoom;
    state.transform.x = (width - 512 * zoom) / 2;
    state.transform.y = (height - 512 * zoom) / 2;
    drawViewer();
  }

  function rebuildChannelControls() {
    elements.channelBar.replaceChildren();
    elements.intensityChannel.replaceChildren();
    state.layers.forEach((layer) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "channel-chip";
      button.dataset.channel = layer.key;
      button.setAttribute("aria-pressed", String(layer.visible));
      button.style.setProperty("--chip-color", layer.color);
      const dot = document.createElement("span");
      const label = document.createTextNode(layer.label);
      button.append(dot, label);
      button.addEventListener("click", () => {
        layer.visible = !layer.visible;
        button.setAttribute("aria-pressed", String(layer.visible));
        scheduleRender();
      });
      elements.channelBar.append(button);

      const option = document.createElement("option");
      option.value = layer.key;
      option.textContent = layer.label;
      elements.intensityChannel.append(option);
    });
    if (!state.layers.has(state.activeLayer)) state.activeLayer = state.layers.keys().next().value;
    elements.intensityChannel.value = state.activeLayer;
    syncIntensityControls();
  }

  function syncIntensityControls() {
    const layer = state.layers.get(state.activeLayer);
    if (!layer) return;
    elements.displayMin.value = layer.lut.min;
    elements.displayMax.value = layer.lut.max;
    elements.brightness.value = layer.lut.brightness;
    elements.contrast.value = layer.lut.contrast;
    elements.channelColor.value = layer.color;
    const span = layer.stats.maximum - layer.stats.minimum;
    elements.displayMinValue.textContent = formatValue(sliderToStats(layer.lut.min, layer.stats), span);
    elements.displayMaxValue.textContent = formatValue(sliderToStats(layer.lut.max, layer.stats), span);
    elements.brightnessValue.textContent = `${layer.lut.brightness}%`;
    elements.contrastValue.textContent = `${layer.lut.contrast}%`;
    drawHistogram(layer);
  }

  function drawHistogram(layer) {
    const canvas = elements.histogramCanvas;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 284;
    const height = canvas.clientHeight || 86;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const histogram = layer.stats.histogram || [];
    if (!histogram.length) return;
    const max = Math.max(...histogram, 1);
    const minX = (layer.lut.min / 255) * width;
    const maxX = (layer.lut.max / 255) * width;
    context.fillStyle = `${layer.color}1f`;
    context.fillRect(minX, 0, Math.max(1, maxX - minX), height);
    context.beginPath();
    context.moveTo(0, height);
    histogram.forEach((value, index) => {
      const x = (index / Math.max(1, histogram.length - 1)) * width;
      const y = height - 5 - Math.sqrt(value / max) * (height - 12);
      context.lineTo(x, y);
    });
    context.lineTo(width, height);
    context.closePath();
    context.fillStyle = `${layer.color}8a`;
    context.fill();
    context.strokeStyle = layer.color;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(minX, 0); context.lineTo(minX, height);
    context.moveTo(maxX, 0); context.lineTo(maxX, height);
    context.stroke();
  }

  function updateLutFromControls() {
    const layer = state.layers.get(state.activeLayer);
    if (!layer) return;
    let min = Number(elements.displayMin.value);
    let max = Number(elements.displayMax.value);
    if (min >= max) {
      if (document.activeElement === elements.displayMin) min = Math.max(0, max - 1);
      else max = Math.min(255, min + 1);
    }
    layer.lut = {
      min,
      max,
      brightness: Number(elements.brightness.value),
      contrast: Number(elements.contrast.value),
    };
    syncIntensityControls();
    scheduleRender();
  }

  function autoIntensity() {
    const layer = state.layers.get(state.activeLayer);
    if (!layer) return;
    const min = Math.max(0, Math.min(254, statsToSlider(layer.stats.p005, layer.stats)));
    const max = Math.max(min + 1, Math.min(255, statsToSlider(layer.stats.p995, layer.stats)));
    layer.lut = { min, max, brightness: 100, contrast: 100 };
    syncIntensityControls();
    scheduleRender();
  }

  function resetIntensity() {
    const layer = state.layers.get(state.activeLayer);
    if (!layer) return;
    layer.lut = { min: 0, max: 255, brightness: 100, contrast: 100 };
    syncIntensityControls();
    scheduleRender();
  }

  async function ensureRunForInput(record) {
    if (!state.currentRun) return;
    if (state.currentRun.input_id === record.id) return;
    if (state.currentRun.prediction_count > 0) {
      const createNew = window.confirm(
        "The reference context is locked because this run already has predictions. Create a new run with this reference cell?"
      );
      if (!createNew) throw new Error("Reference change cancelled");
      state.currentRun = await api("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `${record.name} · new run`,
          input_id: record.id,
        }),
      });
      state.selectedPredictionIds.clear();
    } else {
      state.currentRun = await api(`/api/runs/${state.currentRun.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input_id: record.id }),
      });
    }
    await refreshRuns(false);
  }

  function clearInput() {
    state.input = null;
    state.currentJob = null;
    state.layers.clear();
    elements.inputName.textContent = "Add reference channels";
    elements.inputMeta.textContent = "TIFF, PNG, JPEG, BMP, WebP · any field of view";
    elements.openUpload.classList.remove("loaded");
    $(".input-icon", elements.openUpload).textContent = "＋";
    elements.emptyState.hidden = false;
    elements.viewerTitle.textContent = "Reference collection";
    elements.viewerSubtitle.textContent = "Upload a cell or open the example study";
    elements.filmstrip.hidden = true;
    elements.resultBadge.hidden = true;
    elements.channelBar.replaceChildren();
    elements.generateButton.disabled = true;
    elements.savePreview.disabled = true;
    compositeContext.clearRect(0, 0, 512, 512);
    drawViewer();
  }

  async function setInput(record, { attachToRun = true, preserveJob = false } = {}) {
    if (attachToRun) await ensureRunForInput(record);
    state.input = record;
    if (!preserveJob) state.currentJob = null;
    state.layers.clear();
    const loaded = await Promise.all(record.channels.map(async (channel) => {
      const image = await loadGrayImage(`/api/inputs/${record.id}/channels/${channel.key}.png`);
      return makeLayer({ key: channel.key, label: channel.label, color: COLORS[channel.key], pixels: image.pixels, stats: channel });
    }));
    loaded.forEach((layer) => state.layers.set(layer.key, layer));
    state.activeLayer = "microtubules";
    elements.inputName.textContent = record.name;
    const normalized = Boolean(record.preprocessing?.normalization?.applied);
    elements.inputMeta.textContent = `${record.shape[1]} × ${record.shape[0]} · ${record.dtype}${normalized ? " · ProtiCelli normalized" : ""} · SHA ${record.sha256.slice(0, 8)}`;
    elements.openUpload.classList.add("loaded");
    $(".input-icon", elements.openUpload).textContent = "✓";
    elements.pixelType.textContent = record.dtype;
    elements.emptyState.hidden = true;
    elements.viewerTitle.textContent = "Reference channels";
    elements.viewerSubtitle.textContent = `${record.name} · display-only composite`;
    elements.filmstrip.hidden = true;
    elements.resultBadge.hidden = true;
    elements.reliabilityValue.textContent = "—";
    elements.reliabilityFill.style.width = "0";
    elements.selectionValue.textContent = "Reference";
    elements.selectedSeed.textContent = "—";
    elements.exportButton.classList.add("disabled");
    elements.exportButton.setAttribute("aria-disabled", "true");
    elements.savePreview.disabled = false;
    elements.engineNote.textContent = "Reference intensities are rendered through independent, non-destructive LUTs.";
    rebuildChannelControls();
    elements.generateButton.disabled = false;
    syncRunControls();
    requestAnimationFrame(() => { fitView(); scheduleRender(); });
  }

  function allUploadPlanes(session) {
    return session.files.flatMap((file) => file.planes.map((plane) => ({
      ...plane,
      fileIndex: file.file_index,
      fileName: file.name,
      fileFormat: file.format,
      lossy: Boolean(file.lossy),
      value: `${file.file_index}:${plane.index}`,
    })));
  }

  function selectedMappedPlanes() {
    if (!state.uploadSession) return [];
    const planes = allUploadPlanes(state.uploadSession);
    return $$('select[data-role]', elements.mappingGrid).map((select) => {
      const [fileIndex, planeIndex] = select.value.split(":").map(Number);
      const plane = planes.find((item) => item.fileIndex === fileIndex && item.index === planeIndex);
      return { role: select.dataset.role, fileIndex, planeIndex, plane };
    });
  }

  function cropTargetShape(sourceWidth, sourceHeight) {
    const pixelSize = Number(elements.uploadPixelSize.value);
    const nativePixelSize = state.config?.native_pixel_size_um || 0.1067;
    const scale = elements.uploadResample.checked && Number.isFinite(pixelSize) && pixelSize > 0
      ? pixelSize / nativePixelSize
      : 1;
    return {
      width: Math.max(1, Math.round(sourceWidth * scale)),
      height: Math.max(1, Math.round(sourceHeight * scale)),
    };
  }

  function clampCropWindow() {
    const half = 256;
    state.crop.x = Math.max(-half, Math.min(state.crop.targetWidth - half, Math.round(state.crop.x)));
    state.crop.y = Math.max(-half, Math.min(state.crop.targetHeight - half, Math.round(state.crop.y)));
  }

  function centerCropWindow() {
    state.crop.x = Math.floor((state.crop.targetWidth - 512) / 2);
    state.crop.y = Math.floor((state.crop.targetHeight - 512) / 2);
    clampCropWindow();
    drawCropSelector();
  }

  function cropPadding() {
    return {
      left: Math.max(0, -state.crop.x),
      top: Math.max(0, -state.crop.y),
      right: Math.max(0, state.crop.x + 512 - state.crop.targetWidth),
      bottom: Math.max(0, state.crop.y + 512 - state.crop.targetHeight),
    };
  }

  function updateCropReadout() {
    const padding = cropPadding();
    const padded = Object.values(padding).some((value) => value > 0);
    const paddingText = padded
      ? ` · padding L${padding.left} T${padding.top} R${padding.right} B${padding.bottom}`
      : " · no padding";
    elements.cropReadout.textContent = `FOV ${state.crop.targetWidth} × ${state.crop.targetHeight} · window x ${state.crop.x}, y ${state.crop.y}${paddingText}`;
  }

  function buildCropComposite(images) {
    const canvas = document.createElement("canvas");
    canvas.width = images[0].width;
    canvas.height = images[0].height;
    const context = canvas.getContext("2d");
    const output = context.createImageData(canvas.width, canvas.height);
    const colors = [COLORS.microtubules, COLORS.nucleus, COLORS.er].map(hexToRgb);
    for (let pixel = 0; pixel < canvas.width * canvas.height; pixel += 1) {
      let red = 0;
      let green = 0;
      let blue = 0;
      images.forEach((image, index) => {
        const value = image.pixels[pixel] / 255;
        red += colors[index][0] * value;
        green += colors[index][1] * value;
        blue += colors[index][2] * value;
      });
      const offset = pixel * 4;
      output.data[offset] = Math.min(255, red);
      output.data[offset + 1] = Math.min(255, green);
      output.data[offset + 2] = Math.min(255, blue);
      output.data[offset + 3] = 255;
    }
    context.putImageData(output, 0, 0);
    return canvas;
  }

  async function refreshCropPreview(resetWindow = false) {
    const selections = selectedMappedPlanes();
    if (selections.length !== 3 || selections.some((selection) => !selection.plane)) return;
    const shapes = new Set(selections.map((selection) => selection.plane.shape.join("x")));
    if (shapes.size !== 1) {
      state.crop.sourceCanvas = null;
      elements.cropSelector.hidden = true;
      elements.uploadStatus.classList.add("error");
      elements.uploadStatus.textContent = "Mapped landmark planes must have identical dimensions.";
      return;
    }
    const token = ++state.crop.requestToken;
    try {
      const images = await Promise.all(selections.map((selection) => loadGrayImage(
        `/api/uploads/${state.uploadSession.id}/planes/${selection.fileIndex}/${selection.planeIndex}.png?max_size=1024`,
      )));
      if (token !== state.crop.requestToken) return;
      const [sourceHeight, sourceWidth] = selections[0].plane.shape;
      const target = cropTargetShape(sourceWidth, sourceHeight);
      const shapeChanged = target.width !== state.crop.targetWidth || target.height !== state.crop.targetHeight;
      state.crop.sourceWidth = sourceWidth;
      state.crop.sourceHeight = sourceHeight;
      state.crop.targetWidth = target.width;
      state.crop.targetHeight = target.height;
      state.crop.sourceCanvas = buildCropComposite(images);
      elements.cropSelector.hidden = false;
      elements.uploadStatus.classList.remove("error");
      if (resetWindow || shapeChanged) centerCropWindow();
      else {
        clampCropWindow();
        drawCropSelector();
      }
    } catch (error) {
      if (token !== state.crop.requestToken) return;
      state.crop.sourceCanvas = null;
      elements.cropSelector.hidden = true;
      elements.uploadStatus.classList.add("error");
      elements.uploadStatus.textContent = `Could not build crop preview: ${error.message}`;
    }
  }

  function drawCropSelector() {
    if (!state.crop.sourceCanvas || elements.cropSelector.hidden) return;
    const canvas = elements.cropCanvas;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (!width || !height) return;
    canvas.width = Math.max(1, Math.round(width * ratio));
    canvas.height = Math.max(1, Math.round(height * ratio));
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#07100e";
    context.fillRect(0, 0, width, height);

    const worldWidth = state.crop.targetWidth + 512;
    const worldHeight = state.crop.targetHeight + 512;
    const scale = Math.min((width - 20) / worldWidth, (height - 20) / worldHeight);
    const offsetX = (width - worldWidth * scale) / 2 + 256 * scale;
    const offsetY = (height - worldHeight * scale) / 2 + 256 * scale;
    state.crop.drawing = { scale, offsetX, offsetY };

    const imageX = offsetX;
    const imageY = offsetY;
    const imageWidth = state.crop.targetWidth * scale;
    const imageHeight = state.crop.targetHeight * scale;
    context.fillStyle = "#111b17";
    context.fillRect(imageX, imageY, imageWidth, imageHeight);
    context.drawImage(state.crop.sourceCanvas, imageX, imageY, imageWidth, imageHeight);
    context.fillStyle = "rgba(2, 7, 5, .62)";
    context.fillRect(0, 0, width, height);

    const cropX = offsetX + state.crop.x * scale;
    const cropY = offsetY + state.crop.y * scale;
    const cropSize = 512 * scale;
    context.save();
    context.beginPath();
    context.rect(cropX, cropY, cropSize, cropSize);
    context.clip();
    context.fillStyle = "#0b1511";
    context.fillRect(cropX, cropY, cropSize, cropSize);
    context.drawImage(state.crop.sourceCanvas, imageX, imageY, imageWidth, imageHeight);
    context.restore();

    context.strokeStyle = "#e4bd78";
    context.lineWidth = 2;
    context.strokeRect(cropX, cropY, cropSize, cropSize);
    const handle = Math.max(4, Math.min(8, cropSize * .045));
    context.fillStyle = "#f3d69e";
    [[cropX, cropY], [cropX + cropSize, cropY], [cropX, cropY + cropSize], [cropX + cropSize, cropY + cropSize]].forEach(([x, y]) => {
      context.fillRect(x - handle / 2, y - handle / 2, handle, handle);
    });
    updateCropReadout();
  }

  function renderChannelMapper(session) {
    const planes = allUploadPlanes(session);
    elements.channelMapper.hidden = false;
    elements.mapperSummary.textContent = `${session.files.length} file${session.files.length === 1 ? "" : "s"} · ${planes.length} detected plane${planes.length === 1 ? "" : "s"}`;
    elements.mappingGrid.replaceChildren();
    const firstPixelSize = session.files.find((file) => Number.isFinite(file.pixel_size_um))?.pixel_size_um;
    if (firstPixelSize) elements.uploadPixelSize.value = firstPixelSize;
    const roles = [
      ["microtubules", "Microtubules · red"],
      ["nucleus", "Nucleus · blue"],
      ["er", "Endoplasmic reticulum · yellow"],
    ];
    roles.forEach(([role, label], roleIndex) => {
      const card = document.createElement("article");
      card.className = "mapping-card";
      const image = document.createElement("img");
      image.alt = `${label} mapping preview`;
      const content = document.createElement("div");
      const heading = document.createElement("strong");
      heading.textContent = label;
      const select = document.createElement("select");
      select.dataset.role = role;
      select.setAttribute("aria-label", `${label} plane`);
      planes.forEach((plane) => {
        const option = document.createElement("option");
        option.value = plane.value;
        option.textContent = `${plane.fileName} · ${plane.label} · ${plane.shape[1]}×${plane.shape[0]}`;
        select.append(option);
      });
      const suggested = planes.find((plane) => plane.suggested_role === role) || planes[Math.min(roleIndex, planes.length - 1)];
      if (suggested) select.value = suggested.value;
      const detail = document.createElement("small");
      const update = () => {
        const [fileIndex, planeIndex] = select.value.split(":").map(Number);
        const selected = planes.find((plane) => plane.fileIndex === fileIndex && plane.index === planeIndex);
        image.src = `/api/uploads/${session.id}/planes/${fileIndex}/${planeIndex}.png`;
        detail.textContent = selected ? `${selected.fileFormat} · ${selected.dtype} · ${selected.fileName} · ${selected.label}` : "";
      };
      select.addEventListener("change", () => {
        update();
        void refreshCropPreview(false);
      });
      update();
      content.append(heading, select, detail);
      card.append(image, content);
      elements.mappingGrid.append(card);
    });
    void refreshCropPreview(true);
  }

  async function inspectUploadFiles(fileList) {
    const files = [...fileList];
    if (!files.length) return;
    elements.uploadStatus.classList.remove("error");
    elements.uploadStatus.textContent = `Inspecting ${files.length} image file${files.length === 1 ? "" : "s"}…`;
    elements.channelMapper.hidden = true;
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    try {
      state.uploadSession = await api("/api/uploads", { method: "POST", body: form });
      renderChannelMapper(state.uploadSession);
      const lossyNames = state.uploadSession.files.filter((file) => file.lossy).map((file) => file.name);
      elements.uploadStatus.textContent = lossyNames.length
        ? `Planes inspected. ${lossyNames.join(", ")} is lossy; verify the mapping and use TIFF/PNG when quantitative intensities matter.`
        : "Planes inspected. Verify each landmark mapping.";
    } catch (error) {
      elements.uploadStatus.classList.add("error");
      elements.uploadStatus.textContent = error.message;
    } finally {
      elements.fileInput.value = "";
    }
  }

  async function prepareMappedInput() {
    if (!state.uploadSession) return;
    const mapping = {};
    $$('select[data-role]', elements.mappingGrid).forEach((select) => {
      const [fileIndex, planeIndex] = select.value.split(":").map(Number);
      mapping[select.dataset.role] = { file_index: fileIndex, plane_index: planeIndex };
    });
    elements.prepareInput.disabled = true;
    elements.uploadStatus.classList.remove("error");
    elements.uploadStatus.textContent = "Preparing the reproducible 512 × 512 model input…";
    try {
      const record = await api("/api/inputs/assemble", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          upload_id: state.uploadSession.id,
          mapping,
          pixel_size_um: Number(elements.uploadPixelSize.value),
          resample: elements.uploadResample.checked,
          crop_x: state.crop.x,
          crop_y: state.crop.y,
          normalize: elements.uploadNormalize.checked,
          normalization_bit_depth: elements.uploadNormalize.checked && elements.normalizationBitDepth.value !== "auto"
            ? Number(elements.normalizationBitDepth.value)
            : null,
        }),
      });
      await setInput(record);
      elements.pixelSizeInput.value = record.preprocessing?.effective_pixel_size_um || elements.uploadPixelSize.value;
      elements.uploadDialog.close();
      notify(record.preprocessing?.normalization?.applied
        ? "Mapped channels are cropped and ProtiCelli-normalized"
        : "Mapped landmark channels are ready");
    } catch (error) {
      elements.uploadStatus.classList.add("error");
      elements.uploadStatus.textContent = error.message;
    } finally {
      elements.prepareInput.disabled = false;
    }
  }

  async function loadExample() {
    elements.loadExample.disabled = true;
    elements.emptyExample.disabled = true;
    try {
      const record = await api("/api/inputs/example", { method: "POST" });
      await setInput(record);
      if (elements.uploadDialog.open) elements.uploadDialog.close();
      notify("Bundled example cell loaded");
    } catch (error) {
      notify(error.message, 6000);
    } finally {
      elements.loadExample.disabled = false;
      elements.emptyExample.disabled = false;
    }
  }

  async function generate() {
    if (!state.input || !state.currentRun) return;
    const payload = {
      run_id: state.currentRun.id,
      input_id: state.input.id,
      protein: elements.proteinInput.value.trim(),
      cell_line: elements.cellLineSelect.value || null,
      num_samples: Number(elements.ensembleInput.value),
      num_inference_steps: Number(elements.stepsInput.value),
      seed: Number(elements.seedInput.value),
      pixel_size_um: Number(elements.pixelSizeInput.value),
      dtype: elements.dtypeSelect.value,
    };
    elements.generateButton.disabled = true;
    try {
      state.currentRun = await api(`/api/runs/${state.currentRun.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          defaults: {
            num_samples: payload.num_samples,
            num_inference_steps: payload.num_inference_steps,
            seed: payload.seed,
            pixel_size_um: payload.pixel_size_um,
            cell_line: payload.cell_line,
            dtype: payload.dtype,
          },
        }),
      });
      state.currentJob = await api("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (state.config.runtime?.device && state.config.runtime.device !== "cpu") {
        state.config.runtime.dtype = payload.dtype;
        $("span:last-child", elements.runtimeStatus).textContent = `${state.config.runtime.device_name} · ${payload.dtype} · batch ${state.config.runtime.trajectory_batch_size}`;
        renderRuntimeDiagnostics(state.config.runtime);
      }
      state.currentRun.prediction_count += 1;
      syncRunControls();
      upsertJob(state.currentJob);
      renderPredictionGallery();
      elements.jobProgress.hidden = false;
      await pollJob(state.currentJob.id);
    } catch (error) {
      elements.generateButton.disabled = false;
      elements.jobProgress.hidden = true;
      notify(error.message, 6000);
    }
  }

  async function pollJob(jobId) {
    if (state.pollingJobIds.has(jobId)) return;
    state.pollingJobIds.add(jobId);
    try {
      while (true) {
        const job = await api(`/api/jobs/${jobId}`);
        const activeRun = state.currentRun?.id === job.run_id;
        if (activeRun) {
          state.currentJob = job;
          elements.progressMessage.textContent = job.message;
          upsertJob(job);
          renderPredictionGallery();
        }
        if (["succeeded", "failed", "cancelled"].includes(job.status)) {
          if (job.status === "succeeded" && activeRun) await showJob(job);
          else if (job.status === "succeeded") notify(`${job.request.protein} finished in another run`);
          else notify(job.error || job.message, 7000);
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 700));
      }
    } catch (error) {
      notify(error.message, 6000);
    } finally {
      state.pollingJobIds.delete(jobId);
      elements.generateButton.disabled = !state.input;
      if (state.currentJob?.id === jobId) elements.jobProgress.hidden = true;
      await refreshRuns(false);
    }
  }

  async function showJob(job) {
    state.currentJob = job;
    const result = job.result;
    const hasEnsemble = result.images.length > 1;
    const medoid = result.images[result.medoid_index];
    await selectSample(job, medoid.index);
    elements.viewerTitle.textContent = job.request.protein;
    elements.viewerSubtitle.textContent = `${job.input.name} · ${result.images.length} stochastic ${hasEnsemble ? "trajectories" : "trajectory"}`;
    elements.resultBadge.textContent = result.engine === "demo"
      ? (hasEnsemble ? "simulated medoid" : "simulated prediction")
      : (hasEnsemble ? "ensemble medoid" : "prediction");
    elements.resultBadge.hidden = false;
    elements.filmstrip.hidden = false;
    elements.sampleList.replaceChildren();
    const ordered = [...result.images].sort((a, b) => Number(b.is_medoid) - Number(a.is_medoid));
    ordered.forEach((sample) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sample-card";
      button.dataset.index = sample.index;
      button.setAttribute("aria-pressed", String(sample.index === result.medoid_index));
      const image = document.createElement("img");
      image.src = `/api/jobs/${job.id}/images/${sample.index}.png`;
      image.alt = `${sample.is_medoid ? "Medoid" : "Alternative"} sample, seed ${sample.seed}`;
      const label = document.createElement("span");
      label.textContent = sample.is_medoid && hasEnsemble ? `medoid · ${sample.seed}` : `seed ${sample.seed}`;
      button.append(image, label);
      button.addEventListener("click", async () => {
        await selectSample(job, sample.index);
        $$(".sample-card", elements.sampleList).forEach((item) => item.setAttribute("aria-pressed", String(Number(item.dataset.index) === sample.index)));
      });
      elements.sampleList.append(button);
    });
    const reliability = result.reliability;
    elements.reliabilityValue.textContent = reliability == null ? "n/a" : reliability.toFixed(2);
    elements.reliabilityFill.style.width = `${Math.round((reliability || 0) * 100)}%`;
    elements.exportButton.href = `/api/jobs/${job.id}/export`;
    elements.exportButton.classList.remove("disabled");
    elements.exportButton.removeAttribute("aria-disabled");
    elements.engineNote.textContent = result.engine === "demo"
      ? "Demo renderer active: these are simulated UI-development images, not ProtiCelli predictions."
      : hasEnsemble
        ? "ProtiCelli checkpoint inference. Reliability summarizes ensemble agreement, not pixelwise confidence."
        : "ProtiCelli checkpoint inference. Increase Ensemble size to 2 or more to calculate agreement.";
    notify(result.engine === "demo"
      ? (hasEnsemble ? "Simulated ensemble ready" : "Simulated prediction ready")
      : (hasEnsemble ? "ProtiCelli ensemble ready" : "ProtiCelli prediction ready"));
  }

  async function selectSample(job, index) {
    const sample = job.result.images.find((item) => item.index === index);
    if (!sample) return;
    const loaded = await loadGrayImage(`/api/jobs/${job.id}/images/${index}.png?${Date.now()}`);
    const prediction = makeLayer({
      key: "prediction",
      label: `Prediction · ${job.request.protein}`,
      color: COLORS.prediction,
      pixels: loaded.pixels,
      stats: sample.statistics,
      opacity: 1,
    });
    state.layers.set("prediction", prediction);
    state.activeLayer = "prediction";
    elements.selectionValue.textContent = job.result.images.length === 1
      ? "Prediction"
      : sample.is_medoid ? "Medoid" : `Alternative ${sample.index + 1}`;
    elements.selectedSeed.textContent = sample.seed;
    elements.pixelType.textContent = "float32";
    rebuildChannelControls();
    scheduleRender();
  }

  function upsertJob(job) {
    const index = state.jobs.findIndex((item) => item.id === job.id);
    if (index >= 0) state.jobs[index] = job;
    else state.jobs.unshift(job);
  }

  function syncRunControls() {
    if (!state.currentRun) return;
    const defaults = state.currentRun.defaults || {};
    elements.cellLineSelect.value = defaults.cell_line || "";
    elements.cellLineSelect.disabled = false;
    if (defaults.num_samples != null) elements.ensembleInput.value = defaults.num_samples;
    if (defaults.num_inference_steps != null) {
      elements.stepsInput.value = defaults.num_inference_steps;
      elements.stepsValue.textContent = defaults.num_inference_steps;
    }
    if (defaults.seed != null) elements.seedInput.value = defaults.seed;
    if (defaults.pixel_size_um != null) elements.pixelSizeInput.value = defaults.pixel_size_um;
    const precision = defaults.dtype || "float32";
    const precisionOption = elements.dtypeSelect.querySelector(`option[value="${precision}"]`);
    elements.dtypeSelect.value = precisionOption?.disabled ? "float32" : precision;
  }

  function renderRunSelector() {
    elements.activeRunSelect.replaceChildren();
    state.runs.forEach((run) => {
      const option = document.createElement("option");
      option.value = run.id;
      option.textContent = `${run.name} · ${run.prediction_count} prediction${run.prediction_count === 1 ? "" : "s"}`;
      option.selected = run.id === state.currentRun?.id;
      elements.activeRunSelect.append(option);
    });
  }

  async function switchRun(runId) {
    const run = await api(`/api/runs/${runId}`);
    state.currentRun = run;
    state.jobs = run.predictions || [];
    state.currentJob = state.jobs.find((job) => job.status === "running") || state.jobs.find((job) => job.status === "queued") || null;
    elements.jobProgress.hidden = !state.currentJob;
    if (state.currentJob) elements.progressMessage.textContent = state.currentJob.message;
    state.selectedPredictionIds.clear();
    renderRunSelector();
    syncRunControls();
    renderPredictionGallery();
    await updateGalleryComparison();
    if (run.input) await setInput(run.input, { attachToRun: false, preserveJob: true });
    else clearInput();
    activateTab("gallery");
    if (state.currentJob) void pollJob(state.currentJob.id);
  }

  async function refreshRuns(loadContext = true) {
    try {
      const response = await api("/api/runs");
      state.runs = response.items;
      if (!state.runs.length) {
        const created = await api("/api/runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: "Untitled run" }),
        });
        state.runs = [created];
      }
      const currentId = state.currentRun?.id;
      state.currentRun = state.runs.find((run) => run.id === currentId) || state.runs[0];
      const predictions = await api(`/api/jobs?run_id=${encodeURIComponent(state.currentRun.id)}`);
      state.jobs = predictions.items;
      const activeJob = state.jobs.find((job) => job.status === "running")
        || state.jobs.find((job) => job.status === "queued");
      if (activeJob) state.currentJob = activeJob;
      else if (!state.jobs.some((job) => job.id === state.currentJob?.id)) state.currentJob = null;
      elements.jobProgress.hidden = !activeJob;
      if (activeJob) elements.progressMessage.textContent = activeJob.message;
      renderRunSelector();
      renderRuns();
      renderPredictionGallery();
      syncRunControls();
      await updateGalleryComparison();
      if (loadContext) {
        if (state.currentRun.input) await setInput(
          state.currentRun.input,
          { attachToRun: false, preserveJob: true },
        );
        else clearInput();
        if (state.currentJob) void pollJob(state.currentJob.id);
      }
    } catch (error) {
      notify(error.message);
    }
  }

  function renderRuns() {
    elements.runsBody.replaceChildren();
    state.runs.forEach((run) => {
      const row = document.createElement("tr");
      const cells = [
        run.name,
        run.input?.name || "Not assigned",
        run.conditions?.join(", ") || "—",
        String(run.prediction_count),
        run.active_count ? String(run.active_count) : "—",
        new Date(run.updated_at).toLocaleString(),
      ];
      cells.forEach((value) => {
        const cell = document.createElement("td");
        cell.textContent = value;
        row.append(cell);
      });
      const action = document.createElement("td");
      const open = document.createElement("button");
      open.type = "button";
      open.textContent = "Open";
      open.addEventListener("click", () => switchRun(run.id));
      const rename = document.createElement("button");
      rename.type = "button";
      rename.textContent = "Rename";
      rename.addEventListener("click", () => openRenameRun(run));
      const duplicate = document.createElement("button");
      duplicate.type = "button";
      duplicate.textContent = "Duplicate";
      duplicate.addEventListener("click", async () => {
        const copy = await api(`/api/runs/${run.id}/duplicate`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
        await refreshRuns(false);
        await switchRun(copy.id);
      });
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "run-delete-button";
      remove.textContent = "Delete";
      remove.addEventListener("click", () => openDeleteRun(run));
      action.append(open, rename, duplicate, remove);
      row.append(action);
      elements.runsBody.append(row);
    });
  }

  function resultSummary(result) {
    const count = result.images.length;
    const trajectories = `${count} ${count === 1 ? "trajectory" : "trajectories"}`;
    const reliability = result.reliability == null ? "n/a" : result.reliability.toFixed(2);
    return `${trajectories} · reliability ${reliability}`;
  }

  function renderPredictionGallery() {
    elements.predictionGallery.replaceChildren();
    elements.predictionCount.textContent = `${state.jobs.length} prediction${state.jobs.length === 1 ? "" : "s"}`;
    if (!state.jobs.length) {
      const empty = document.createElement("div");
      empty.className = "prediction-empty";
      empty.textContent = "Generated proteins will accumulate here without replacing earlier results.";
      elements.predictionGallery.append(empty);
      elements.clearPredictionSelection.hidden = true;
      return;
    }
    const validIds = new Set(state.jobs.filter((job) => job.status === "succeeded").map((job) => job.id));
    state.selectedPredictionIds.forEach((id) => { if (!validIds.has(id)) state.selectedPredictionIds.delete(id); });
    elements.clearPredictionSelection.hidden = state.selectedPredictionIds.size === 0;
    state.jobs.forEach((job) => {
      const card = document.createElement("article");
      card.className = `prediction-card ${job.status}`;
      card.dataset.jobId = job.id;
      if (job.status === "succeeded") {
        const medoid = job.result.images[job.result.medoid_index];
        const imageButton = document.createElement("button");
        imageButton.type = "button";
        imageButton.className = "prediction-open";
        const image = document.createElement("img");
        image.src = `/api/jobs/${job.id}/images/${medoid.index}.png`;
        image.alt = job.result.images.length === 1
          ? `${job.request.protein} prediction`
          : `${job.request.protein} ensemble medoid`;
        imageButton.append(image);
        imageButton.addEventListener("click", () => showJob(job));
        card.append(imageButton);
      } else {
        const pending = document.createElement("div");
        pending.className = "prediction-pending";
        pending.textContent = job.status === "running" ? "Generating…" : job.status;
        card.append(pending);
      }
      const footer = document.createElement("footer");
      const copy = document.createElement("div");
      const label = document.createElement("strong");
      label.textContent = job.request.protein;
      const detail = document.createElement("small");
      detail.textContent = job.status === "succeeded"
        ? resultSummary(job.result)
        : job.message;
      copy.append(label, detail);
      footer.append(copy);
      if (job.status === "succeeded") {
        const compare = document.createElement("label");
        compare.className = "prediction-select";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = state.selectedPredictionIds.has(job.id);
        const text = document.createElement("span");
        text.textContent = "Compare";
        checkbox.addEventListener("change", async () => {
          if (checkbox.checked) state.selectedPredictionIds.add(job.id);
          else state.selectedPredictionIds.delete(job.id);
          renderPredictionGallery();
          await updateGalleryComparison();
        });
        compare.append(checkbox, text);
        footer.append(compare);
      } else if (["queued", "running"].includes(job.status)) {
        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.textContent = "Stop";
        cancel.addEventListener("click", async () => {
          try {
            const cancelled = await api(`/api/jobs/${job.id}`, { method: "DELETE" });
            upsertJob(cancelled);
            renderPredictionGallery();
          } catch (error) {
            notify(error.message, 6000);
          }
        });
        footer.append(cancel);
      }
      card.append(footer);
      elements.predictionGallery.append(card);
    });
  }

  async function updateGalleryComparison() {
    const selected = state.jobs.filter(
      (job) => state.selectedPredictionIds.has(job.id) && job.status === "succeeded"
    );
    elements.clearPredictionSelection.hidden = selected.length === 0;
    if (selected.length < 2) {
      elements.galleryComparison.hidden = true;
      state.compareItems = [];
      elements.compareGrid.replaceChildren();
      return;
    }
    elements.galleryComparison.hidden = false;
    elements.comparisonTitle.textContent = `${selected.length} selected predictions`;
    await renderComparison(selected);
  }

  async function renderComparison(jobs) {
    state.compareItems = [];
    elements.compareGrid.replaceChildren();
    const referenceLayers = [...state.layers.values()].filter((layer) => layer.key !== "prediction" && layer.visible);
    for (const job of jobs) {
      const medoid = job.result.images[job.result.medoid_index];
      const loaded = await loadGrayImage(`/api/jobs/${job.id}/images/${medoid.index}.png?${Date.now()}`);
      const prediction = makeLayer({
        key: "prediction",
        label: job.request.protein,
        color: COLORS.prediction,
        pixels: loaded.pixels,
        stats: medoid.statistics,
        opacity: 1,
      });
      const source = document.createElement("canvas");
      source.width = 512;
      source.height = 512;
      paintComposite([...referenceLayers, prediction], source.getContext("2d"));
      const card = document.createElement("article");
      card.className = "compare-card";
      const canvas = document.createElement("canvas");
      canvas.setAttribute(
        "aria-label",
        job.result.images.length === 1
          ? `${job.request.protein} prediction`
          : `${job.request.protein} ensemble medoid`,
      );
      const footer = document.createElement("footer");
      const label = document.createElement("span");
      label.textContent = job.request.protein;
      const detail = document.createElement("small");
      detail.textContent = resultSummary(job.result);
      footer.append(label, detail);
      card.append(canvas, footer);
      elements.compareGrid.append(card);
      const item = { job, source, canvas };
      state.compareItems.push(item);
      bindCompareCanvas(item);
    }
    fitComparison();
  }

  function clampCompareCenter() {
    const half = 256 / state.compareTransform.zoom;
    state.compareTransform.centerX = Math.max(half, Math.min(512 - half, state.compareTransform.centerX));
    state.compareTransform.centerY = Math.max(half, Math.min(512 - half, state.compareTransform.centerY));
  }

  function drawComparison() {
    clampCompareCenter();
    const sourceSize = 512 / state.compareTransform.zoom;
    const sourceX = state.compareTransform.centerX - sourceSize / 2;
    const sourceY = state.compareTransform.centerY - sourceSize / 2;
    state.compareItems.forEach(({ source, canvas }) => {
      const ratio = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      canvas.width = Math.max(1, Math.round(width * ratio));
      canvas.height = Math.max(1, Math.round(height * ratio));
      const context = canvas.getContext("2d");
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);
      context.imageSmoothingEnabled = state.compareTransform.zoom < 2.5;
      context.drawImage(source, sourceX, sourceY, sourceSize, sourceSize, 0, 0, width, height);
    });
  }

  function fitComparison() {
    state.compareTransform = { zoom: 1, centerX: 256, centerY: 256 };
    drawComparison();
  }

  function bindCompareCanvas(item) {
    const canvas = item.canvas;
    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      state.compareDragging = {
        x: event.clientX,
        y: event.clientY,
        centerX: state.compareTransform.centerX,
        centerY: state.compareTransform.centerY,
        width: canvas.clientWidth,
      };
      canvas.classList.add("dragging");
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!state.compareDragging) return;
      const scale = 512 / state.compareTransform.zoom / state.compareDragging.width;
      state.compareTransform.centerX = state.compareDragging.centerX - (event.clientX - state.compareDragging.x) * scale;
      state.compareTransform.centerY = state.compareDragging.centerY - (event.clientY - state.compareDragging.y) * scale;
      drawComparison();
    });
    const stop = () => { state.compareDragging = null; $$(".compare-card canvas").forEach((itemCanvas) => itemCanvas.classList.remove("dragging")); };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const fractionX = (event.clientX - rect.left) / rect.width;
      const fractionY = (event.clientY - rect.top) / rect.height;
      const oldSize = 512 / state.compareTransform.zoom;
      const sourceX = state.compareTransform.centerX + (fractionX - .5) * oldSize;
      const sourceY = state.compareTransform.centerY + (fractionY - .5) * oldSize;
      const zoom = Math.max(1, Math.min(12, state.compareTransform.zoom * Math.exp(-event.deltaY * .0015)));
      const newSize = 512 / zoom;
      state.compareTransform.zoom = zoom;
      state.compareTransform.centerX = sourceX - (fractionX - .5) * newSize;
      state.compareTransform.centerY = sourceY - (fractionY - .5) * newSize;
      drawComparison();
    }, { passive: false });
    new ResizeObserver(drawComparison).observe(canvas);
  }

  function openNewRunDialog() {
    const now = new Date();
    elements.newRunName.value = `Run · ${now.toLocaleDateString()} ${now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
    elements.newRunUseInput.checked = Boolean(state.input);
    elements.newRunUseInput.disabled = !state.input;
    elements.newRunStatus.textContent = "";
    elements.newRunDialog.showModal();
  }

  function openRenameRun(run = state.currentRun) {
    if (!run) return;
    state.renameRunId = run.id;
    elements.renameRunName.value = run.name;
    elements.renameRunStatus.textContent = "";
    elements.renameRunStatus.classList.remove("error");
    elements.renameRunDialog.showModal();
    requestAnimationFrame(() => elements.renameRunName.select());
  }

  async function saveRunName() {
    if (!state.renameRunId) return;
    const name = elements.renameRunName.value.trim();
    if (!name) {
      elements.renameRunStatus.classList.add("error");
      elements.renameRunStatus.textContent = "Enter a run name.";
      return;
    }
    elements.saveRunName.disabled = true;
    try {
      const renamed = await api(`/api/runs/${state.renameRunId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (state.currentRun?.id === renamed.id) state.currentRun = renamed;
      elements.renameRunDialog.close();
      await refreshRuns(false);
      notify(`Run renamed to “${renamed.name}”`);
    } catch (error) {
      elements.renameRunStatus.classList.add("error");
      elements.renameRunStatus.textContent = error.message;
    } finally {
      elements.saveRunName.disabled = false;
    }
  }

  function openDeleteRun(run) {
    state.deleteRunId = run.id;
    elements.deleteRunName.textContent = `“${run.name}”`;
    const count = run.prediction_count || 0;
    elements.deleteRunSummary.textContent = count
      ? `This permanently removes the run and ${count} prediction${count === 1 ? "" : "s"}, including their exported artifacts.`
      : "This permanently removes the empty run.";
    elements.deleteRunStatus.classList.remove("error");
    elements.confirmDeleteRun.disabled = Boolean(run.active_count);
    elements.deleteRunStatus.textContent = run.active_count
      ? "Cancel active inference and wait for it to stop before deleting this run."
      : "";
    elements.deleteRunDialog.showModal();
  }

  async function deleteRun() {
    if (!state.deleteRunId) return;
    const run = state.runs.find((item) => item.id === state.deleteRunId);
    if (!run) return;
    elements.confirmDeleteRun.disabled = true;
    elements.deleteRunStatus.classList.remove("error");
    elements.deleteRunStatus.textContent = "Deleting run…";
    try {
      await api(`/api/runs/${run.id}`, { method: "DELETE" });
      const deletedCurrentRun = state.currentRun?.id === run.id;
      if (deletedCurrentRun) {
        state.currentRun = null;
        state.currentJob = null;
        state.jobs = [];
        state.selectedPredictionIds.clear();
      }
      state.deleteRunId = null;
      elements.deleteRunDialog.close();
      await refreshRuns(deletedCurrentRun);
      notify(`Run “${run.name}” deleted`);
    } catch (error) {
      elements.deleteRunStatus.classList.add("error");
      elements.deleteRunStatus.textContent = error.message;
      elements.confirmDeleteRun.disabled = false;
    }
  }

  async function createRun() {
    elements.createRunButton.disabled = true;
    elements.newRunStatus.classList.remove("error");
    elements.newRunStatus.textContent = "Creating workspace…";
    try {
      const run = await api("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: elements.newRunName.value.trim() || "Untitled run",
          input_id: elements.newRunUseInput.checked && state.input ? state.input.id : null,
        }),
      });
      elements.newRunDialog.close();
      await refreshRuns(false);
      await switchRun(run.id);
      notify(`Run “${run.name}” created`);
    } catch (error) {
      elements.newRunStatus.classList.add("error");
      elements.newRunStatus.textContent = error.message;
    } finally {
      elements.createRunButton.disabled = false;
    }
  }

  function activateTab(name) {
    $$(".tab").forEach((tab) => {
      const active = tab.dataset.tab === name;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    $$(".workspace").forEach((panel) => { panel.hidden = panel.dataset.panel !== name; });
    if (name === "gallery") requestAnimationFrame(() => { drawViewer(); drawComparison(); });
    if (name === "runs") refreshRuns(false);
  }

  async function searchProteins(term) {
    if (state.proteinAbort) state.proteinAbort.abort();
    state.proteinAbort = new AbortController();
    try {
      const payload = await api(`/api/vocabulary/proteins?q=${encodeURIComponent(term)}&limit=18`, { signal: state.proteinAbort.signal });
      elements.proteinOptions.replaceChildren();
      payload.items.forEach((name) => {
        const option = document.createElement("button");
        option.type = "button";
        option.role = "option";
        option.textContent = name;
        option.addEventListener("mousedown", (event) => {
          event.preventDefault();
          elements.proteinInput.value = name;
          elements.proteinOptions.hidden = true;
        });
        elements.proteinOptions.append(option);
      });
      elements.proteinOptions.hidden = payload.items.length === 0;
    } catch (error) {
      if (error.name !== "AbortError") notify(error.message);
    }
  }

  function savePreview() {
    if (!state.layers.size) return;
    const link = document.createElement("a");
    const target = state.currentJob?.request?.protein || "reference";
    link.download = `proticelli-${target.toLowerCase().replace(/[^a-z0-9]+/g, "-")}-display.png`;
    link.href = compositeCanvas.toDataURL("image/png");
    link.click();
    notify("Display preview saved; raw image values were not changed");
  }

  function renderRuntimeDiagnostics(runtime) {
    elements.runtimeDetails.replaceChildren();
    const rows = runtime ? [
      ["Platform", [runtime.platform_system, runtime.platform_machine].filter(Boolean).join(" · ") || "—"],
      ["Selected device", runtime.device || "—"],
      ["Accelerator", runtime.accelerator?.toUpperCase() || "CPU"],
      ["PyTorch", runtime.torch_version || "Not detected"],
      ["CUDA build", runtime.torch_cuda_build || "No"],
      ["ROCm build", runtime.torch_hip_build || "No"],
      ["CUDA/ROCm available", runtime.cuda_available ? "Yes" : "No"],
      ["Apple MPS built", runtime.mps_built ? "Yes" : "No"],
      ["Apple MPS available", runtime.mps_available ? "Yes" : "No"],
      ["NVIDIA driver", runtime.nvidia_driver_detected ? "Detected" : "Not detected"],
      ["Compute device", runtime.nvidia_gpus?.join(" · ") || runtime.device_name || "Not available"],
    ] : [["Runtime", "Demo renderer"]];
    rows.forEach(([term, value]) => {
      const dt = document.createElement("dt");
      const dd = document.createElement("dd");
      dt.textContent = term;
      dd.textContent = value;
      elements.runtimeDetails.append(dt, dd);
    });
    const guidance = {
      pytorch_cpu_build: "An NVIDIA driver may be present, but this Python environment has a CPU-only PyTorch wheel. Install a CUDA-enabled PyTorch wheel into this same environment, then restart ProtiCelli.",
      nvidia_driver_missing: "PyTorch includes CUDA support but cannot see an NVIDIA driver. Update the NVIDIA driver and restart the computer before trying again.",
      cuda_runtime_unavailable: "The NVIDIA driver is visible, but PyTorch could not initialize CUDA. Run the diagnostic command below and check that the driver supports the installed CUDA wheel.",
      cpu_requested: "CPU execution was explicitly requested through PROTICELLI_WEB_DEVICE. Remove that setting or set it to auto.",
      torch_missing: "PyTorch is missing from this Python environment. Re-run the local installer before starting the Gallery.",
      cuda_ready: "CUDA is ready. ProtiCelli will keep the model resident on the GPU and batch ensemble trajectories.",
      rocm_ready: "ROCm is ready. ProtiCelli uses PyTorch's CUDA-compatible device API on this Linux GPU.",
      mps_ready: "Apple Metal acceleration is ready. ProtiCelli will use MPS with float32 compute.",
      mps_not_built: "This PyTorch installation does not include Apple MPS support. Re-run the macOS launcher in a fresh environment or use CPU.",
      mps_unavailable: "PyTorch includes MPS, but the current macOS version or hardware cannot make it available. ProtiCelli is using CPU.",
      cpu_ready: "No supported accelerator is available in this environment, so ProtiCelli is using CPU.",
      unsupported_device: "PROTICELLI_WEB_DEVICE is invalid. Use auto, cpu, mps, cuda, or cuda:N.",
    };
    elements.runtimeGuidance.textContent = runtime
      ? (guidance[runtime.status] || "This runtime is using CPU. Choose an accelerator supported by this operating system and PyTorch installation.")
      : "The demo renderer does not run the ProtiCelli checkpoint.";
    elements.runtimeCommand.textContent = "proticelli-web --diagnose";
  }

  function bindEvents() {
    $$(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));
    elements.activeRunSelect.addEventListener("change", () => switchRun(elements.activeRunSelect.value));
    elements.newRunButton.addEventListener("click", openNewRunDialog);
    elements.renameRunButton.addEventListener("click", () => openRenameRun());
    elements.createRunButton.addEventListener("click", createRun);
    elements.saveRunName.addEventListener("click", saveRunName);
    elements.cancelDeleteRun.addEventListener("click", () => elements.deleteRunDialog.close());
    elements.confirmDeleteRun.addEventListener("click", deleteRun);
    elements.renameRunName.addEventListener("keydown", (event) => {
      if (event.key === "Enter") saveRunName();
    });
    elements.runtimeStatus.addEventListener("click", () => elements.runtimeDialog.showModal());
    elements.returnToGallery.addEventListener("click", () => activateTab("gallery"));
    elements.openUpload.addEventListener("click", () => elements.uploadDialog.showModal());
    elements.loadExample.addEventListener("click", loadExample);
    elements.emptyExample.addEventListener("click", loadExample);
    elements.fileInput.addEventListener("change", () => inspectUploadFiles(elements.fileInput.files));
    ["dragenter", "dragover"].forEach((name) => elements.dropZone.addEventListener(name, (event) => { event.preventDefault(); elements.dropZone.classList.add("dragging"); }));
    ["dragleave", "drop"].forEach((name) => elements.dropZone.addEventListener(name, (event) => { event.preventDefault(); elements.dropZone.classList.remove("dragging"); }));
    elements.dropZone.addEventListener("drop", (event) => inspectUploadFiles(event.dataTransfer.files));
    elements.prepareInput.addEventListener("click", prepareMappedInput);
    elements.centerCrop.addEventListener("click", centerCropWindow);
    elements.uploadPixelSize.addEventListener("change", () => void refreshCropPreview(true));
    elements.uploadResample.addEventListener("change", () => void refreshCropPreview(true));
    elements.uploadNormalize.addEventListener("change", () => {
      elements.normalizationBitDepth.disabled = !elements.uploadNormalize.checked;
    });
    elements.stepsInput.addEventListener("input", () => { elements.stepsValue.textContent = elements.stepsInput.value; });
    elements.ensembleInput.addEventListener("change", () => {
      const requested = Math.trunc(Number(elements.ensembleInput.value) || 1);
      elements.ensembleInput.value = String(Math.max(1, Math.min(50, requested)));
    });
    elements.generateButton.addEventListener("click", generate);
    elements.cancelJob.addEventListener("click", async () => {
      if (!state.currentJob) return;
      try { await api(`/api/jobs/${state.currentJob.id}`, { method: "DELETE" }); } catch (error) { notify(error.message); }
    });
    elements.openIntensity.addEventListener("click", () => {
      const open = elements.intensityPanel.hidden;
      elements.intensityPanel.hidden = !open;
      elements.openIntensity.setAttribute("aria-expanded", String(open));
      if (open) syncIntensityControls();
    });
    elements.closeIntensity.addEventListener("click", () => { elements.intensityPanel.hidden = true; elements.openIntensity.setAttribute("aria-expanded", "false"); });
    elements.intensityChannel.addEventListener("change", () => { state.activeLayer = elements.intensityChannel.value; syncIntensityControls(); });
    elements.channelColor.addEventListener("input", () => {
      const layer = state.layers.get(state.activeLayer);
      if (!layer) return;
      layer.color = elements.channelColor.value;
      const chip = elements.channelBar.querySelector(`[data-channel="${layer.key}"]`);
      if (chip) chip.style.setProperty("--chip-color", layer.color);
      drawHistogram(layer);
      scheduleRender();
    });
    [elements.displayMin, elements.displayMax, elements.brightness, elements.contrast].forEach((control) => control.addEventListener("input", updateLutFromControls));
    elements.autoIntensity.addEventListener("click", autoIntensity);
    elements.resetIntensity.addEventListener("click", resetIntensity);
    elements.savePreset.addEventListener("click", () => {
      const layer = state.layers.get(state.activeLayer);
      if (!layer) return;
      localStorage.setItem(`proticelli-lut-${layer.key}`, JSON.stringify({ color: layer.color, lut: layer.lut }));
      notify(`${layer.label} color and display preset saved in this browser`);
    });
    elements.resetView.addEventListener("click", fitView);
    elements.savePreview.addEventListener("click", savePreview);
    elements.refreshRuns.addEventListener("click", () => refreshRuns(false));
    elements.compareFit.addEventListener("click", fitComparison);
    elements.clearPredictionSelection.addEventListener("click", async () => {
      state.selectedPredictionIds.clear();
      renderPredictionGallery();
      await updateGalleryComparison();
    });

    elements.cropCanvas.addEventListener("pointerdown", (event) => {
      if (!state.crop.drawing) return;
      const bounds = elements.cropCanvas.getBoundingClientRect();
      const screenX = event.clientX - bounds.left;
      const screenY = event.clientY - bounds.top;
      const { scale, offsetX, offsetY } = state.crop.drawing;
      const worldX = (screenX - offsetX) / scale;
      const worldY = (screenY - offsetY) / scale;
      const inside = worldX >= state.crop.x && worldX <= state.crop.x + 512
        && worldY >= state.crop.y && worldY <= state.crop.y + 512;
      if (!inside) {
        state.crop.x = Math.round(worldX - 256);
        state.crop.y = Math.round(worldY - 256);
        clampCropWindow();
        drawCropSelector();
      }
      elements.cropCanvas.setPointerCapture(event.pointerId);
      state.crop.dragging = {
        clientX: event.clientX,
        clientY: event.clientY,
        x: state.crop.x,
        y: state.crop.y,
        scale,
      };
      elements.cropCanvas.classList.add("dragging");
    });
    elements.cropCanvas.addEventListener("pointermove", (event) => {
      if (!state.crop.dragging) return;
      state.crop.x = state.crop.dragging.x
        + (event.clientX - state.crop.dragging.clientX) / state.crop.dragging.scale;
      state.crop.y = state.crop.dragging.y
        + (event.clientY - state.crop.dragging.clientY) / state.crop.dragging.scale;
      clampCropWindow();
      drawCropSelector();
    });
    const stopCropDrag = () => {
      state.crop.dragging = null;
      elements.cropCanvas.classList.remove("dragging");
    };
    elements.cropCanvas.addEventListener("pointerup", stopCropDrag);
    elements.cropCanvas.addEventListener("pointercancel", stopCropDrag);
    new ResizeObserver(drawCropSelector).observe(elements.cropCanvasWrap);

    let proteinTimer;
    elements.proteinInput.addEventListener("input", () => {
      clearTimeout(proteinTimer);
      proteinTimer = setTimeout(() => searchProteins(elements.proteinInput.value), 150);
    });
    elements.proteinInput.addEventListener("focus", () => searchProteins(elements.proteinInput.value));
    elements.proteinInput.addEventListener("blur", () => setTimeout(() => { elements.proteinOptions.hidden = true; }, 120));

    elements.viewerCanvas.addEventListener("pointerdown", (event) => {
      if (!state.layers.size) return;
      elements.viewerCanvas.setPointerCapture(event.pointerId);
      state.dragging = { x: event.clientX, y: event.clientY, originX: state.transform.x, originY: state.transform.y };
      elements.viewerCanvas.classList.add("dragging");
    });
    elements.viewerCanvas.addEventListener("pointermove", (event) => {
      if (!state.dragging) return;
      state.transform.x = state.dragging.originX + event.clientX - state.dragging.x;
      state.transform.y = state.dragging.originY + event.clientY - state.dragging.y;
      drawViewer();
    });
    const stopDrag = () => { state.dragging = null; elements.viewerCanvas.classList.remove("dragging"); };
    elements.viewerCanvas.addEventListener("pointerup", stopDrag);
    elements.viewerCanvas.addEventListener("pointercancel", stopDrag);
    elements.viewerCanvas.addEventListener("wheel", (event) => {
      if (!state.layers.size) return;
      event.preventDefault();
      const bounds = elements.viewerCanvas.getBoundingClientRect();
      const x = event.clientX - bounds.left;
      const y = event.clientY - bounds.top;
      const sourceX = (x - state.transform.x) / state.transform.zoom;
      const sourceY = (y - state.transform.y) / state.transform.zoom;
      const fit = Math.min(bounds.width / 512, bounds.height / 512) * 0.92;
      const next = Math.max(fit * 0.5, Math.min(fit * 12, state.transform.zoom * Math.exp(-event.deltaY * 0.0015)));
      state.transform.x = x - sourceX * next;
      state.transform.y = y - sourceY * next;
      state.transform.zoom = next;
      drawViewer();
    }, { passive: false });
    new ResizeObserver(() => drawViewer()).observe(elements.canvasStage);
  }

  async function init() {
    bindEvents();
    try {
      state.config = await api("/api/config");
      const isDemo = state.config.engine === "demo";
      const runtime = state.config.runtime;
      const isCuda = runtime?.device?.startsWith("cuda");
      const isMps = runtime?.device === "mps";
      const isAccelerated = isCuda || isMps;
      const float16Option = elements.dtypeSelect.querySelector('option[value="float16"]');
      if (float16Option) float16Option.disabled = !isCuda && !isDemo;
      elements.runtimeStatus.classList.add(isDemo ? "demo" : (isAccelerated ? "ready" : "cpu"));
      if (isDemo) {
        $("span:last-child", elements.runtimeStatus).textContent = "Demo renderer · no checkpoints";
      } else if (isAccelerated) {
        $("span:last-child", elements.runtimeStatus).textContent = `${runtime.device_name} · ${runtime.dtype} · batch ${runtime.trajectory_batch_size}`;
      } else if (runtime?.status === "pytorch_cpu_build") {
        $("span:last-child", elements.runtimeStatus).textContent = "CPU-only PyTorch · GPU setup needed";
      } else if (runtime?.status === "cpu_requested") {
        $("span:last-child", elements.runtimeStatus).textContent = "CPU selected by configuration";
      } else if (["mps_not_built", "mps_unavailable"].includes(runtime?.status)) {
        $("span:last-child", elements.runtimeStatus).textContent = "Apple MPS unavailable · CPU";
      } else {
        $("span:last-child", elements.runtimeStatus).textContent = "CPU · no accelerator detected";
      }
      renderRuntimeDiagnostics(runtime);
      elements.cellLineSelect.replaceChildren();
      const unconditional = document.createElement("option");
      unconditional.value = "";
      unconditional.textContent = "Unconditional";
      elements.cellLineSelect.append(unconditional);
      state.config.cell_lines.forEach((name) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = name;
        elements.cellLineSelect.append(option);
      });
      await refreshRuns();
    } catch (error) {
      $("span:last-child", elements.runtimeStatus).textContent = "Backend unavailable";
      notify(error.message, 8000);
    }
  }

  init();
})();
