/**
 * NanoStack 3D // Control Plane & Telemetry Observatory JavaScript Engine
 * Powers interactive 3D crystal studio, live 60Hz streaming terminal,
 * dual-mode switching (Tech vs ELI5), and model telemetry diagnostics.
 */

// ================= GLOBAL APPLICATION STATE =================
let glViewer = null;
let currentCifText = "";
let currentStructureMetadata = null;
let stableCifsList = [];
let isSpinning = false;
let currentTab = "tab-control";
let currentMode = "tech"; // 'tech' | 'story'
let rawAtomsData = [];

// ================= DOM ELEMENT REFERENCES =================
// Navigation & Mode
const btnTechMode = document.getElementById("btn-tech-mode");
const btnStoryMode = document.getElementById("btn-story-mode");
const modeBadgeText = document.getElementById("mode-badge-text");
const navTabs = document.querySelectorAll(".nav-tab");
const tabContents = document.querySelectorAll(".tab-content");
const tabPillCount = document.getElementById("tab-pill-count");

// Telemetry HUD Bar
const hudBilayers = document.getElementById("hud-bilayers");
const hudTrilayers = document.getElementById("hud-trilayers");
const hudStable = document.getElementById("hud-stable");
const hudMcc = document.getElementById("hud-mcc");
const hudRoc = document.getElementById("hud-roc");
const hudThreshold = document.getElementById("hud-threshold");
const engineStatusDot = document.getElementById("engine-status-dot");
const engineStatusLabel = document.getElementById("engine-status-label");

// Mission Control / Job Dispatch
const elLayerCount = document.getElementById("layer-count");
const btnLayerDec = document.getElementById("btn-layer-dec");
const btnLayerInc = document.getElementById("btn-layer-inc");
const elCores = document.getElementById("cores");
const elCoresBadge = document.getElementById("cores-badge");
const elCoresTip = document.getElementById("cores-tip");
const btnGenerate = document.getElementById("btn-generate");
const btnExtract = document.getElementById("btn-extract");
const btnPredict = document.getElementById("btn-predict");
const btnFullPipeline = document.getElementById("btn-full-pipeline");

// Terminal Output
const terminalOutput = document.getElementById("terminal-output");
const pipelineProgressBar = document.getElementById("pipeline-progress-bar");
const pipelineProgressText = document.getElementById("pipeline-progress-text");
const btnClearTerminal = document.getElementById("btn-clear-terminal");
const btnCopyTerminal = document.getElementById("btn-copy-terminal");

// 3D Studio Elements
const viewerStyle = document.getElementById("viewer-style");
const viewerColor = document.getElementById("viewer-color");
const btnSpin = document.getElementById("btn-spin");
const btnResetCam = document.getElementById("btn-reset-cam");
const btnScreenshot = document.getElementById("btn-screenshot");
const visualizerOverlay = document.getElementById("visualizer-overlay");
const crystalHudBadge = document.getElementById("crystal-hud-badge");
const hudCifName = document.getElementById("hud-cif-name");
const tagLayer = document.getElementById("tag-layer");
const tagSide = document.getElementById("tag-side");
const tagRot = document.getElementById("tag-rot");
const tagDisp = document.getElementById("tag-disp");
const explodeSlider = document.getElementById("explode-slider");
const explodeVal = document.getElementById("explode-val");
const specFormula = document.getElementById("spec-formula");
const specAtoms = document.getElementById("spec-atoms");
const btnDownloadActiveCif = document.getElementById("btn-download-active-cif");
const storyExplanationBox = document.getElementById("story-explanation-box");
const storyText = document.getElementById("story-text");

// Model Telemetry Elements
const featureBarsList = document.getElementById("feature-bars-list");

// Repository Elements
const repoSearch = document.getElementById("repo-search");
const filterBtns = document.querySelectorAll(".filter-btn");
const repoCountBadge = document.getElementById("repo-count-badge");
const repoCardsGrid = document.getElementById("repo-cards-grid");

// ================= INITIALIZATION =================
document.addEventListener("DOMContentLoaded", () => {
    init3DViewer();
    initTabNavigation();
    initModeSwitcher();
    initPipelineControls();
    initStudioControls();
    initRepositoryFilters();
    
    // Initial workspace status sync & metrics fetch
    updateWorkspaceStatus();
    fetchModelMetrics();
});

// ================= TAB NAVIGATION =================
function initTabNavigation() {
    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const target = tab.getAttribute("data-tab");
            switchTab(target);
        });
    });
}

function switchTab(tabId) {
    currentTab = tabId;
    navTabs.forEach(t => {
        if (t.getAttribute("data-tab") === tabId) {
            t.classList.add("active");
        } else {
            t.classList.remove("active");
        }
    });

    tabContents.forEach(content => {
        if (content.id === tabId) {
            content.classList.add("active");
        } else {
            content.classList.remove("active");
        }
    });

    // Resize 3D viewer when switching to studio tab
    if (tabId === "tab-studio" && glViewer) {
        setTimeout(() => {
            glViewer.resize();
            glViewer.render();
        }, 100);
    }
}

// ================= DUAL-MODE SWITCHER (Tech vs ELI5) =================
function initModeSwitcher() {
    btnTechMode.addEventListener("click", () => setMode("tech"));
    btnStoryMode.addEventListener("click", () => setMode("story"));
}

function setMode(mode) {
    currentMode = mode;
    if (mode === "tech") {
        btnTechMode.classList.add("active");
        btnStoryMode.classList.remove("active");
        document.body.classList.remove("story-mode");
        document.body.classList.add("tech-mode");
        modeBadgeText.textContent = "MODELED ESTIMATE // PBC-STRICT";
        storyExplanationBox.style.display = "none";
        logTerminal("[SYSTEM] Switched to ⚡ Tech Mode: High-precision physics & telemetry active.", "system");
    } else {
        btnStoryMode.classList.add("active");
        btnTechMode.classList.remove("active");
        document.body.classList.remove("tech-mode");
        document.body.classList.add("story-mode");
        modeBadgeText.textContent = "👶 MATERIALS STORY MODE (ELI5)";
        storyExplanationBox.style.display = "block";
        logTerminal("[STORY] Switched to 👶 Materials Story Mode: Intuitive layer-stacking concepts enabled.", "info");
    }
}

// ================= PIPELINE CONTROLS =================
function initPipelineControls() {
    // Layer Count Stepper
    btnLayerDec.addEventListener("click", () => {
        let val = parseInt(elLayerCount.value, 10);
        if (val > 3) elLayerCount.value = val - 1;
    });
    btnLayerInc.addEventListener("click", () => {
        let val = parseInt(elLayerCount.value, 10);
        if (val < 8) elLayerCount.value = val + 1;
    });

    // Worker Threads Slider
    elCores.addEventListener("input", (e) => {
        const val = e.target.value;
        elCoresBadge.textContent = `${val} Cores`;
        elCoresTip.textContent = `Allocating ${val} parallel workers (Numba JIT accelerated)`;
    });

    // Execution Buttons
    btnGenerate.addEventListener("click", () => triggerPipeline("generate"));
    btnExtract.addEventListener("click", () => triggerPipeline("extract"));
    btnPredict.addEventListener("click", () => triggerPipeline("predict"));
    btnFullPipeline.addEventListener("click", () => triggerPipeline("all"));

    // Terminal Buttons
    btnClearTerminal.addEventListener("click", () => {
        terminalOutput.innerHTML = "";
    });
    btnCopyTerminal.addEventListener("click", () => {
        const text = terminalOutput.innerText;
        navigator.clipboard.writeText(text);
        logTerminal("[SYSTEM] Terminal log copied to clipboard.", "system");
    });
}

// ================= 3D CRYSTAL STUDIO =================
function init3DViewer() {
    const container = document.getElementById("molecule-viewer");
    if (!container || typeof $3Dmol === "undefined") return;

    try {
        glViewer = $3Dmol.createViewer(container, {
            backgroundColor: "#020408",
            id: "molViewer"
        });
        glViewer.render();
    } catch (e) {
        logTerminal(`[ERROR] 3Dmol viewer failed to initialize: ${e}`, "error");
    }
}

function initStudioControls() {
    viewerStyle.addEventListener("change", applyViewerStyle);
    viewerColor.addEventListener("change", applyViewerStyle);
    
    btnSpin.addEventListener("click", () => {
        isSpinning = !isSpinning;
        btnSpin.classList.toggle("active", isSpinning);
        if (glViewer) {
            glViewer.spin(isSpinning ? "y" : false);
        }
    });

    btnResetCam.addEventListener("click", () => {
        if (glViewer) {
            glViewer.zoomTo();
            glViewer.render();
        }
    });

    btnScreenshot.addEventListener("click", () => {
        if (glViewer) {
            const imgUri = glViewer.pngURI();
            const link = document.createElement("a");
            link.download = `${hudCifName.textContent || "nanostack_crystal"}.png`;
            link.href = imgUri;
            link.click();
            logTerminal("[SYSTEM] Snapshot exported successfully.", "system");
        }
    });

    explodeSlider.addEventListener("input", (e) => {
        const shift = parseFloat(e.target.value);
        explodeVal.textContent = `+${shift} Å`;
        applyExplodeSeparation(shift);
    });

    btnDownloadActiveCif.addEventListener("click", () => {
        if (currentStructureMetadata && currentStructureMetadata.file_path) {
            const url = `/api/download_cif?path=${encodeURIComponent(currentStructureMetadata.file_path)}`;
            window.open(url, "_blank");
        }
    });
}

async function loadCifIntoStudio(metadata) {
    if (!metadata || !metadata.file_path) return;
    currentStructureMetadata = metadata;

    try {
        logTerminal(`[STUDIO] Streaming CIF coordinates: ${metadata.filename}...`, "info");
        const res = await fetch(`/api/view_cif?path=${encodeURIComponent(metadata.file_path)}`);
        if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
        
        currentCifText = await res.text();
        visualizerOverlay.style.display = "none";
        crystalHudBadge.style.display = "block";

        // Update Metadata Badges
        hudCifName.textContent = metadata.filename;
        tagLayer.textContent = `L=${metadata.L} Layer`;
        tagSide.textContent = metadata.kind === "lower" ? "Bottom Stacking" : "Top Stacking";
        tagRot.textContent = `Rotation: ${metadata.rotation}°`;
        tagDisp.textContent = `Displacement: ${metadata.displacement} Å`;

        // Update Spec Table
        specAtoms.textContent = `${metadata.L * 78} atoms (${metadata.L} × 78/layer)`;
        specFormula.textContent = `C${metadata.L * 24} H${metadata.L * 36} O${metadata.L * 18} (Stacked)`;

        // Parse atoms coordinates for exploded view and rendering
        parseCifAtoms(currentCifText, metadata.L);

        // Render in 3Dmol
        if (glViewer) {
            glViewer.clear();
            glViewer.addModel(currentCifText, "cif");
            applyViewerStyle();
            glViewer.zoomTo();
            glViewer.render();
        }

        switchTab("tab-studio");
        logTerminal(`[STUDIO] Rendered ${metadata.filename} successfully in 3D viewport.`, "success");

    } catch (err) {
        logTerminal(`[ERROR] Failed to load CIF: ${err}`, "error");
    }
}

function parseCifAtoms(cifText, L) {
    rawAtomsData = [];
    const lines = cifText.split("\n");
    let inLoop = false;
    
    for (let line of lines) {
        line = line.trim();
        if (line.startsWith("_atom_site_")) {
            inLoop = true;
            continue;
        }
        if (inLoop && line.length > 0 && !line.startsWith("_") && !line.startsWith("loop_")) {
            const tokens = line.split(/\s+/);
            if (tokens.length >= 6) {
                const elem = tokens[0];
                const x = parseFloat(tokens[3]);
                const y = parseFloat(tokens[4]);
                const z = parseFloat(tokens[5]);
                if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
                    rawAtomsData.push({ elem, x, y, z });
                }
            }
        }
    }
}

function applyViewerStyle() {
    if (!glViewer) return;
    const style = viewerStyle.value;
    const colorMode = viewerColor.value;

    let styleObj = {};
    if (style === "stick") {
        styleObj = { stick: { radius: 0.22 } };
    } else if (style === "sphere") {
        styleObj = { sphere: { scale: 0.35 } };
    } else if (style === "ball_and_stick") {
        styleObj = { stick: { radius: 0.18 }, sphere: { scale: 0.28 } };
    } else if (style === "line") {
        styleObj = { line: { linewidth: 2 } };
    }

    // Color schemes
    if (colorMode === "element") {
        // CPK default
        glViewer.setStyle({}, styleObj);
    } else if (colorMode === "layer") {
        // Color distinct layers based on Z coordinates
        const model = glViewer.getModel();
        if (model) {
            const atoms = model.selectedAtoms({});
            atoms.sort((a, b) => a.z - b.z);
            const total = atoms.length;
            const L = currentStructureMetadata ? currentStructureMetadata.L : 3;
            const chunkSize = Math.floor(total / L);

            const layerColors = ["#00f2fe", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899"];

            for (let l = 0; l < L; l++) {
                const start = l * chunkSize;
                const end = (l === L - 1) ? total : (l + 1) * chunkSize;
                const col = layerColors[l % layerColors.length];
                for (let i = start; i < end; i++) {
                    const customStyle = Object.assign({}, styleObj);
                    for (let k in customStyle) {
                        customStyle[k].color = col;
                    }
                    glViewer.setStyle({ index: atoms[i].index }, customStyle);
                }
            }
        }
    } else if (colorMode === "stability") {
        // Uniform thermal emerald for stable
        const customStyle = Object.assign({}, styleObj);
        for (let k in customStyle) customStyle[k].color = "#34d399";
        glViewer.setStyle({}, customStyle);
    }

    glViewer.render();
}

function applyExplodeSeparation(shiftZ) {
    if (!glViewer || rawAtomsData.length === 0) return;
    const model = glViewer.getModel();
    if (!model) return;

    const atoms = model.selectedAtoms({});
    atoms.sort((a, b) => a.z - b.z);
    const total = atoms.length;
    const L = currentStructureMetadata ? currentStructureMetadata.L : 3;
    const chunkSize = Math.floor(total / L);

    for (let l = 0; l < L; l++) {
        const start = l * chunkSize;
        const end = (l === L - 1) ? total : (l + 1) * chunkSize;
        const layerOffset = (l - (L - 1) / 2) * shiftZ;

        for (let i = start; i < end; i++) {
            const orig = rawAtomsData[atoms[i].index];
            if (orig) {
                atoms[i].z = orig.z + layerOffset;
            }
        }
    }

    glViewer.render();
}

// ================= STREAMING PIPELINE EXECUTION =================
async function triggerPipeline(mode) {
    const L = parseInt(elLayerCount.value, 10);
    const cores = parseInt(elCores.value, 10);

    const angleCheckboxes = document.querySelectorAll("#angle-matrix input:checked");
    const angles = Array.from(angleCheckboxes).map(cb => parseInt(cb.value, 10));

    if (angles.length === 0) {
        logTerminal("[ERROR] Please select at least one rotation angle.", "error");
        return;
    }

    // Set Engine Status to Running
    setEngineRunning(true);
    updateProgress(0, `DISPATCHING (${mode.toUpperCase()})...`);
    logTerminal(`[SYSTEM] Dispatched Job: Mode=${mode.toUpperCase()}, Target L=${L}, Workers=${cores}, Angles=[${angles.join(", ")}]`, "system");

    try {
        const response = await fetch("/api/run_pipeline", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ L, mode, cores, angles })
        });

        if (!response.ok) {
            throw new Error(`Pipeline API returned error: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split("\n");

            for (const line of lines) {
                if (!line.trim()) continue;

                // Parse progress percentages
                const pctMatch = line.match(/(\d+)%\s+completed/);
                if (pctMatch) {
                    const pct = parseInt(pctMatch[1], 10);
                    updateProgress(pct, `PROCESSING (${pct}%)`);
                }

                // Colorize log entries
                let type = "info";
                if (line.includes("[SYSTEM]")) type = "system";
                else if (line.includes("Error") || line.includes("[ERROR]")) type = "error";
                else if (line.includes("complete") || line.includes("Finished") || line.includes("SUCCESS")) type = "success";
                else if (line.includes("Class 0") || line.includes("Predict")) type = "warn";

                logTerminal(line, type);
            }
        }

        updateProgress(100, "COMPLETED (100%)");
        logTerminal("[SYSTEM] Pipeline execution finished successfully.", "success");
        await updateWorkspaceStatus();

    } catch (err) {
        logTerminal(`[ERROR] Execution stream crashed: ${err}`, "error");
        updateProgress(0, "FAILED");
    } finally {
        setEngineRunning(false);
    }
}

function setEngineRunning(running) {
    if (running) {
        engineStatusDot.className = "status-indicator-dot running";
        engineStatusLabel.textContent = "ENGINE BUSY";
        btnGenerate.disabled = true;
        btnExtract.disabled = true;
        btnPredict.disabled = true;
        btnFullPipeline.disabled = true;
    } else {
        engineStatusDot.className = "status-indicator-dot idle";
        engineStatusLabel.textContent = "ENGINE IDLE";
        btnGenerate.disabled = false;
        btnExtract.disabled = false;
        btnPredict.disabled = false;
        btnFullPipeline.disabled = false;
    }
}

function updateProgress(percent, label) {
    pipelineProgressBar.style.width = `${percent}%`;
    pipelineProgressText.textContent = label;
}

function logTerminal(message, type = "info") {
    const entry = document.createElement("div");
    entry.className = `log-entry ${type}`;
    
    const time = new Date().toLocaleTimeString();
    entry.innerHTML = `<span class="timestamp">[${time}]</span> ${escapeHtml(message)}`;
    
    terminalOutput.appendChild(entry);
    terminalOutput.scrollTop = terminalOutput.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

// ================= WORKSPACE STATUS SYNC =================
async function updateWorkspaceStatus() {
    try {
        const res = await fetch("/api/status");
        if (!res.ok) return;
        const data = await res.json();

        hudBilayers.textContent = data.bilayers_count.toLocaleString();
        hudTrilayers.textContent = data.trilayers_count.toLocaleString();
        hudStable.textContent = data.stable_cifs_count.toLocaleString();
        tabPillCount.textContent = data.stable_cifs_count;

        stableCifsList = data.stable_cifs || [];
        renderRepositoryCards(stableCifsList);

    } catch (err) {
        logTerminal(`[WARN] Status sync error: ${err}`, "warn");
    }
}

// ================= MODEL METRICS FETCH =================
async function fetchModelMetrics() {
    try {
        const res = await fetch("/api/metrics");
        if (!res.ok) return;
        const metrics = await res.json();

        hudMcc.textContent = metrics.mcc.toFixed(4);
        hudRoc.textContent = metrics.roc_auc.toFixed(4);
        hudThreshold.textContent = metrics.threshold.toFixed(4);

        renderFeatureImportanceBars(metrics.feature_importances || []);

    } catch (err) {
        logTerminal(`[WARN] Model metrics sync failed: ${err}`, "warn");
    }
}

function renderFeatureImportanceBars(features) {
    featureBarsList.innerHTML = "";
    const maxImp = Math.max(...features.map(f => f.importance), 0.3);

    features.forEach(feat => {
        const pct = ((feat.importance / maxImp) * 100).toFixed(1);
        const item = document.createElement("div");
        item.className = "feature-bar-item";
        item.innerHTML = `
            <div class="bar-meta">
                <span class="bar-name">${feat.label} <small style="color:var(--text-muted);">(${feat.name}, cutoff: ${feat.cutoff})</small></span>
                <span class="bar-val">${(feat.importance * 100).toFixed(1)}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width: ${pct}%;"></div>
            </div>
        `;
        featureBarsList.appendChild(item);
    });
}

// ================= CRYSTAL REPOSITORY CARDS =================
function initRepositoryFilters() {
    repoSearch.addEventListener("input", filterRepository);
    
    filterBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            filterBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            filterRepository();
        });
    });
}

function filterRepository() {
    const query = repoSearch.value.toLowerCase().trim();
    const activeFilterBtn = document.querySelector(".filter-btn.active");
    const layerFilter = activeFilterBtn ? activeFilterBtn.getAttribute("data-filter") : "all";

    const filtered = stableCifsList.filter(cif => {
        // Layer filter
        if (layerFilter !== "all") {
            const targetL = parseInt(layerFilter.replace("L=", ""), 10);
            if (cif.L !== targetL) return false;
        }

        // Text query
        if (!query) return true;
        const text = `${cif.filename} r${cif.rotation} t${cif.displacement} ${cif.kind} L=${cif.L}`.toLowerCase();
        return text.includes(query);
    });

    renderRepositoryCards(filtered);
}

function renderRepositoryCards(cifs) {
    repoCardsGrid.innerHTML = "";
    repoCountBadge.textContent = `${cifs.length} Structures Discovered`;

    if (cifs.length === 0) {
        repoCardsGrid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 50px;">
                No stable crystal structures match your filter. Run the prediction pipeline to discover stable configurations.
            </div>
        `;
        return;
    }

    cifs.slice(0, 100).forEach(cif => {
        const card = document.createElement("div");
        card.className = "crystal-card";
        card.innerHTML = `
            <div class="card-top">
                <span class="card-layer-pill">L=${cif.L} Monolayers</span>
                <span class="prop-chip" style="color:var(--emerald-glow);">ΔE ≤ 0</span>
            </div>
            <div class="card-title" title="${cif.filename}">${cif.filename}</div>
            <div class="card-props">
                <span class="prop-chip">Rot: ${cif.rotation}°</span>
                <span class="prop-chip">Disp: ${cif.displacement} Å</span>
                <span class="prop-chip">${cif.kind === "lower" ? "Bottom" : "Top"} Stack</span>
            </div>
            <div class="card-actions">
                <button class="card-action-btn view-btn">⚡ 3D Studio</button>
                <button class="card-action-btn dl-btn">💾 CIF</button>
            </div>
        `;

        card.querySelector(".view-btn").addEventListener("click", (e) => {
            e.stopPropagation();
            loadCifIntoStudio(cif);
        });

        card.querySelector(".dl-btn").addEventListener("click", (e) => {
            e.stopPropagation();
            const url = `/api/download_cif?path=${encodeURIComponent(cif.file_path)}`;
            window.open(url, "_blank");
        });

        card.addEventListener("click", () => {
            loadCifIntoStudio(cif);
        });

        repoCardsGrid.appendChild(card);
    });
}
