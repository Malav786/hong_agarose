/**
 * NanoStack 3D // Academic Research & VESTA Crystallography Engine
 * Powers high-fidelity VESTA 3D molecular inspection, full repository pagination,
 * real-time 60Hz streaming pipeline execution, and model telemetry diagnostics.
 */

// ================= GLOBAL APPLICATION STATE =================
let glViewer = null;
let currentCifText = "";
let currentStructureMetadata = null;
let stableCifsList = [];
let filteredCifsList = [];
let isSpinning = false;
let currentTab = "tab-control";
let rawAtomsData = [];

// Pagination State
let currentPage = 1;
let pageSize = 24;

// ================= DOM ELEMENT REFERENCES =================
// Navigation Tabs
const navTabs = document.querySelectorAll(".nav-tab");
const tabContents = document.querySelectorAll(".tab-content");
const tabPillCount = document.getElementById("tab-pill-count");

// Telemetry HUD Bar
const hudBilayers = document.getElementById("hud-bilayers");
const hudTrilayers = document.getElementById("hud-trilayers");
const hudStable = document.getElementById("hud-stable");
const hudMcc = document.getElementById("hud-mcc");
const hudRoc = document.getElementById("hud-roc");
const engineStatusDot = document.getElementById("engine-status-dot");
const engineStatusLabel = document.getElementById("engine-status-label");

// Mission Control / Parameters
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

// Execution Console
const terminalOutput = document.getElementById("terminal-output");
const pipelineProgressBar = document.getElementById("pipeline-progress-bar");
const pipelineProgressText = document.getElementById("pipeline-progress-text");
const btnClearTerminal = document.getElementById("btn-clear-terminal");
const btnCopyTerminal = document.getElementById("btn-copy-terminal");

// VESTA 3D Viewer Elements
const viewerStyle = document.getElementById("viewer-style");
const viewerColor = document.getElementById("viewer-color");
const btnViewC = document.getElementById("btn-view-c");
const btnViewA = document.getElementById("btn-view-a");
const btnViewB = document.getElementById("btn-view-b");
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
const specGap = document.getElementById("spec-gap");
const btnDownloadActiveCif = document.getElementById("btn-download-active-cif");

// Model Telemetry Elements
const featureBarsList = document.getElementById("feature-bars-list");

// Repository & Pagination Elements
const repoSearch = document.getElementById("repo-search");
const filterBtns = document.querySelectorAll(".filter-btn");
const repoCountBadge = document.getElementById("repo-count-badge");
const repoCardsGrid = document.getElementById("repo-cards-grid");
const paginationBar = document.getElementById("pagination-bar");
const paginationInfo = document.getElementById("pagination-info");
const paginationControls = document.getElementById("pagination-controls");
const pageSizeSelect = document.getElementById("page-size-select");
const btnBatchDownloadZip = document.getElementById("btn-batch-download-zip");

// ================= INITIALIZATION =================
document.addEventListener("DOMContentLoaded", () => {
    init3DViewer();
    initTabNavigation();
    initPipelineControls();
    initVestaControls();
    initRepository();

    // Sync initial workspace status & metrics
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

    // Resize 3D viewer when switching to structure inspector tab
    if (tabId === "tab-studio" && glViewer) {
        setTimeout(() => {
            glViewer.resize();
            glViewer.render();
        }, 120);
    }
}

// ================= PIPELINE CONTROLS =================
function initPipelineControls() {
    // Target Layer Stepper
    btnLayerDec.addEventListener("click", () => {
        let val = parseInt(elLayerCount.value, 10);
        if (val > 3) elLayerCount.value = val - 1;
    });
    btnLayerInc.addEventListener("click", () => {
        let val = parseInt(elLayerCount.value, 10);
        if (val < 8) elLayerCount.value = val + 1;
    });

    // CPU Worker Threads Slider
    elCores.addEventListener("input", (e) => {
        const val = e.target.value;
        elCoresBadge.textContent = `${val} Cores`;
        elCoresTip.textContent = `Allocating ${val} parallel workers (Numba JIT accelerated)`;
    });

    // Pipeline Action Buttons
    btnGenerate.addEventListener("click", () => triggerPipeline("generate"));
    btnExtract.addEventListener("click", () => triggerPipeline("extract"));
    btnPredict.addEventListener("click", () => triggerPipeline("predict"));
    btnFullPipeline.addEventListener("click", () => triggerPipeline("all"));

    // Console Action Buttons
    btnClearTerminal.addEventListener("click", () => {
        terminalOutput.innerHTML = "";
    });
    btnCopyTerminal.addEventListener("click", () => {
        navigator.clipboard.writeText(terminalOutput.innerText);
        logTerminal("[SYSTEM] Terminal log copied to clipboard.", "system");
    });
}

// ================= VESTA-STYLE 3D MOLECULAR INSPECTOR =================
function init3DViewer() {
    const container = document.getElementById("molecule-viewer");
    if (!container || typeof $3Dmol === "undefined") return;

    try {
        // VESTA Crisp White Background with Anti-Aliasing
        glViewer = $3Dmol.createViewer(container, {
            backgroundColor: "#ffffff",
            id: "molViewer",
            antialias: true
        });
        glViewer.render();
    } catch (e) {
        logTerminal(`[ERROR] 3Dmol viewer failed to initialize: ${e}`, "error");
    }
}

function initVestaControls() {
    viewerStyle.addEventListener("change", applyViewerStyle);
    viewerColor.addEventListener("change", applyViewerStyle);

    // VESTA Standard Crystallographic Projections
    if (btnViewC) {
        btnViewC.addEventListener("click", () => {
            if (!glViewer) return;
            // Top-down view along c* (Z-axis)
            glViewer.setCameraParameters({
                view: [0, 0, 1],
                up: [0, 1, 0]
            });
            glViewer.zoomTo();
            glViewer.render();
            logTerminal("[VESTA] Oriented view along c* axis (Top-down)", "info");
        });
    }

    if (btnViewA) {
        btnViewA.addEventListener("click", () => {
            if (!glViewer) return;
            // Side view along a* (X-axis)
            glViewer.setCameraParameters({
                view: [1, 0, 0],
                up: [0, 0, 1]
            });
            glViewer.zoomTo();
            glViewer.render();
            logTerminal("[VESTA] Oriented view along a* axis (Side projection)", "info");
        });
    }

    if (btnViewB) {
        btnViewB.addEventListener("click", () => {
            if (!glViewer) return;
            // Front view along b* (Y-axis)
            glViewer.setCameraParameters({
                view: [0, 1, 0],
                up: [0, 0, 1]
            });
            glViewer.zoomTo();
            glViewer.render();
            logTerminal("[VESTA] Oriented view along b* axis (Front projection)", "info");
        });
    }

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
            link.download = `${hudCifName.textContent || "nanostack_vesta_crystal"}.png`;
            link.href = imgUri;
            link.click();
            logTerminal("[VESTA] Snapshot exported successfully.", "system");
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
        logTerminal(`[VESTA] Loading crystal coordinates: ${metadata.filename}...`, "info");
        const res = await fetch(`/api/view_cif?path=${encodeURIComponent(metadata.file_path)}`);
        if (!res.ok) throw new Error(`HTTP error: ${res.status}`);

        currentCifText = await res.text();
        visualizerOverlay.style.display = "none";
        crystalHudBadge.style.display = "flex";

        // Update Metadata Badges
        hudCifName.textContent = metadata.filename;
        tagLayer.textContent = getLayerTitle(metadata.L);
        tagSide.textContent = getInterfaceLabel(metadata);
        tagRot.textContent = `Rot: ${metadata.rotation}°`;
        tagDisp.textContent = `Disp: ${metadata.displacement} Å`;

        // Update Crystallographic Specs
        const atomsPerLayer = 78;
        const totalAtoms = metadata.L * atomsPerLayer;
        specAtoms.textContent = `${totalAtoms} atoms (${metadata.L} × 78/layer)`;
        specFormula.textContent = `C${metadata.L * 24} H${metadata.L * 36} O${metadata.L * 18}`;
        if (specGap) {
            specGap.textContent = metadata.displacement > 0 ? `~${(metadata.displacement + 6.0).toFixed(2)} Å` : "~7.58 Å";
        }

        // Parse coordinates for exploded separation view
        parseCifAtoms(currentCifText, metadata.L);

        // Render in 3Dmol with VESTA CPK Ball & Stick
        if (glViewer) {
            glViewer.clear();
            glViewer.addModel(currentCifText, "cif");
            applyViewerStyle();
            glViewer.zoomTo();
            glViewer.render();
        }

        switchTab("tab-studio");
        logTerminal(`[VESTA] Rendered ${metadata.filename} in VESTA Ball & Stick style.`, "success");

    } catch (err) {
        logTerminal(`[ERROR] Failed to load CIF into VESTA: ${err}`, "error");
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

    // VESTA CPK Element Palette
    // Carbon: Charcoal/Dark Grey, Oxygen: Vivid Red, Hydrogen: Off-White
    const vestaCPK = {
        C: "#2b2b2b",
        O: "#dc2626",
        H: "#e2e8f0"
    };

    let baseStyle = {};
    if (style === "ball_and_stick") {
        // VESTA Signature Ball & Stick representation
        baseStyle = {
            stick: { radius: 0.16, color: "#64748b" },
            sphere: { scale: 0.28 }
        };
    } else if (style === "sphere") {
        // Space-filling van der Waals spheres
        baseStyle = { sphere: { scale: 0.36 } };
    } else if (style === "stick") {
        baseStyle = { stick: { radius: 0.22 } };
    } else if (style === "line") {
        baseStyle = { line: { linewidth: 2 } };
    }

    if (colorMode === "element") {
        // Apply VESTA standard CPK colors to each atom
        const model = glViewer.getModel();
        if (model) {
            const atoms = model.selectedAtoms({});
            for (let atom of atoms) {
                const elem = atom.elem || "C";
                const atomColor = vestaCPK[elem] || "#64748b";

                const atomStyle = Object.assign({}, baseStyle);
                if (atomStyle.sphere) {
                    atomStyle.sphere = Object.assign({}, atomStyle.sphere, { color: atomColor });
                }
                if (atomStyle.stick) {
                    atomStyle.stick = Object.assign({}, atomStyle.stick, { color: "#475569" });
                }
                glViewer.setStyle({ index: atom.index }, atomStyle);
            }
        } else {
            glViewer.setStyle({}, baseStyle);
        }
    } else if (colorMode === "layer") {
        // Color distinct layers based on Z coordinates
        const model = glViewer.getModel();
        if (model) {
            const atoms = model.selectedAtoms({});
            atoms.sort((a, b) => a.z - b.z);
            const total = atoms.length;
            const L = currentStructureMetadata ? currentStructureMetadata.L : 3;
            const chunkSize = Math.floor(total / L);

            const layerColors = ["#1d4ed8", "#059669", "#7c3aed", "#d97706", "#db2777"];

            for (let l = 0; l < L; l++) {
                const start = l * chunkSize;
                const end = (l === L - 1) ? total : (l + 1) * chunkSize;
                const col = layerColors[l % layerColors.length];
                for (let i = start; i < end; i++) {
                    const customStyle = Object.assign({}, baseStyle);
                    if (customStyle.sphere) customStyle.sphere.color = col;
                    if (customStyle.stick) customStyle.stick.color = col;
                    glViewer.setStyle({ index: atoms[i].index }, customStyle);
                }
            }
        }
    } else if (colorMode === "stability") {
        // Emerald for verified thermodynamically stable
        const customStyle = Object.assign({}, baseStyle);
        if (customStyle.sphere) customStyle.sphere.color = "#059669";
        if (customStyle.stick) customStyle.stick.color = "#047857";
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

    setEngineRunning(true);
    updateProgress(0, `STARTING (${mode.toUpperCase()})...`);
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

                const pctMatch = line.match(/(\d+)%\s+completed/);
                if (pctMatch) {
                    const pct = parseInt(pctMatch[1], 10);
                    updateProgress(pct, `PROCESSING (${pct}%)`);
                }

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
        engineStatusDot.className = "status-indicator-dot";
        engineStatusLabel.textContent = "ENGINE READY";
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
        filterRepository();

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
                <span class="bar-name">${feat.label} <small style="color:var(--text-muted); font-weight:normal;">(${feat.name}, cutoff: ${feat.cutoff})</small></span>
                <span class="bar-val">${(feat.importance * 100).toFixed(1)}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width: ${pct}%;"></div>
            </div>
        `;
        featureBarsList.appendChild(item);
    });
}

// ================= CRYSTAL REPOSITORY WITH FULL PAGINATION =================
function initRepository() {
    repoSearch.addEventListener("input", () => {
        currentPage = 1;
        filterRepository();
    });

    filterBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            filterBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentPage = 1;
            filterRepository();
        });
    });

    pageSizeSelect.addEventListener("change", (e) => {
        pageSize = parseInt(e.target.value, 10);
        currentPage = 1;
        renderPaginatedView();
    });

    // Batch ZIP Download Button
    if (btnBatchDownloadZip) {
        btnBatchDownloadZip.addEventListener("click", () => {
            const activeFilterBtn = document.querySelector(".filter-btn.active");
            const layerFilter = activeFilterBtn ? activeFilterBtn.getAttribute("data-filter") : "all";
            let layerNum = 3;
            if (layerFilter.startsWith("L=")) {
                layerNum = parseInt(layerFilter.replace("L=", ""), 10);
            }
            const url = `/api/download_zip?layer=${layerNum}`;
            window.open(url, "_blank");
            logTerminal(`[EXPORT] Initiated batch ZIP download for L=${layerNum} stable structures.`, "system");
        });
    }
}

function filterRepository() {
    const query = repoSearch.value.toLowerCase().trim();
    const activeFilterBtn = document.querySelector(".filter-btn.active");
    const layerFilter = activeFilterBtn ? activeFilterBtn.getAttribute("data-filter") : "all";

    filteredCifsList = stableCifsList.filter(cif => {
        // Layer filter
        if (layerFilter !== "all") {
            const targetL = parseInt(layerFilter.replace("L=", ""), 10);
            if (cif.L !== targetL) return false;
        }

        // Search text query
        if (!query) return true;
        const text = `${cif.filename} r${cif.rotation} t${cif.displacement} ${cif.kind} L=${cif.L}`.toLowerCase();
        return text.includes(query);
    });

    repoCountBadge.textContent = `${filteredCifsList.length} Structures Found`;
    renderPaginatedView();
}

function renderPaginatedView() {
    const total = filteredCifsList.length;
    const totalPages = Math.ceil(total / pageSize) || 1;

    if (currentPage > totalPages) currentPage = totalPages;
    if (currentPage < 1) currentPage = 1;

    const startIdx = (currentPage - 1) * pageSize;
    const endIdx = Math.min(startIdx + pageSize, total);
    const pageItems = filteredCifsList.slice(startIdx, endIdx);

    // Update Showing text
    if (total === 0) {
        paginationInfo.innerHTML = "Showing <strong>0</strong> structures";
    } else {
        paginationInfo.innerHTML = `Showing <strong>${startIdx + 1}–${endIdx}</strong> of <strong>${total}</strong> structures`;
    }

    // Render Cards
    repoCardsGrid.innerHTML = "";
    if (pageItems.length === 0) {
        repoCardsGrid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 50px; background: var(--bg-card); border-radius: var(--radius-md); border: 1px dashed var(--border-medium);">
                No crystal structures match your filter. Select another layer or run the prediction pipeline to generate new configurations.
            </div>
        `;
    } else {
        pageItems.forEach(cif => {
            const card = createCrystalCard(cif);
            repoCardsGrid.appendChild(card);
        });
    }

    // Render Pagination Controls
    renderPaginationControls(totalPages);
}

function createCrystalCard(cif) {
    const card = document.createElement("div");
    card.className = "crystal-card";

    const layerClass = `l${cif.L}`;
    const layerTitle = getLayerTitle(cif.L);
    const interfaceText = getInterfaceLabel(cif);

    card.innerHTML = `
        <div class="card-top">
            <span class="card-layer-pill ${layerClass}">${layerTitle}</span>
            <span class="card-stable-tag">&Delta;E &le; 0</span>
        </div>
        <div class="card-title" title="${cif.filename}">${cif.filename}</div>
        <div class="card-props">
            <span class="prop-chip">Twist: ${cif.rotation}&deg;</span>
            <span class="prop-chip">Disp: ${cif.displacement} &Aring;</span>
            <span class="prop-chip">${interfaceText}</span>
        </div>
        <div class="card-actions">
            <button class="card-action-btn view-btn">&micro; Inspect in VESTA</button>
            <button class="card-action-btn dl-btn" title="Download CIF">&#128190; CIF</button>
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

    return card;
}

function renderPaginationControls(totalPages) {
    paginationControls.innerHTML = "";
    if (totalPages <= 1) return;

    // Previous Button
    const prevBtn = document.createElement("button");
    prevBtn.className = "page-btn";
    prevBtn.innerHTML = "&laquo; Prev";
    prevBtn.disabled = currentPage === 1;
    prevBtn.addEventListener("click", () => {
        if (currentPage > 1) {
            currentPage--;
            renderPaginatedView();
            window.scrollTo({ top: repoCardsGrid.offsetTop - 100, behavior: "smooth" });
        }
    });
    paginationControls.appendChild(prevBtn);

    // Dynamic Page Number Buttons (smart ellipsis)
    const pagesToShow = getPageNumbers(currentPage, totalPages);
    pagesToShow.forEach(p => {
        if (p === "...") {
            const ellipsis = document.createElement("span");
            ellipsis.textContent = "...";
            ellipsis.style.padding = "0 6px";
            ellipsis.style.color = "var(--text-muted)";
            paginationControls.appendChild(ellipsis);
        } else {
            const pageBtn = document.createElement("button");
            pageBtn.className = `page-btn ${p === currentPage ? "active" : ""}`;
            pageBtn.textContent = p;
            pageBtn.addEventListener("click", () => {
                currentPage = p;
                renderPaginatedView();
                window.scrollTo({ top: repoCardsGrid.offsetTop - 100, behavior: "smooth" });
            });
            paginationControls.appendChild(pageBtn);
        }
    });

    // Next Button
    const nextBtn = document.createElement("button");
    nextBtn.className = "page-btn";
    nextBtn.innerHTML = "Next &raquo;";
    nextBtn.disabled = currentPage === totalPages;
    nextBtn.addEventListener("click", () => {
        if (currentPage < totalPages) {
            currentPage++;
            renderPaginatedView();
            window.scrollTo({ top: repoCardsGrid.offsetTop - 100, behavior: "smooth" });
        }
    });
    paginationControls.appendChild(nextBtn);
}

function getPageNumbers(current, total) {
    if (total <= 7) {
        return Array.from({ length: total }, (_, i) => i + 1);
    }
    if (current <= 4) {
        return [1, 2, 3, 4, 5, "...", total];
    }
    if (current >= total - 3) {
        return [1, "...", total - 4, total - 3, total - 2, total - 1, total];
    }
    return [1, "...", current - 1, current, current + 1, "...", total];
}

function getLayerTitle(L) {
    if (L === 2) return "L=2 Bilayer";
    if (L === 3) return "L=3 Trilayer";
    if (L === 4) return "L=4 Tetralayer";
    if (L === 5) return "L=5 Pentalayer";
    return `L=${L} Multilayer`;
}

function getInterfaceLabel(cif) {
    if (cif.L === 2) return "Bilayer Parent Seed";
    if (cif.kind === "lower") return "Lower Stacking (Bottom)";
    return "Upper Stacking (Top)";
}
