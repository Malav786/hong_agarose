// Global Error Catcher - Writes directly to the browser console UI for easy debugging
window.onerror = function(message, source, lineno, colno, error) {
    const output = document.getElementById("console-output");
    if (output) {
        const span = document.createElement("span");
        span.className = "error-log";
        span.textContent = `[UI RUNTIME ERROR] ${message} (${source}:${lineno}:${colno})`;
        output.appendChild(span);
        output.scrollTop = output.scrollHeight;
    }
    return false;
};

// Core Application State
let glViewer = null;
let currentCifText = "";
let stableCifsList = [];

// DOM Elements
const elLayerCount = document.getElementById("layer-count");
const elCores = document.getElementById("cores");
const elCoresVal = document.getElementById("cores-val");
const elStatusIndicator = document.getElementById("status-indicator");
const elStatusText = document.getElementById("status-text");
const elConsoleOutput = document.getElementById("console-output");
const elPipelineProgress = document.getElementById("pipeline-progress");
const elViewerStyle = document.getElementById("viewer-style");
const elBtnSpin = document.getElementById("btn-spin");
const elVisualizerOverlay = document.getElementById("visualizer-overlay");
const elViewDetails = document.getElementById("view-details");
const elTagKind = document.getElementById("tag-kind");
const elTagRotation = document.getElementById("tag-rotation");
const elTagDisp = document.getElementById("tag-disp");
const elTagFilename = document.getElementById("tag-filename");
const elExplorerSearch = document.getElementById("explorer-search");
const elExplorerGrid = document.getElementById("explorer-grid");
const elCifCount = document.getElementById("cif-count");

const elStatBilayers = document.getElementById("stat-bilayers");
const elStatTrilayers = document.getElementById("stat-trilayers");
const elStatStable = document.getElementById("stat-stable");

const elBtnGenerate = document.getElementById("btn-generate");
const elBtnExtract = document.getElementById("btn-extract");
const elBtnPredict = document.getElementById("btn-predict");
const elBtnFullPipeline = document.getElementById("btn-full-pipeline");
const elBtnClearConsole = document.getElementById("btn-clear-console");

// Initialize Event Listeners
document.addEventListener("DOMContentLoaded", () => {
    // 3Dmol viewer init
    init3DViewer();
    
    // UI Event Listeners
    elCores.addEventListener("input", (e) => {
        elCoresVal.textContent = `${e.target.value} Cores`;
    });
    
    elViewerStyle.addEventListener("change", updateViewerStyle);
    elBtnSpin.addEventListener("click", toggleSpin);
    elBtnClearConsole.addEventListener("click", () => elConsoleOutput.innerHTML = "");
    
    elExplorerSearch.addEventListener("input", filterCifsGrid);
    
    // Pipeline control buttons
    elBtnGenerate.addEventListener("click", () => triggerPipeline("generate"));
    elBtnExtract.addEventListener("click", () => triggerPipeline("extract"));
    elBtnPredict.addEventListener("click", () => triggerPipeline("predict"));
    elBtnFullPipeline.addEventListener("click", () => triggerPipeline("all"));
    
    // Fetch initial stats & list
    updateStats();
});

// ================= STATS & WORKSPACE SYNC =================
async function updateStats() {
    try {
        const res = await fetch("/api/status");
        const status = await res.json();
        
        elStatBilayers.textContent = status.bilayers_count;
        elStatTrilayers.textContent = status.trilayers_count;
        elStatStable.textContent = status.stable_cifs_count;
        
        stableCifsList = status.stable_cifs;
        renderCifsGrid(stableCifsList);
    } catch (err) {
        logMessage(`[SYSTEM] Error updating stats: ${err}`, "error");
    }
}

// ================= STABLE CIF EXPLORER =================
function renderCifsGrid(cifs) {
    elExplorerGrid.innerHTML = "";
    elCifCount.textContent = `${cifs.length} Structures`;
    
    if (cifs.length === 0) {
        elExplorerGrid.innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; color: var(--text-muted); padding: 40px;">
                No stable CIF files found. Run the Pipeline prediction to classify stable stacks.
            </div>`;
        return;
    }
    
    cifs.forEach(cif => {
        const card = document.createElement("div");
        card.className = "structure-card";
        card.innerHTML = `
            <div class="card-header">
                <span class="card-type">${cif.kind}</span>
            </div>
            <div class="card-name">${cif.filename}</div>
            <div class="card-meta">
                <span class="meta-tag">r${cif.rotation}</span>
                <span class="meta-tag">t${cif.displacement}</span>
                <span class="meta-tag">${cif.angle}°</span>
            </div>
        `;
        
        card.addEventListener("click", () => selectStructure(cif, card));
        elExplorerGrid.appendChild(card);
    });
}

function filterCifsGrid() {
    const q = elExplorerSearch.value.toLowerCase().trim();
    if (!q) {
        renderCifsGrid(stableCifsList);
        return;
    }
    
    const filtered = stableCifsList.filter(cif => 
        cif.filename.toLowerCase().includes(q) ||
        `r${cif.rotation}`.includes(q) ||
        `t${cif.displacement}`.includes(q) ||
        `${cif.angle}`.includes(q) ||
        cif.kind.includes(q)
    );
    renderCifsGrid(filtered);
}

// ================= 3D MOLECULAR VISUALIZER =================
function init3DViewer() {
    try {
        const container = document.getElementById("molecule-viewer");
        if (window.$3Dmol) {
            glViewer = $3Dmol.createViewer(container, { backgroundColor: "#05070c" });
        } else {
            console.warn("3Dmol.js not loaded yet. Retrying in 1 second...");
            setTimeout(init3DViewer, 1000);
        }
    } catch (err) {
        console.error("Failed to initialize 3D viewer:", err);
    }
}

async function selectStructure(cif, cardElement) {
    // Highlight Selected Card
    document.querySelectorAll(".structure-card").forEach(el => el.classList.remove("selected"));
    cardElement.classList.add("selected");
    
    // Hide overlay & show metadata tags
    elVisualizerOverlay.style.opacity = 0;
    elViewDetails.style.display = "flex";
    
    elTagKind.textContent = cif.kind;
    elTagRotation.textContent = `r${cif.rotation}`;
    elTagDisp.textContent = `t${cif.displacement}`;
    elTagFilename.textContent = cif.filename;
    
    logMessage(`[SYSTEM] Loading 3D model for ${cif.filename}...`, "info");
    
    try {
        const queryParams = new URLSearchParams({ path: cif.file_path });
        const res = await fetch(`/api/view_cif?${queryParams}`);
        currentCifText = await res.text();
        
        renderMolecule();
    } catch (err) {
        logMessage(`[SYSTEM] Error loading CIF content: ${err}`, "error");
    }
}

function renderMolecule() {
    if (!glViewer || !currentCifText) return;
    
    glViewer.clear();
    const model = glViewer.addModel(currentCifText, "cif");
    
    // Diagnostic logging to verify successful CIF parsing
    try {
        const atomCount = model.getAtoms().length;
        logMessage(`[SYSTEM] 3Dmol parsed ${atomCount} atoms successfully.`, "success");
    } catch (e) {
        logMessage(`[SYSTEM] Error parsing atoms: ${e}`, "error");
    }
    
    const style = elViewerStyle.value;
    const styleObj = {};
    
    if (style === "stick") {
        styleObj.stick = { radius: 0.18, colorscheme: "Jmol" };
    } else if (style === "sphere") {
        styleObj.sphere = { radius: 0.5, colorscheme: "Jmol" };
    } else {
        styleObj.line = { colorscheme: "Jmol" };
    }
    
    glViewer.setStyle({}, styleObj);
    
    // Ensure WebGL canvas fits container dimensions
    glViewer.resize();
    
    // Zoom onto the atoms first to focus on the structure
    glViewer.zoomTo();
    
    // Add visual unit cell box (wireframe) to outline the enlarged cell boundaries
    try {
        glViewer.addUnitCell(model, { box: { color: "#ffffff", opacity: 0.5 } });
    } catch (e) {
        console.warn("Failed to render unit cell outline:", e);
    }
    
    glViewer.render();
}

function updateViewerStyle() {
    renderMolecule();
}

let isSpinning = false;
function toggleSpin() {
    if (!glViewer) return;
    isSpinning = !isSpinning;
    if (isSpinning) {
        glViewer.animate({ loop: "yes", step: 0.4 });
        elBtnSpin.textContent = "Stop Spin";
    } else {
        glViewer.stopAnimate();
        elBtnSpin.textContent = "Toggle Spin";
    }
}

// ================= PIPELINE CONTROLLERS (LOG STREAMING) =================
async function triggerPipeline(mode) {
    // 1) Fetch inputs
    const L = parseInt(elLayerCount.value);
    const cores = parseInt(elCores.value);
    
    const checkedChips = document.querySelectorAll(".angle-chips input[type='checkbox']:checked");
    const angles = Array.from(checkedChips).map(el => parseInt(el.value));
    
    if (angles.length === 0) {
        alert("Please select at least one rotation angle.");
        return;
    }
    
    // Disable inputs & update status
    setPipelineRunning(true, mode);
    
    const params = { L, mode, cores, angles };
    
    try {
        const response = await fetch("/api/run_pipeline", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params)
        });
        
        // 2) Stream Chunk logs in real time
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            
            // Keep last partial line in buffer
            buffer = lines.pop();
            
            for (const line of lines) {
                if (line.trim()) {
                    processLogLine(line);
                }
            }
        }
        
        if (buffer.trim()) {
            processLogLine(buffer);
        }
        
        logMessage(`[SYSTEM] Stage "${mode.toUpperCase()}" execution complete.`, "success");
    } catch (err) {
        logMessage(`[SYSTEM] Error executing pipeline: ${err}`, "error");
    } finally {
        setPipelineRunning(false);
        updateStats();
    }
}

function processLogLine(line) {
    // Determine progress from logs
    // e.g. "Generating Trilayers in Parallel:  50%"
    const m = line.match(/(\d+)%/);
    if (m) {
        const pct = parseInt(m[1]);
        elPipelineProgress.style.width = `${pct}%`;
    }
    
    if (line.includes("Error") || line.includes("Failed")) {
        logMessage(line, "error");
    } else if (line.includes("complete") || line.includes("success") || line.includes("Finished")) {
        logMessage(line, "success");
    } else if (line.startsWith("[") || line.startsWith("Total") || line.startsWith("Initializing")) {
        logMessage(line, "system");
    } else {
        logMessage(line, "info");
    }
}

function setPipelineRunning(isRunning, mode = "") {
    const inputs = [elLayerCount, elCores, elViewerStyle, elBtnGenerate, elBtnExtract, elBtnPredict, elBtnFullPipeline];
    inputs.forEach(el => el.disabled = isRunning);
    
    document.querySelectorAll(".angle-chips input").forEach(el => el.disabled = isRunning);
    
    if (isRunning) {
        elStatusIndicator.className = "status-dot blue";
        elStatusText.textContent = `Running ${mode.toUpperCase()}...`;
        elPipelineProgress.style.width = "0%";
    } else {
        elStatusIndicator.className = "status-dot green";
        elStatusText.textContent = "Idle";
        elPipelineProgress.style.width = "0%";
    }
}

function logMessage(text, type = "info") {
    const span = document.createElement("span");
    span.className = `${type}-log`;
    span.textContent = text;
    elConsoleOutput.appendChild(span);
    
    // Auto-scroll console
    elConsoleOutput.scrollTop = elConsoleOutput.scrollHeight;
}
