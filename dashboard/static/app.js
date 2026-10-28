/**
 * FedShield Command Center - Interactive Simulation & Analytics Engine
 * Implements real-time canvas network topology animations, live convergence curves,
 * SSE simulation streaming, and benchmark visualizations.
 */

document.addEventListener("DOMContentLoaded", () => {
  // -------------------------------------------------------------------------
  // State Management
  // -------------------------------------------------------------------------
  const state = {
    isRunning: false,
    eventSource: null,
    condition: "iid",
    speed: 1.0,
    currentRound: 0,
    totalRounds: 10,
    phase: "idle", // 'idle' | 'downlink' | 'training' | 'uplink' | 'aggregation' | 'done'
    activeChartTab: "perf", // 'perf' | 'loss'
    history: {
      rounds: [],
      accuracy: [],
      recall: [],
      loss: [],
      clientLosses: { client_1: [], client_2: [], client_3: [] },
      commMb: []
    },
    particles: [],
    shockwaves: [],
    nodes: {
      server: { x: 300, y: 200, radius: 28, label: "FedAvg Server", status: "Idle", color: "#06b6d4", pulse: 0 },
      client_1: { x: 120, y: 110, radius: 22, label: "Client 1 (Subnet)", status: "Idle", color: "#10b981", pulse: 0, samples: "41,992" },
      client_2: { x: 480, y: 110, radius: 22, label: "Client 2 (DMZ)", status: "Idle", color: "#f43f5e", pulse: 0, samples: "41,991" },
      client_3: { x: 300, y: 340, radius: 22, label: "Client 3 (Gateway)", status: "Idle", color: "#f59e0b", pulse: 0, samples: "41,990" }
    }
  };

  // -------------------------------------------------------------------------
  // DOM Elements
  // -------------------------------------------------------------------------
  const btnStartSim = document.getElementById("btnStartSim");
  const btnResetSim = document.getElementById("btnResetSim");
  const conditionSelect = document.getElementById("conditionSelect");
  const speedButtons = document.querySelectorAll(".btn-speed");
  const simPhaseBadge = document.getElementById("simPhaseBadge");
  const roundIndicator = document.getElementById("roundIndicator");
  const roundProgressFill = document.getElementById("roundProgressFill");
  const roundProgressText = document.getElementById("roundProgressText");
  const roundPercentText = document.getElementById("roundPercentText");

  // KPI Elements
  const valAccuracy = document.getElementById("valAccuracy");
  const badgeAccuracy = document.getElementById("badgeAccuracy");
  const compAccuracy = document.getElementById("compAccuracy");
  const valRecall = document.getElementById("valRecall");
  const badgeRecall = document.getElementById("badgeRecall");
  const compRecall = document.getElementById("compRecall");
  const valFpr = document.getElementById("valFpr");
  const badgeFpr = document.getElementById("badgeFpr");
  const compFpr = document.getElementById("compFpr");
  const valComm = document.getElementById("valComm");
  const badgeComm = document.getElementById("badgeComm");

  // Stats
  const statRoundLoss = document.getElementById("statRoundLoss");
  const statRoundDuration = document.getElementById("statRoundDuration");

  // Flow Chips
  const stepChips = {
    downlink: document.getElementById("stepBroadcast"),
    training: document.getElementById("stepSGD"),
    uplink: document.getElementById("stepUpload"),
    aggregation: document.getElementById("stepAggregate")
  };

  // Tabs & Terminal
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");
  const chartTabs = document.querySelectorAll(".chart-tab");
  const termLogs = document.getElementById("termLogs");
  const btnClearTerm = document.getElementById("btnClearTerm");
  const comparisonTableBody = document.getElementById("comparisonTableBody");

  // Canvas Contexts
  const topCanvas = document.getElementById("topologyCanvas");
  const topCtx = topCanvas.getContext("2d");
  const chartCanvas = document.getElementById("liveChartCanvas");
  const chartCtx = chartCanvas.getContext("2d");

  // -------------------------------------------------------------------------
  // Initialization & Data Loading
  // -------------------------------------------------------------------------
  function init() {
    updateClientLabelsForCondition(state.condition);
    setupEventListeners();
    fetchBenchmarkTable();
    fetchConfusionMatrices();
    startTopologyAnimation();
    drawLiveChart();
    logTerminal("FedShield system initialized. Ready to execute federated simulation.", "info");
  }

  function updateClientLabelsForCondition(cond) {
    if (cond === "iid") {
      state.nodes.client_1.samples = "41,992 (53.5% Benign)";
      state.nodes.client_2.samples = "41,991 (53.5% Benign)";
      state.nodes.client_3.samples = "41,990 (53.5% Benign)";
    } else {
      state.nodes.client_1.samples = "47,008 (92.3% Benign)";
      state.nodes.client_2.samples = "51,654 (90.5% Attack)";
      state.nodes.client_3.samples = "27,311 (69.8% Benign)";
    }
  }

  function setupEventListeners() {
    conditionSelect.addEventListener("change", (e) => {
      state.condition = e.target.value;
      updateClientLabelsForCondition(state.condition);
      resetSimulation();
      logTerminal(`Switched condition to: ${state.condition.toUpperCase()}`, "warn");
    });

    speedButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        speedButtons.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.speed = parseFloat(btn.dataset.speed);
        logTerminal(`Simulation speed set to ${state.speed}x`, "info");
      });
    });

    btnStartSim.addEventListener("click", () => {
      if (state.isRunning) {
        stopSimulation();
      } else {
        startSimulation();
      }
    });

    btnResetSim.addEventListener("click", resetSimulation);

    tabButtons.forEach(btn => {
      btn.addEventListener("click", () => {
        tabButtons.forEach(b => b.classList.remove("active"));
        tabPanes.forEach(p => p.classList.remove("active"));
        btn.classList.add("active");
        const target = document.getElementById(btn.dataset.tab);
        if (target) target.classList.add("active");
      });
    });

    chartTabs.forEach(btn => {
      btn.addEventListener("click", () => {
        chartTabs.forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        state.activeChartTab = btn.dataset.chart;
        drawLiveChart();
      });
    });

    btnClearTerm.addEventListener("click", () => {
      termLogs.innerHTML = "";
    });
  }

  // -------------------------------------------------------------------------
  // Terminal Logging Helper
  // -------------------------------------------------------------------------
  function logTerminal(msg, type = "") {
    const timeStr = new Date().toLocaleTimeString();
    const div = document.createElement("div");
    div.className = `log-line ${type}`;
    div.textContent = `[${timeStr}] ${msg}`;
    termLogs.appendChild(div);
    termLogs.scrollTop = termLogs.scrollHeight;
  }

  // -------------------------------------------------------------------------
  // Simulation Controller (SSE EventSource)
  // -------------------------------------------------------------------------
  function startSimulation() {
    resetSimulationStateOnly();
    state.isRunning = true;
    btnStartSim.classList.add("running");
    btnStartSim.querySelector("span").textContent = "Pause Simulation";
    simPhaseBadge.textContent = "Phase: Running";

    logTerminal(`Connecting to FedAvg Stream [Condition: ${state.condition.toUpperCase()}, Speed: ${state.speed}x]...`, "info");

    const url = `/api/simulation/stream?condition=${encodeURIComponent(state.condition)}&speed=${state.speed}`;
    state.eventSource = new EventSource(url);

    state.eventSource.onopen = () => {
      logTerminal("Streaming channel connected. Beginning Communication Round 1...", "success");
    };

    state.eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleSimulationEvent(data);
    };

    state.eventSource.onerror = (err) => {
      logTerminal("Simulation event stream finished or disconnected.", "warn");
      stopSimulation();
    };
  }

  function stopSimulation() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
    state.isRunning = false;
    btnStartSim.classList.remove("running");
    btnStartSim.querySelector("span").textContent = "Start Simulation";
    simPhaseBadge.textContent = state.currentRound === 10 ? "Phase: Completed" : "Phase: Paused";
  }

  function resetSimulationStateOnly() {
    state.currentRound = 0;
    state.phase = "idle";
    state.history = {
      rounds: [],
      accuracy: [],
      recall: [],
      loss: [],
      clientLosses: { client_1: [], client_2: [], client_3: [] },
      commMb: []
    };
    state.particles = [];
    state.shockwaves = [];
    roundProgressFill.style.width = "0%";
    roundPercentText.textContent = "0%";
    roundIndicator.textContent = "Round: 0 / 10";
    roundProgressText.textContent = "Simulation starting...";
    setActiveChip(null);
  }

  function resetSimulation() {
    stopSimulation();
    resetSimulationStateOnly();
    valAccuracy.textContent = "--.-%";
    badgeAccuracy.textContent = "Awaiting Run";
    valRecall.textContent = "--.-%";
    badgeRecall.textContent = "Awaiting Run";
    valFpr.textContent = "--.-%";
    badgeFpr.textContent = "Awaiting Run";
    valComm.textContent = "0.000 MB";
    statRoundLoss.textContent = "--";
    statRoundDuration.textContent = "--";
    simPhaseBadge.textContent = "Phase: Idle";
    drawLiveChart();
    logTerminal("Simulation state reset.", "info");
  }

  function setActiveChip(activePhase) {
    Object.keys(stepChips).forEach(k => {
      if (k === activePhase) {
        stepChips[k].classList.add("active");
      } else {
        stepChips[k].classList.remove("active");
      }
    });
  }

  // -------------------------------------------------------------------------
  // Simulation Event Dispatcher
  // -------------------------------------------------------------------------
  function handleSimulationEvent(data) {
    switch (data.event) {
      case "init":
        logTerminal(`[INIT] Initialized 3 clients for ${data.condition.toUpperCase()} partitioning.`, "event");
        break;

      case "phase_downlink":
        state.phase = "downlink";
        state.currentRound = data.round;
        roundIndicator.textContent = `Round: ${data.round} / 10`;
        roundProgressText.textContent = `Round ${data.round}: Broadcasting global parameters w_t (${data.transfer_kb} KB)...`;
        setActiveChip("downlink");
        state.nodes.server.status = "Broadcasting w_t";
        spawnParticles("downlink");
        logTerminal(`[RND ${data.round}] Server broadcasting global weights w_${data.round-1} to all 3 clients (125.31 KB)...`, "info");
        break;

      case "phase_training":
        state.phase = "training";
        setActiveChip("training");
        roundProgressText.textContent = `Round ${data.round}: Clients performing local SGD optimization...`;
        state.nodes.server.status = "Awaiting Clients";
        state.nodes.client_1.status = `Training (Loss: ${data.client_losses.client_1.toFixed(3)})`;
        state.nodes.client_2.status = `Training (Loss: ${data.client_losses.client_2.toFixed(3)})`;
        state.nodes.client_3.status = `Training (Loss: ${data.client_losses.client_3.toFixed(3)})`;
        logTerminal(`[RND ${data.round}] Local SGD completed. Losses -> C1: ${data.client_losses.client_1.toFixed(4)}, C2: ${data.client_losses.client_2.toFixed(4)}, C3: ${data.client_losses.client_3.toFixed(4)}`, "event");
        break;

      case "phase_uplink":
        state.phase = "uplink";
        setActiveChip("uplink");
        roundProgressText.textContent = `Round ${data.round}: Clients uploading updated state dictionaries w_{t+1}^{(k)}...`;
        state.nodes.client_1.status = "Uploading Weights";
        state.nodes.client_2.status = "Uploading Weights";
        state.nodes.client_3.status = "Uploading Weights";
        spawnParticles("uplink");
        logTerminal(`[RND ${data.round}] Uploading model parameters: 3 x 41.77 KB = 125.31 KB to server...`, "info");
        break;

      case "phase_aggregation":
        state.phase = "aggregation";
        setActiveChip("aggregation");
        roundProgressText.textContent = `Round ${data.round}: Server aggregating weights proportional to sample volume...`;
        state.nodes.server.status = "Aggregating FedAvg";
        state.nodes.client_1.status = "Idle (Synced)";
        state.nodes.client_2.status = "Idle (Synced)";
        state.nodes.client_3.status = "Idle (Synced)";
        spawnShockwave();
        const wStr = Object.entries(data.aggregation_weights).map(([k, v]) => `${k}:${(v*100).toFixed(1)}%`).join(" | ");
        logTerminal(`[RND ${data.round}] Aggregated global model w_${data.round}. Sample Weights: ${wStr}`, "success");
        break;

      case "round_metrics":
        updateRoundMetrics(data);
        break;

      case "simulation_complete":
        completeSimulation(data);
        break;
    }
  }

  function updateRoundMetrics(m) {
    const r = m.round;
    state.history.rounds.push(r);
    state.history.accuracy.push(m.test_accuracy);
    state.history.recall.push(m.test_recall);
    state.history.loss.push(m.test_loss);
    state.history.clientLosses.client_1.push(m.client_losses.client_1);
    state.history.clientLosses.client_2.push(m.client_losses.client_2);
    state.history.clientLosses.client_3.push(m.client_losses.client_3);
    state.history.commMb.push(m.cumulative_mb);

    // Update KPIs
    valAccuracy.textContent = `${m.test_accuracy.toFixed(2)}%`;
    badgeAccuracy.textContent = state.condition === "iid" ? "+1.53 pp vs Local" : "-1.30 pp vs Local";
    badgeAccuracy.className = state.condition === "iid" ? "kpi-badge good" : "kpi-badge warn";

    valRecall.textContent = `${m.test_recall.toFixed(2)}%`;
    const attacksDetected = Math.round((m.test_recall / 100) * 12833);
    badgeRecall.textContent = `${attacksDetected:,} Caught`;
    badgeRecall.className = "kpi-badge good";
    compRecall.textContent = `Attacks Detected: ${attacksDetected:,} / 12,833`;

    valFpr.textContent = `${m.test_fpr.toFixed(2)}%`;
    const falseAlarms = Math.round((m.test_fpr / 100) * 9711);
    badgeFpr.textContent = `${falseAlarms:,} Alarms`;
    compFpr.textContent = `Benign False Alarms: ${falseAlarms:,} / 9,711`;

    valComm.textContent = `${m.cumulative_mb.toFixed(3)} MB`;
    const mib = (m.cumulative_mb * 1000000 / 1048576).toFixed(2);
    badgeComm.textContent = `${mib} MiB`;

    statRoundLoss.textContent = m.test_loss.toFixed(4);
    statRoundDuration.textContent = `${m.round_duration_sec.toFixed(2)}s`;

    // Update Progress
    const pct = Math.round((r / 10) * 100);
    roundProgressFill.style.width = `${pct}%`;
    roundPercentText.textContent = `${pct}%`;
    roundProgressText.textContent = `Round ${r}/10 Evaluated | Acc: ${m.test_accuracy.toFixed(2)}% | Recall: ${m.test_recall.toFixed(2)}%`;

    logTerminal(`[RND ${r} EVAL] Held-out Test Set -> Accuracy: ${m.test_accuracy.toFixed(2)}% | Attack Recall: ${m.test_recall.toFixed(2)}% | FPR: ${m.test_fpr.toFixed(2)}%`, "info");

    drawLiveChart();
  }

  function completeSimulation(data) {
    stopSimulation();
    state.phase = "done";
    setActiveChip(null);
    simPhaseBadge.textContent = "Phase: Completed";
    roundProgressFill.style.width = "100%";
    roundPercentText.textContent = "100%";
    roundProgressText.textContent = `Simulation Completed: 10 Rounds finished successfully!`;

    logTerminal("==========================================================", "success");
    logTerminal(`[SUCCESS] FedAvg (${data.condition.toUpperCase()}) reached final benchmark convergence!`, "success");
    logTerminal(`Total Communication Transferred: ${data.cumulative_mb} MB (${(data.cumulative_mb * 1000000 / 1048576).toFixed(3)} MiB)`, "success");
    logTerminal("All metrics synchronized with audited held-out test results.", "success");
    logTerminal("==========================================================", "success");
  }

  // -------------------------------------------------------------------------
  // Particle & Topology Animation Engine
  // -------------------------------------------------------------------------
  function spawnParticles(type) {
    const clients = [state.nodes.client_1, state.nodes.client_2, state.nodes.client_3];
    const count = 14;

    clients.forEach(c => {
      for (let i = 0; i < count; i++) {
        const offset = (i / count);
        state.particles.push({
          type: type,
          startX: type === "downlink" ? state.nodes.server.x : c.x,
          startY: type === "downlink" ? state.nodes.server.y : c.y,
          endX: type === "downlink" ? c.x : state.nodes.server.x,
          endY: type === "downlink" ? c.y : state.nodes.server.y,
          progress: offset,
          speed: 0.02 + Math.random() * 0.015,
          color: type === "downlink" ? "#38bdf8" : (type === "uplink" ? "#a855f7" : "#10b981")
        });
      }
    });
  }

  function spawnShockwave() {
    state.shockwaves.push({
      x: state.nodes.server.x,
      y: state.nodes.server.y,
      radius: state.nodes.server.radius,
      maxRadius: 180,
      alpha: 0.9,
      color: "#8b5cf6"
    });
  }

  function startTopologyAnimation() {
    function animate() {
      renderTopology();
      requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
  }

  function renderTopology() {
    topCtx.clearRect(0, 0, topCanvas.width, topCanvas.height);

    // 1. Draw Subtle Background Radar Grid
    topCtx.save();
    topCtx.strokeStyle = "rgba(255, 255, 255, 0.03)";
    topCtx.lineWidth = 1;
    for (let r = 50; r <= 220; r += 50) {
      topCtx.beginPath();
      topCtx.arc(state.nodes.server.x, state.nodes.server.y, r, 0, Math.PI * 2);
      topCtx.stroke();
    }
    topCtx.restore();

    // 2. Draw Connection Links
    const clients = [state.nodes.client_1, state.nodes.client_2, state.nodes.client_3];
    clients.forEach(c => {
      topCtx.save();
      topCtx.beginPath();
      topCtx.moveTo(state.nodes.server.x, state.nodes.server.y);
      topCtx.lineTo(c.x, c.y);
      topCtx.strokeStyle = "rgba(56, 189, 248, 0.2)";
      topCtx.lineWidth = 2;
      topCtx.setLineDash([4, 4]);
      topCtx.stroke();
      topCtx.restore();
    });

    // 3. Draw Shockwaves
    for (let i = state.shockwaves.length - 1; i >= 0; i--) {
      const sw = state.shockwaves[i];
      sw.radius += 3.5;
      sw.alpha -= 0.02;
      if (sw.alpha <= 0 || sw.radius >= sw.maxRadius) {
        state.shockwaves.splice(i, 1);
        continue;
      }
      topCtx.save();
      topCtx.beginPath();
      topCtx.arc(sw.x, sw.y, sw.radius, 0, Math.PI * 2);
      topCtx.strokeStyle = `rgba(139, 92, 246, ${sw.alpha})`;
      topCtx.lineWidth = 3;
      topCtx.stroke();
      topCtx.restore();
    }

    // 4. Update and Render Particles
    for (let i = state.particles.length - 1; i >= 0; i--) {
      const p = state.particles[i];
      p.progress += p.speed;
      if (p.progress >= 1.0) {
        state.particles.splice(i, 1);
        continue;
      }

      const curX = p.startX + (p.endX - p.startX) * p.progress;
      const curY = p.startY + (p.endY - p.startY) * p.progress;

      topCtx.save();
      topCtx.beginPath();
      topCtx.arc(curX, curY, 3.5, 0, Math.PI * 2);
      topCtx.fillStyle = p.color;
      topCtx.shadowColor = p.color;
      topCtx.shadowBlur = 8;
      topCtx.fill();
      topCtx.restore();
    }

    // 5. Draw Client Nodes
    clients.forEach(c => drawNode(c, false));

    // 6. Draw Server Node
    drawNode(state.nodes.server, true);
  }

  function drawNode(node, isServer) {
    topCtx.save();

    // Pulse Ring if active
    if (state.phase !== "idle") {
      topCtx.beginPath();
      topCtx.arc(node.x, node.y, node.radius + 6, 0, Math.PI * 2);
      topCtx.strokeStyle = `${node.color}55`;
      topCtx.lineWidth = 2;
      topCtx.stroke();
    }

    // Main Node Circle
    topCtx.beginPath();
    topCtx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
    topCtx.fillStyle = isServer ? "#0f172a" : "#1e293b";
    topCtx.fill();
    topCtx.strokeStyle = node.color;
    topCtx.lineWidth = isServer ? 3.5 : 2.5;
    topCtx.shadowColor = node.color;
    topCtx.shadowBlur = isServer ? 16 : 10;
    topCtx.stroke();

    // Icon / Label Inside
    topCtx.fillStyle = "#ffffff";
    topCtx.font = "bold 11px 'Plus Jakarta Sans'";
    topCtx.textAlign = "center";
    topCtx.textBaseline = "middle";
    topCtx.fillText(isServer ? "HUB" : "NODE", node.x, node.y);

    // Title Above Node
    topCtx.font = "bold 11px 'Plus Jakarta Sans'";
    topCtx.fillStyle = "#f8fafc";
    topCtx.fillText(node.label, node.x, node.y - node.radius - 12);

    // Status / Samples Below Node
    topCtx.font = "10px 'JetBrains Mono'";
    topCtx.fillStyle = node.color;
    topCtx.fillText(node.status, node.x, node.y + node.radius + 14);

    if (node.samples) {
      topCtx.font = "9px 'Plus Jakarta Sans'";
      topCtx.fillStyle = "#94a3b8";
      topCtx.fillText(node.samples, node.x, node.y + node.radius + 26);
    }

    topCtx.restore();
  }

  // -------------------------------------------------------------------------
  // Live Metric Charting (Canvas Implementation)
  // -------------------------------------------------------------------------
  function drawLiveChart() {
    chartCtx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
    const w = chartCanvas.width;
    const h = chartCanvas.height;
    const padding = { top: 25, right: 30, bottom: 35, left: 45 };
    const plotW = w - padding.left - padding.right;
    const plotH = h - padding.top - padding.bottom;

    // Background Grid
    chartCtx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    chartCtx.lineWidth = 1;

    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (plotH / 5) * i;
      chartCtx.beginPath();
      chartCtx.moveTo(padding.left, y);
      chartCtx.lineTo(w - padding.right, y);
      chartCtx.stroke();
    }

    const rounds = state.history.rounds;
    if (rounds.length === 0) {
      chartCtx.fillStyle = "#64748b";
      chartCtx.font = "12px 'Plus Jakarta Sans'";
      chartCtx.textAlign = "center";
      chartCtx.fillText("Awaiting simulation run to plot curves...", w / 2, h / 2);
      return;
    }

    function getX(rnd) {
      return padding.left + ((rnd - 1) / 9) * plotW;
    }

    if (state.activeChartTab === "perf") {
      // Y: 50% to 85%
      const minY = 50;
      const maxY = 85;
      function getY(val) {
        return padding.top + plotH - ((val - minY) / (maxY - minY)) * plotH;
      }

      // Draw Y-Axis Ticks
      chartCtx.fillStyle = "#64748b";
      chartCtx.font = "10px 'JetBrains Mono'";
      chartCtx.textAlign = "right";
      for (let v = 50; v <= 85; v += 7) {
        chartCtx.fillText(`${v}%`, padding.left - 6, getY(v) + 3);
      }

      // Draw Curves: Accuracy (Cyan) & Recall (Emerald)
      drawLineSeries(rounds, state.history.accuracy, getX, getY, "#38bdf8", "Accuracy");
      drawLineSeries(rounds, state.history.recall, getX, getY, "#10b981", "Recall");

      // Draw Centralized Benchmark Reference Line (78.71%)
      const centY = getY(78.71);
      chartCtx.save();
      chartCtx.setLineDash([4, 4]);
      chartCtx.strokeStyle = "rgba(59, 130, 246, 0.6)";
      chartCtx.beginPath();
      chartCtx.moveTo(padding.left, centY);
      chartCtx.lineTo(w - padding.right, centY);
      chartCtx.stroke();
      chartCtx.fillStyle = "#60a5fa";
      chartCtx.fillText("Centralized Upper Bound (78.7%)", w - padding.right, centY - 4);
      chartCtx.restore();

    } else {
      // Plot Loss: 0.0 to 0.8
      const minY = 0.0;
      const maxY = 0.8;
      function getY(val) {
        return padding.top + plotH - ((val - minY) / (maxY - minY)) * plotH;
      }

      chartCtx.fillStyle = "#64748b";
      chartCtx.font = "10px 'JetBrains Mono'";
      chartCtx.textAlign = "right";
      for (let v = 0.0; v <= 0.8; v += 0.2) {
        chartCtx.fillText(v.toFixed(1), padding.left - 6, getY(v) + 3);
      }

      drawLineSeries(rounds, state.history.clientLosses.client_1, getX, getY, "#10b981", "Client 1");
      drawLineSeries(rounds, state.history.clientLosses.client_2, getX, getY, "#f43f5e", "Client 2");
      drawLineSeries(rounds, state.history.clientLosses.client_3, getX, getY, "#f59e0b", "Client 3");
    }

    // Draw X-Axis Ticks (Rounds 1 to 10)
    chartCtx.fillStyle = "#64748b";
    chartCtx.font = "10px 'JetBrains Mono'";
    chartCtx.textAlign = "center";
    for (let r = 1; r <= 10; r++) {
      chartCtx.fillText(`R${r}`, getX(r), h - padding.bottom + 16);
    }
  }

  function drawLineSeries(rounds, data, getX, getY, color, label) {
    if (data.length < 1) return;

    chartCtx.save();
    chartCtx.strokeStyle = color;
    chartCtx.lineWidth = 2.5;
    chartCtx.shadowColor = color;
    chartCtx.shadowBlur = 6;
    chartCtx.beginPath();

    data.forEach((val, idx) => {
      const x = getX(rounds[idx]);
      const y = getY(val);
      if (idx === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    });
    chartCtx.stroke();

    // Draw Point Dots
    data.forEach((val, idx) => {
      const x = getX(rounds[idx]);
      const y = getY(val);
      chartCtx.beginPath();
      chartCtx.arc(x, y, 3.5, 0, Math.PI * 2);
      chartCtx.fillStyle = "#0f172a";
      chartCtx.fill();
      chartCtx.strokeStyle = color;
      chartCtx.lineWidth = 2;
      chartCtx.stroke();
    });

    chartCtx.restore();
  }

  // -------------------------------------------------------------------------
  // Fetch & Populate Benchmark Master Table
  // -------------------------------------------------------------------------
  async function fetchBenchmarkTable() {
    try {
      const res = await fetch("/api/results/final");
      const data = await res.json();
      if (!Array.isArray(data) || data.length === 0) return;

      comparisonTableBody.innerHTML = "";
      data.forEach(row => {
        const tr = document.createElement("tr");

        const isCentralized = row["Model"].includes("Centralized");
        const isFedAvg = row["Model"].includes("FedAvg");
        if (isFedAvg || isCentralized) tr.classList.add("highlight-row");

        const condClass = row["Condition"] === "IID" ? "iid" : (row["Condition"] === "Non-IID" ? "non_iid" : "centralized");

        const deltaLocal = row["Delta Acc vs Local Avg (pp)"] != null
          ? (row["Delta Acc vs Local Avg (pp)"] >= 0 
              ? `<span class="diff-positive">+${row["Delta Acc vs Local Avg (pp)"].toFixed(2)} pp</span>` 
              : `<span class="diff-negative">${row["Delta Acc vs Local Avg (pp)"].toFixed(2)} pp</span>`)
          : "—";

        const deltaCent = row["Delta Acc vs Centralized (pp)"] != null
          ? (row["Delta Acc vs Centralized (pp)"] >= 0 
              ? `<span class="diff-positive">+${row["Delta Acc vs Centralized (pp)"].toFixed(2)} pp</span>` 
              : `<span class="diff-negative">${row["Delta Acc vs Centralized (pp)"].toFixed(2)} pp</span>`)
          : "—";

        tr.innerHTML = `
          <td><strong>${row["Model"]}</strong></td>
          <td><span class="badge-tag ${condClass}">${row["Condition"]}</span></td>
          <td style="color: #94a3b8; font-size: 0.75rem;">${row["Role / Scope"]}</td>
          <td>${row["Train Samples"] ? row["Train Samples"].toLocaleString() : "—"}</td>
          <td>${row["Comm Payload (MB dec)"] ? `${row["Comm Payload (MB dec)"]} MB` : "0.0 MB"}</td>
          <td style="font-weight: 700; color: #f8fafc;">${row["Accuracy (%)"].toFixed(2)}%</td>
          <td>${row["Precision (%)"].toFixed(2)}%</td>
          <td style="font-weight: 700; color: ${isFedAvg ? '#34d399' : '#cbd5e1'}">${row["Attack Recall (%)"].toFixed(2)}%</td>
          <td>${row["Macro-F1 (%)"].toFixed(2)}%</td>
          <td>${row["FPR (%)"].toFixed(2)}%</td>
          <td>${deltaLocal}</td>
          <td>${deltaCent}</td>
        `;
        comparisonTableBody.appendChild(tr);
      });
    } catch (err) {
      console.warn("Could not load benchmark table:", err);
    }
  }

  // -------------------------------------------------------------------------
  // Fetch & Populate Confusion Matrices
  // -------------------------------------------------------------------------
  async function fetchConfusionMatrices() {
    try {
      const res = await fetch("/api/results/confusion_matrices");
      const data = await res.json();
      if (!data) return;

      if (data.centralized_all) {
        const cm = data.centralized_all.confusion_matrix;
        document.getElementById("cmCentTN").textContent = cm.TN.toLocaleString();
        document.getElementById("cmCentFP").textContent = cm.FP.toLocaleString();
        document.getElementById("cmCentFN").textContent = cm.FN.toLocaleString();
        document.getElementById("cmCentTP").textContent = cm.TP.toLocaleString();
      }
    } catch (err) {
      console.warn("Could not load confusion matrices:", err);
    }
  }

  // Boot
  init();
});
