const API_PORT = 6501;
const API = `${window.location.protocol}//${window.location.hostname}:${API_PORT}`;

let lightOn = false;
let speed = 50;

const el = (id) => document.getElementById(id);
const statusEl = el('status');
const connectionEl = el('connection');
const speedSlider = el('speed');
const speedVal = el('speedVal');

function setStatus(text) {
  statusEl.textContent = text;
}

function setConnection(ok, text) {
  connectionEl.textContent = text;
  connectionEl.classList.toggle('ok', ok);
  connectionEl.classList.toggle('bad', !ok);
}

async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== null) opts.body = JSON.stringify(body);
  const res = await fetch(API + path, opts);
  const text = await res.text();
  const data = text ? JSON.parse(text) : {};
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || `HTTP ${res.status}`);
  }
  return data;
}

function move(cmd) {
  setStatus(`发送指令 ${cmd} ...`);
  api('/api/move', 'POST', { cmd, speed })
    .then(() => setStatus(`已发送指令 ${cmd}`))
    .catch((err) => setStatus(`控制失败：${err.message}`));
}

function stop() {
  api('/api/stop', 'POST')
    .then(() => setStatus('已停止'))
    .catch((err) => setStatus(`停止失败：${err.message}`));
}

function bindMoveButton(btn) {
  const cmd = parseInt(btn.dataset.cmd, 10);
  let active = false;
  let timer = null;

  const start = (event) => {
    event.preventDefault();
    if (active) return;
    active = true;
    btn.classList.add('active');
    move(cmd);
    if (cmd !== 0) {
      timer = window.setInterval(() => move(cmd), 140);
    }
  };

  const end = (event) => {
    event.preventDefault();
    if (!active) return;
    active = false;
    btn.classList.remove('active');
    if (timer) {
      window.clearInterval(timer);
      timer = null;
    }
    if (cmd !== 0) stop();
  };

  btn.addEventListener('pointerdown', start, { passive: false });
  btn.addEventListener('pointerup', end, { passive: false });
  btn.addEventListener('pointercancel', end, { passive: false });
  btn.addEventListener('pointerleave', end, { passive: false });
  btn.addEventListener('contextmenu', (event) => event.preventDefault());
}

speedSlider.addEventListener('input', () => {
  speed = parseInt(speedSlider.value, 10);
  speedVal.textContent = speed;
});

document.querySelectorAll('[data-cmd]').forEach(bindMoveButton);

el('btnStop').addEventListener('click', stop);

el('btnLight').addEventListener('click', async () => {
  lightOn = !lightOn;
  try {
    await api('/api/light', 'POST', { on: lightOn });
    const btn = el('btnLight');
    btn.textContent = lightOn ? '车灯 开' : '车灯 关';
    btn.classList.toggle('on', lightOn);
  } catch (err) {
    setStatus(`车灯失败：${err.message}`);
  }
});

el('btnBeep').addEventListener('click', () => {
  api('/api/beep', 'POST', { duration: 100 }).catch((err) => setStatus(`蜂鸣失败：${err.message}`));
});

// ---- 语音控制 ----
const voiceBtn = el('btnVoice');
const voiceResult = el('voiceResult');
if (voiceBtn) {
  let voiceBusy = false;

  voiceBtn.addEventListener('pointerdown', async (event) => {
  event.preventDefault();
  if (voiceBusy) return;
  voiceBusy = true;
  voiceBtn.classList.add('recording');
  voiceBtn.textContent = '🔴 录音中...';
  voiceResult.textContent = '正在聆听...';
  voiceResult.classList.remove('error');

  try {
    const res = await api('/api/voice', 'POST');
    if (res.action) {
      voiceResult.textContent = `"${res.text}" → ${res.exec.action}`;
      voiceResult.classList.remove('error');
      setStatus(`语音: "${res.text}" → 已执行`);
    } else if (res.text) {
      voiceResult.textContent = `听到: "${res.text}" (未匹配指令)`;
      voiceResult.classList.add('error');
    } else {
      voiceResult.textContent = res.error || '未识别到语音';
      voiceResult.classList.add('error');
    }
  } catch (err) {
    voiceResult.textContent = `语音识别失败：${err.message}`;
    voiceResult.classList.add('error');
  } finally {
    voiceBtn.textContent = '🎤 按住说话';
    voiceBtn.classList.remove('recording');
    voiceBtn.disabled = false;
    voiceBusy = false;
  }
});
}  // if (voiceBtn)

// ---- 人物追踪 ----
const btnTrack = el('btnTrack');
const trackResult = el('trackResult');
if (btnTrack) {
  let tracking = false;
  btnTrack.addEventListener('click', async () => {
  if (tracking) {
    // 停止追踪
    try {
      await api('/api/track/stop', 'POST');
      tracking = false;
      btnTrack.textContent = '🎯 启动人物追踪';
      btnTrack.classList.remove('tracking');
      trackResult.textContent = '追踪已停止';
    } catch (err) {
      trackResult.textContent = `停止失败：${err.message}`;
    }
  } else {
    // 启动追踪
    try {
      await api('/api/track/start', 'POST');
      tracking = true;
      btnTrack.textContent = '🛑 停止追踪';
      btnTrack.classList.add('tracking');
      trackResult.textContent = '追踪中...';
    } catch (err) {
      trackResult.textContent = `启动失败：${err.message}`;
    }
  }
});
}  // if (btnTrack)

async function refreshSensors() {
  try {
    const res = await api('/api/sensors');
    el('temp').textContent = res.data.temperature;
    el('humidity').textContent = res.data.humidity;
    el('battery').textContent = res.data.battery;
  } catch (err) {
    setStatus(`传感器读取失败：${err.message}`);
  }
}

async function refreshStatus() {
  try {
    const res = await api('/api/status');
    setConnection(true, 'API 已连接');
    el('videoMeta').textContent = `${res.data.video.width}×${res.data.video.height} @ ${res.data.video.fps}fps`;
    setStatus(`固件 V${res.data.version} | 摄像头 ${res.data.camera_ok ? '正常' : '异常'} | ${res.data.car_type_name}`);
  } catch (err) {
    setConnection(false, 'API 未连接');
    setStatus(`状态失败：${err.message}`);
  }
}

refreshStatus();
refreshSensors();
window.setInterval(refreshSensors, 1000);
window.setInterval(refreshStatus, 5000);
