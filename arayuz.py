# -*- coding: utf-8 -*-
# 4 Kamera Web Paneli (Flask) + Pan-Tilt (Arduino Mega) + YOLO (tüm kameralar)
# Kamera 1 (slot 0) için: Dijital Zoom (1x–5x) + Zoom'a bağlı YOLO imgsz ve JPEG kalite
# + Dijital Güzelleştirme (bilateral + unsharp + CLAHE + gamma) AÇ/KAPAT

import time, threading, os
import cv2, numpy as np
from collections import deque
from flask import Flask, Response, request, jsonify, render_template_string
import serial

# ================== KAMERA/GENEL AYARLAR ==================
NUM_SLOTS         = 4
DEFAULT_IDS       = [0, 1, 2, 3]

# USB bant genişliği için dengeli değerler
CAP_W, CAP_H      = 424, 240      # (640,360) da deneyebilirsin
FPS_TARGET        = 15
VIEW_W, VIEW_H    = 352, 240
DEFAULT_JPEG_Q    = 70

# ================== YOLO AYARLARI (TÜM SLOTLAR) ==================
DETECT_SLOTS      = {0, 1, 2, 3}   # tüm kameralar
MODEL_PATH        = "yolov8n_custom_hedef.pt"
YOLO_CONF         = 0.35
BASE_IMGSZ        = 320            # zoom yokken
BOX_COLOR         = (0, 0, 255)
GHOST_COLOR       = (0, 255, 255)
TEXT_SCALE        = 0.6
TEXT_THICK        = 2
DETECT_EVERY_N    = 6              # 4 kamera için yük dengeleme
DETECT_PHASE_STAGGER = True        # slot bazlı faz kaydırma

# ================== DİJİTAL ZOOM (Per-slot) ==================
MIN_ZOOM          = 1
MAX_ZOOM          = 5
ZOOM_FACTORS      = [1, 1, 1, 1]   # butonla sadece slot 0 değişecek

# ================== DİJİTAL GÜZELLEŞTİRME ==================
ENHANCE_SLOTS = {0}                            # şimdilik sadece kamera 1
ENHANCE_ON    = [True, False, False, False]    # slot 0 başlangıçta AÇIK

# ================== PAN–TILT SERİ ==================
SERIAL_PORT = "/dev/ttyACM0"       # gerekirse /dev/ttyUSB0 /dev/ttyACM1
BAUDRATE    = 9600
try:
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    try:
        ser.setDTR(False); ser.setRTS(False)
        time.sleep(0.1)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
    except Exception:
        pass
    print(f"[INFO] Arduino bağlı: {SERIAL_PORT}")
except Exception as e:
    print("[WARN] Arduino bağlanamadı:", e)
    ser = None

# ================== YOLO MODEL ==================
yolo_ok = False
INFER_LOCK = threading.Lock()  # model.predict için global kilit
try:
    import torch
    from ultralytics import YOLO
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half   = torch.cuda.is_available()
    print(f"[YOLO] Device={device} half={half}")
    # model yolu: aynı klasör yoksa /mnt/data da dene
    if not os.path.exists(MODEL_PATH) and os.path.exists("/mnt/data/" + MODEL_PATH):
        MODEL_PATH = "/mnt/data/" + MODEL_PATH
    model = YOLO(MODEL_PATH)
    try: model.fuse()
    except Exception: pass
    model.to(device)
    if half:
        try:
            model.model.half()
        except Exception:
            half = False
    yolo_ok = True
except Exception as e:
    print("[YOLO] Yükleme başarısız, YOLO devre dışı:", e)
    yolo_ok = False
    device = "cpu"
    half   = False

# ================== FLASK ==================
app = Flask(__name__)

# ================== ÇEKİRDEK ==================
class CameraSlot:
    def __init__(self, slot:int, init_id:int):
        self.slot    = slot
        self.cam_id  = init_id
        self.cap     = None
        self.running = False
        self.thread  = None
        self.lock    = threading.Lock()
        self.last_rgb = None

        # YOLO per-slot
        self.frame_count   = 0
        self.last_boxes    = []   # (x1,y1,x2,y2, conf, cls)
        self.last_infer_ms = 0.0
        self.perf_fps_hist = deque(maxlen=90)

    def set_id(self, new_id:int):
        self.cam_id = int(new_id)

    def _try_open(self):
        cap = cv2.VideoCapture(int(self.cam_id), cv2.CAP_V4L2)
        try: cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except: pass
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except: pass
        for (w,h,fps) in [(CAP_W, CAP_H, FPS_TARGET),
                          (352, 240, 15),
                          (320, 240, 10)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS,          fps)
            ok, _ = cap.read()
            if ok:
                print(f"[INFO] cam{self.slot} -> {w}x{h}@{fps} MJPG")
                return cap
        cap.release()
        return None

    def open(self):
        if self.running: return True
        cap = self._try_open()
        if cap is None: return False
        self.cap = cap
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        return True

    def _reader_loop(self):
        interval = 1.0 / max(1, FPS_TARGET)
        while self.running:
            t0 = time.time()
            ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if ok:
                frame = cv2.resize(frame, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.last_rgb = rgb
            else:
                time.sleep(0.02)
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)

    # ====== Zoom/kalite tabloları ======
    def _current_quality_and_size(self):
        """Zoom faktörüne göre (imgsz, jpeg_quality) döndür."""
        z = ZOOM_FACTORS[self.slot] if 0 <= self.slot < len(ZOOM_FACTORS) else 1
        if z <= 1:
            return 320, 70
        elif z <= 3:
            return 480, 80
        else:
            return 640, 90

    def _apply_zoom(self, bgr):
        factor = ZOOM_FACTORS[self.slot] if 0 <= self.slot < len(ZOOM_FACTORS) else 1
        if factor <= 1:
            return bgr
        h, w = bgr.shape[:2]
        new_w, new_h = max(1, w // factor), max(1, h // factor)
        cx, cy = w // 2, h // 2
        x1, y1 = cx - new_w // 2, cy - new_h // 2
        x2, y2 = x1 + new_w, y1 + new_h
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w, x2); y2 = min(h, y2)
        roi = bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return bgr
        return cv2.resize(roi, (w, h), interpolation=cv2.INTER_LINEAR)

    # ====== Dijital Güzelleştirme ======
    def _enhance(self, bgr):
        """
        Kenar koruyan yumuşatma -> unsharp mask -> CLAHE -> gamma.
        Etki, zoom faktörüne göre artar. Yalnız slot 0'da (varsayılan) kullanılır.
        """
        z = ZOOM_FACTORS[self.slot] if 0 <= self.slot < len(ZOOM_FACTORS) else 1
        if z < 1: z = 1

        # 1) Edge-preserving denoise (bilateral)
        den = cv2.bilateralFilter(bgr, d=0,
                                  sigmaColor=int(15 + 10*z),
                                  sigmaSpace=int(5 + 2*z))

        # 2) Unsharp mask
        sigma  = 0.8 + 0.25 * z
        amount = 0.8 + 0.25 * z
        blur   = cv2.GaussianBlur(den, (0,0), sigma)
        sharp  = cv2.addWeighted(den, 1.0 + amount, blur, -amount, 0)

        # 3) CLAHE (Y kanalında)
        ycrcb = cv2.cvtColor(sharp, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clip = 1.6 + 0.5 * (z-1)   # 1x:~1.6, 5x:~3.6
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8,8))
        y2 = clahe.apply(y)
        out = cv2.cvtColor(cv2.merge([y2, cr, cb]), cv2.COLOR_YCrCb2BGR)

        # 4) Gamma
        gamma = max(0.8, 1.0 - 0.06*(z-1))   # 1x≈1.0, 5x≈0.76 -> taban 0.8
        if abs(gamma - 1.0) > 1e-3:
            inv = 1.0 / gamma
            table = (np.arange(256) / 255.0) ** inv
            table = np.clip(table * 255.0, 0, 255).astype(np.uint8)
            out = cv2.LUT(out, table)

        return out

    # ====== YOLO ======
    def _detect(self, frame_bgr):
        if not yolo_ok:
            return []
        imgsz, _ = self._current_quality_and_size()
        t0 = time.time()
        with INFER_LOCK:
            res = model.predict(
                source=frame_bgr,
                imgsz=imgsz,
                conf=YOLO_CONF,
                device=device,
                verbose=False
            )
        boxes = []
        if res and len(res):
            r = res[0]
            if r.boxes is not None and len(r.boxes) > 0:
                names = r.names if hasattr(r, "names") else {}
                for b in r.boxes:
                    xyxy = b.xyxy[0].to('cpu').numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    conf = float(b.conf[0].to('cpu'))
                    cls_id = int(b.cls[0].to('cpu'))
                    cls_name = names.get(cls_id, str(cls_id))
                    boxes.append((x1, y1, x2, y2, conf, cls_name))
        self.last_infer_ms = (time.time() - t0) * 1000.0
        return boxes

    def _draw_boxes(self, img, boxes, color, info_text=""):
        for (x1,y1,x2,y2,conf,cls_name) in boxes:
            cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label, (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, color, TEXT_THICK, cv2.LINE_AA)
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 18, 2)
        if info_text:
            cv2.putText(img, info_text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

    def _approx_fps(self):
        if not self.perf_fps_hist: return 0.0
        return sum(self.perf_fps_hist) / len(self.perf_fps_hist)

    def get_jpeg(self):
        with self.lock:
            rgb = None if self.last_rgb is None else self.last_rgb.copy()
        if rgb is None:
            frame = black_placeholder(VIEW_W, VIEW_H, "KAPALI")
            _, jpeg_q = self._current_quality_and_size()
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
            return buf.tobytes() if ok else None

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 1) Per-slot dijital zoom
        bgr_z = self._apply_zoom(bgr)

        # 2) Dijital güzelleştirme (açıksa)
        if self.slot in ENHANCE_SLOTS and ENHANCE_ON[self.slot]:
            bgr_z = self._enhance(bgr_z)

        # 3) YOLO (tüm slotlarda; zoom'a göre imgsz seçimi içeride)
        if self.slot in DETECT_SLOTS and yolo_ok:
            self.frame_count += 1
            # Faz kaydırma: her slot farklı karede inferans yapsın
            do_detect = (self.frame_count % DETECT_EVERY_N) == (self.slot % DETECT_EVERY_N) if DETECT_PHASE_STAGGER \
                        else (self.frame_count % DETECT_EVERY_N) == 0

            if do_detect:
                self.last_boxes = self._detect(bgr_z)

            out = bgr_z.copy()
            zf = ZOOM_FACTORS[self.slot] if 0 <= self.slot < len(ZOOM_FACTORS) else 1
            enh = " EZ" if (self.slot in ENHANCE_SLOTS and ENHANCE_ON[self.slot]) else ""
            if do_detect:
                self._draw_boxes(out, self.last_boxes, BOX_COLOR,
                                 info_text=f"FPS:{self._approx_fps():.1f}  INF:{self.last_infer_ms:.0f}ms"
                                           + (f"  Z:{zf}x" if zf>1 else "") + enh)
            else:
                ghost = out.copy()
                self._draw_boxes(ghost, self.last_boxes, GHOST_COLOR)
                cv2.addWeighted(ghost, 0.35, out, 0.65, 0, out)
                cv2.putText(out,
                            f"FPS:{self._approx_fps():.1f}  INF:{self.last_infer_ms:.0f}ms (ghost)"
                            + (f"  Z:{zf}x" if zf>1 else "") + enh,
                            (8,22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            self.perf_fps_hist.append(FPS_TARGET)
            frame = out
        else:
            frame = bgr_z

        # 4) JPEG kaliteyi zoom'a göre seçip encode et
        _, jpeg_q = self._current_quality_and_size()
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_q)])
        return buf.tobytes() if ok else None

    def close(self):
        self.running = False
        try:
            if self.cap and self.cap.isOpened(): self.cap.release()
        except: pass
        self.cap = None
        with self.lock:
            self.last_rgb = None

def black_placeholder(w, h, text="KAPALI"):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
    return img

# ================== UYGULAMA (HTML) ==================
slots = [CameraSlot(i, DEFAULT_IDS[i]) for i in range(NUM_SLOTS)]

HTML = """
<!doctype html><html lang="tr"><head><meta charset="utf-8"/>
<title>4 Kamera Panel + Pan–Tilt + YOLO + Kamera1 Zoom & Güzelleştirme</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
:root{--bg:#0f1117;--card:#161a23;--muted:#262c36;--text:#e6e6e6;--sub:#b6bdc6;--brand:#8ab4f8;--warn:#ff8080;}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 system-ui,Arial}
header{padding:12px;text-align:center;border-bottom:1px solid var(--muted)}
h1{margin:0;font-size:18px}
.container{padding:14px}
.grid{display:grid;grid-template-columns:repeat(4, minmax(0,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--muted);border-radius:12px;padding:10px}
.media{width:352px;height:240px;background:#000;border-radius:8px}
.row{display:flex;gap:8px;align-items:center;margin-top:8px;flex-wrap:wrap}
.btn{background:var(--brand);color:#111;border:0;border-radius:8px;padding:6px 10px;font-weight:700;cursor:pointer}
.btn.secondary{background:#8899a6}.btn.warn{background:var(--warn)}
.badge{font-size:12px;color:#0b0;background:#10391f;border:1px solid #194c2a;padding:2px 8px;border-radius:999px}
.badge.off{color:#c77;background:#392f10;border-color:#4d3d15}
.small{font-size:12px;color:var(--sub)}
.toolbar{margin-top:12px;text-align:center}
.pt{display:grid;grid-template-columns:repeat(3,56px);gap:8px;justify-content:center;margin:6px 0 10px}
.pt .btn{width:56px}
.footer{display:flex;justify-content:center;margin-top:14px}
@media (max-width: 1500px){ .grid{grid-template-columns:repeat(2, minmax(0,1fr));} }
.yolobadge{font-size:12px;padding:2px 6px;border-radius:6px;border:1px solid #333;margin-left:6px}
</style></head><body onload="updateZoomLabel();updateEnhLabel();">
<header><h1>4 Kamera — Pan–Tilt — YOLO (hepsi) — Kamera 1: Zoom & Güzelleştirme</h1></header>
<div class="container">
  <div class="grid">
    {% for i in range(num) %}
    <div class="card">
      <div class="row">
        <strong>Kamera {{i+1}} (slot {{i}})</strong>
        {% if i in detect_slots %}
          <span class="yolobadge" title="Bu kamerada YOLO aktif">YOLO</span>
        {% endif %}
        <span id="st{{i}}" class="badge off">Kapalı</span>
      </div>
      <img class="media" id="img{{i}}" src="/stream/{{i}}.mjpg" />
      <div class="row">
        <label>ID:</label><input id="id{{i}}" type="number" value="{{defaults[i]}}" min="0" max="99"/>
        <button class="btn" onclick="setId({{i}})">Ayarla</button>
        <button class="btn secondary" onclick="openCam({{i}})">Aç</button>
        <button class="btn secondary" onclick="closeCam({{i}})">Kapat</button>
      </div>

      {% if i == 0 %}
      <!-- Kamera 1 Zoom Kontrol -->
      <div class="row">
        <strong>Zoom:</strong>
        <span id="zoom0" class="small">1x</span>
        <button class="btn secondary" onclick="setZoom(0, -1)">−</button>
        <button class="btn secondary" onclick="setZoom(0, +1)">+</button>
        <button class="btn" onclick="setZoomAbs(0, 1)">1x</button>
      </div>
      <!-- Kamera 1 Dijital Güzelleştirme -->
      <div class="row">
        <strong>Güzelleştirme:</strong>
        <span id="enh0" class="small">Açık</span>
        <button class="btn secondary" onclick="toggleEnh(0)">Aç/Kapat</button>
      </div>
      <div class="small">Not: Zoom arttıkça YOLO giriş boyutu ve JPEG kalite otomatik yükselir.</div>
      {% endif %}
    </div>
    {% endfor %}
  </div>

  <!-- Pan–Tilt Toolbar -->
  <div class="card toolbar">
    <div class="row" style="justify-content:center"><strong>Pan–Tilt Kontrolü</strong></div>
    <div class="pt">
      <div></div><button class="btn" onclick="pt('UP')">↑</button><div></div>
      <button class="btn" onclick="pt('LEFT')">←</button>
      <button class="btn" onclick="pt('DOWN')">↓</button>
      <button class="btn" onclick="pt('RIGHT')">→</button>
      <div></div>
    </div>
    <div class="row" style="justify-content:center;gap:8px">
      <button class="btn" onclick="pt('CENTER')">Ortala</button>
      <button class="btn secondary" onclick="pt('SET CENTER HERE')">Burayı Merkez Yap</button>
      <button class="btn secondary" onclick="pt('SAVE CENTER')">Merkezi Kaydet (EEPROM)</button>
      <button class="btn secondary" onclick="ptRaw('POS?')">Açıları Yazdır (POS?)</button>
    </div>
    <div class="row" style="justify-content:center">
      <span class="small">Son cevap: <span id="lastRx">—</span></span>
    </div>
  </div>

  <div class="footer">
    <button class="btn warn" onclick="closeAll()">Tüm Kameraları Kapat</button>
  </div>
</div>

<script>
async function api(path, params={}){
  const url = new URL(path, window.location.origin);
  Object.keys(params).forEach(k=>url.searchParams.set(k, params[k]));
  const r = await fetch(url, {method:'POST'});
  try{return await r.json()}catch(e){return{ok:false}}
}
function setBadge(i,on){const el=document.getElementById('st'+i); el.className='badge '+(on?'':'off'); el.textContent=on?'Açık':'Kapalı';}
function showRx(lines){ document.getElementById('lastRx').textContent = (lines && lines.length? lines.join(' | ') : '—'); }

async function setId(i){const v=document.getElementById('id'+i).value; await api('/api/cam/set_id',{slot:i, cam_id:v}); setBadge(i,false)}
async function openCam(i){const r=await api('/api/cam/open',{slot:i}); setBadge(i,!!r.ok)}
async function closeCam(i){await api('/api/cam/close',{slot:i}); setBadge(i,false)}
async function closeAll(){await api('/api/all/close'); for(let i=0;i<{{num}};i++) setBadge(i,false);}

async function pt(dir){
  const r = await api('/api/pt', {dir});
  if(r && r.rx) showRx(r.rx);
  console.log('PT', dir, r);
}
async function ptRaw(cmd){
  const url = new URL('/api/pt_raw', window.location.origin);
  url.searchParams.set('cmd', cmd);
  const r = await fetch(url, {method:'POST'});
  const j = await r.json();
  if(j && j.rx) showRx(j.rx);
  console.log('PT_RAW', cmd, j);
}

// ---- Kamera 1 Zoom (slot 0) ----
let zoom0 = 1;
function updateZoomLabel(){
  const el = document.getElementById('zoom0');
  if(el) el.textContent = zoom0 + 'x';
}
async function setZoom(slot, delta){
  const target = Math.max({{min_zoom}}, Math.min({{max_zoom}}, zoom0 + delta));
  const r = await api('/api/zoom', {slot: slot, factor: target});
  if(r && r.ok){ zoom0 = r.factor; updateZoomLabel(); }
}
async function setZoomAbs(slot, factor){
  const r = await api('/api/zoom', {slot: slot, factor: factor});
  if(r && r.ok){ zoom0 = r.factor; updateZoomLabel(); }
}

// ---- Kamera 1 Güzelleştirme (slot 0) ----
let enh0 = true;
function updateEnhLabel(){
  const el = document.getElementById('enh0');
  if(el) el.textContent = enh0 ? 'Açık' : 'Kapalı';
}
async function toggleEnh(slot){
  const r = await api('/api/enhance', {slot: slot, on: (enh0?0:1)});
  if(r && r.ok){ enh0 = r.on; updateEnhLabel(); }
}
</script>
</body></html>
"""

@app.route("/")
def index():
    return render_template_string(
        HTML,
        num=NUM_SLOTS,
        defaults=DEFAULT_IDS,
        detect_slots=DETECT_SLOTS,
        min_zoom=MIN_ZOOM,
        max_zoom=MAX_ZOOM
    )

@app.route("/stream/<int:slot>.mjpg")
def stream(slot:int):
    if not (0 <= slot < NUM_SLOTS):
        return "slot?", 404
    def gen():
        target_dt = 1.0 / float(FPS_TARGET)
        last_send = time.time()
        while True:
            buf = slots[slot].get_jpeg()
            if buf is None:
                time.sleep(0.02); continue
            now = time.time()
            sleep_t = target_dt - (now - last_send)
            if sleep_t > 0:
                time.sleep(sleep_t)
            last_send = time.time()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n")
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# ========== Kamera API ==========
@app.route("/api/cam/set_id", methods=["POST"])
def api_set_id():
    slot = int(request.args.get("slot","0"))
    cam_id = int(request.args.get("cam_id","0"))
    if 0 <= slot < NUM_SLOTS:
        slots[slot].set_id(cam_id)
        return jsonify(ok=True)
    return jsonify(ok=False)

@app.route("/api/cam/open", methods=["POST"])
def api_open():
    slot = int(request.args.get("slot","0"))
    if 0 <= slot < NUM_SLOTS:
        ok = slots[slot].open()
        return jsonify(ok=bool(ok))
    return jsonify(ok=False)

@app.route("/api/cam/close", methods=["POST"])
def api_close():
    slot = int(request.args.get("slot","0"))
    if 0 <= slot < NUM_SLOTS:
        slots[slot].close()
        return jsonify(ok=True)
    return jsonify(ok=False)

@app.route("/api/all/close", methods=["POST"])
def api_all_close():
    for s in slots: s.close()
    return jsonify(ok=True)

# ========== Zoom API (Kamera 1 için 1x–5x ve kalite/imgsz otomatik) ==========
@app.route("/api/zoom", methods=["POST"])
def api_zoom():
    try:
        slot = int(request.args.get("slot","0"))
        factor = int(request.args.get("factor","1"))
    except ValueError:
        return jsonify(ok=False, err="bad-params")
    if not (0 <= slot < NUM_SLOTS):
        return jsonify(ok=False, err="bad-slot")
    factor = max(MIN_ZOOM, min(MAX_ZOOM, factor))
    ZOOM_FACTORS[slot] = factor
    print(f"[ZOOM] slot{slot} -> {factor}x (imgsz/jpeg kalite otomatik ayarlanacak)")
    return jsonify(ok=True, slot=slot, factor=factor, min=MIN_ZOOM, max=MAX_ZOOM)

# ========== Güzelleştirme API ==========
@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    try:
        slot = int(request.args.get("slot","0"))
        on   = int(request.args.get("on","1"))  # 1: açık, 0: kapalı
    except ValueError:
        return jsonify(ok=False, err="bad-params")
    if not (0 <= slot < len(ENHANCE_ON)):
        return jsonify(ok=False, err="bad-slot")
    ENHANCE_ON[slot] = bool(on)
    state = "ON" if ENHANCE_ON[slot] else "OFF"
    print(f"[ENHANCE] slot{slot} -> {state}")
    return jsonify(ok=True, slot=slot, on=ENHANCE_ON[slot])

# ========== Pan–Tilt API ==========
@app.route("/api/pt", methods=["POST"])
def api_pt():
    direction = request.args.get("dir","")
    print(f"[PT] {direction}")
    reply_lines = []
    if ser and direction:
        try:
            ser.write((direction + "\n").encode())
            t_end = time.time() + 0.3
            while time.time() < t_end:
                if ser.in_waiting:
                    line = ser.readline().decode(errors="ignore").strip()
                    if line:
                        reply_lines.append(line)
                else:
                    time.sleep(0.01)
            if not reply_lines:
                line = ser.readline().decode(errors="ignore").strip()
                if line:
                    reply_lines.append(line)
            print("[PT][ARDUINO]", " | ".join(reply_lines) if reply_lines else "<no-reply>")
            return jsonify(ok=True, dir=direction.upper(), rx=reply_lines)
        except Exception as e:
            return jsonify(ok=False, err=str(e))
    return jsonify(ok=False, err="no-serial-or-empty-dir")

@app.route("/api/pt_raw", methods=["POST"])
def api_pt_raw():
    cmd = request.args.get("cmd","").strip()
    if ser and cmd:
        try:
            ser.write((cmd + "\n").encode())
            time.sleep(0.05)
            lines = []
            t_end = time.time() + 0.4
            while time.time() < t_end:
                if ser.in_waiting:
                    line = ser.readline().decode(errors="ignore").strip()
                    if line:
                        lines.append(line)
                else:
                    time.sleep(0.01)
            print("[PT_RAW][TX]", cmd, "[RX]", lines)
            return jsonify(ok=True, tx=cmd, rx=lines)
        except Exception as e:
            return jsonify(ok=False, err=str(e))
    return jsonify(ok=False, err="no-serial-or-empty-cmd")

if __name__ == "__main__":
    print("🚀 Panel: http://0.0.0.0:8000")
    if yolo_ok:
        print(f"[YOLO] Model: {MODEL_PATH}  base_imgsz={BASE_IMGSZ}  conf={YOLO_CONF}  DETECT_EVERY_N={DETECT_EVERY_N}")
        print(f"[YOLO] Aktif slotlar: {sorted(list(DETECT_SLOTS))}  Stagger={DETECT_PHASE_STAGGER}")
    else:
        print("[YOLO] Devre dışı (model yüklenmedi)")
    app.run(host="0.0.0.0", port=8000, threaded=True)
