/* ============================================================
   MaskForge — Polygon Annotation Studio
   Pure vanilla JS — no frameworks, no build tools
   ============================================================ */

(function () {
  'use strict';

  // ── COLOUR PALETTE for annotations ──
  const COLORS = [
    '#f05b5b', '#f0974a', '#f0c94a', '#4af0a0',
    '#4ac8f0', '#7b7bf0', '#d06af0', '#f06ab0',
    '#ff8a80', '#82b1ff', '#b9f6ca', '#ffe57f',
  ];

  // ── STATE ──
  const state = {
    image: null,           // HTMLImageElement
    imageName: '',
    originalJSON: null,    // preserve original structure for export
    annotations: [],       // { id, label, type, points, color, attributes }
    selectedId: null,
    activeTool: 'select',
    zoom: 1,
    panX: 0,
    panY: 0,
    nextId: 1,
    // drawing state
    drawing: false,
    drawPoints: [],        // for polygon
    drawStart: null,       // for rect/circle/ellipse
    drawCurrent: null,
    // panning
    isPanning: false,
    panStart: null,
    // vertex dragging
    draggingVertex: null,  // { annId, pointIndex }
  };

  // ── DOM REFS ──
  const $ = (s) => document.querySelector(s);
  const canvas = $('#main-canvas');
  const ctx = canvas.getContext('2d');
  const container = $('#canvas-container');
  const emptyState = $('#empty-state');
  const statusBar = $('#status-bar');
  const zoomLabel = $('#zoom-label');
  const annList = $('#annotation-list');
  const annCount = $('#ann-count');
  const modalOverlay = $('#modal-overlay');
  const contextMenu = $('#context-menu');

  // ── INITIALISE ──
  function init() {
    bindToolbar();
    bindCanvas();
    bindZoom();
    bindFileInputs();
    bindModal();
    bindContextMenu();
    bindDragDrop();
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    render();
  }

  // ── TOOLBAR ──
  function bindToolbar() {
    document.querySelectorAll('.tool-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        setTool(btn.dataset.tool);
      });
    });
    $('#btn-load-image').addEventListener('click', () => $('#file-image').click());
    $('#btn-load-json').addEventListener('click', () => $('#file-json').click());
    $('#btn-export').addEventListener('click', exportJSON);
  }

  function setTool(tool) {
    state.activeTool = tool;
    cancelDrawing();
    document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
    const btn = document.querySelector(`.tool-btn[data-tool="${tool}"]`);
    if (btn) btn.classList.add('active');
    container.setAttribute('data-tool', tool);
    setStatus(toolLabel(tool) + ' tool active');
  }

  function toolLabel(t) {
    return { select: 'Select', polygon: 'Polygon', rect: 'Rectangle', circle: 'Circle', ellipse: 'Ellipse' }[t] || t;
  }

  // ── FILE INPUTS ──
  function bindFileInputs() {
    $('#file-image').addEventListener('change', (e) => {
      const f = e.target.files[0];
      if (f) loadImageFile(f);
      e.target.value = '';
    });
    $('#file-json').addEventListener('change', (e) => {
      const f = e.target.files[0];
      if (f) loadJSONFile(f);
      e.target.value = '';
    });
  }

  function loadImageFile(file) {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      state.image = img;
      state.imageName = file.name;
      emptyState.classList.add('hidden');
      fitToView();
      setStatus('Image loaded: ' + file.name + ' (' + img.width + '×' + img.height + ')');
    };
    img.src = url;
  }

  function loadJSONFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target.result);
        state.originalJSON = json;
        parseAnnotations(json);
        setStatus('JSON loaded: ' + state.annotations.length + ' annotations');
      } catch (err) {
        setStatus('Error parsing JSON: ' + err.message);
      }
    };
    reader.readAsText(file);
  }

  // ── JSON PARSING (supports provided format) ──
  function parseAnnotations(json) {
    state.annotations = [];
    state.nextId = 1;

    if (json.Labels && Array.isArray(json.Labels)) {
      json.Labels.forEach((labelGroup) => {
        const label = labelGroup.Label || 'unlabeled';
        if (labelGroup.Shapes && Array.isArray(labelGroup.Shapes)) {
          labelGroup.Shapes.forEach((shape, idx) => {
            const ann = {
              id: state.nextId++,
              label: label,
              type: shape.Type || 'polygon',
              points: (shape.points || []).map(p => [p[0], p[1]]),
              color: COLORS[(state.nextId - 2) % COLORS.length],
              attributes: shape.Attributes ? { ...shape.Attributes } : {},
              _origShape: shape, // keep reference for export
            };
            state.annotations.push(ann);
          });
        }
      });
    }

    // also support flat COCO-style with "annotations" array
    if (json.annotations && Array.isArray(json.annotations)) {
      json.annotations.forEach((ann) => {
        const seg = ann.segmentation;
        if (seg && Array.isArray(seg) && seg.length > 0) {
          const flat = seg[0];
          const pts = [];
          for (let i = 0; i < flat.length; i += 2) {
            pts.push([flat[i], flat[i + 1]]);
          }
          state.annotations.push({
            id: state.nextId++,
            label: 'object_' + ann.category_id,
            type: 'polygon',
            points: pts,
            color: COLORS[(state.nextId - 2) % COLORS.length],
            attributes: { category_id: ann.category_id, area: ann.area },
          });
        }
      });
    }

    state.selectedId = null;
    refreshList();
    render();
  }

  // ── CANVAS SETUP ──
  function resizeCanvas() {
    const rect = container.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    render();
  }

  function fitToView() {
    if (!state.image) return;
    const cw = canvas.width;
    const ch = canvas.height;
    const iw = state.image.width;
    const ih = state.image.height;
    const scale = Math.min(cw / iw, ch / ih) * 0.92;
    state.zoom = scale;
    state.panX = (cw - iw * scale) / 2;
    state.panY = (ch - ih * scale) / 2;
    updateZoomLabel();
    render();
  }

  // ── ZOOM ──
  function bindZoom() {
    $('#zoom-in').addEventListener('click', () => zoomBy(1.2));
    $('#zoom-out').addEventListener('click', () => zoomBy(1 / 1.2));
    $('#zoom-fit').addEventListener('click', fitToView);
    container.addEventListener('wheel', (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      zoomAt(factor, mx, my);
    }, { passive: false });
  }

  function zoomBy(factor) {
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    zoomAt(factor, cx, cy);
  }

  function zoomAt(factor, cx, cy) {
    const newZoom = Math.min(Math.max(state.zoom * factor, 0.05), 50);
    const ratio = newZoom / state.zoom;
    state.panX = cx - ratio * (cx - state.panX);
    state.panY = cy - ratio * (cy - state.panY);
    state.zoom = newZoom;
    updateZoomLabel();
    render();
  }

  function updateZoomLabel() {
    zoomLabel.textContent = Math.round(state.zoom * 100) + '%';
  }

  // ── COORDINATE TRANSFORMS ──
  function screenToImage(sx, sy) {
    return [(sx - state.panX) / state.zoom, (sy - state.panY) / state.zoom];
  }

  function imageToScreen(ix, iy) {
    return [ix * state.zoom + state.panX, iy * state.zoom + state.panY];
  }

  // ── CANVAS EVENTS ──
  function bindCanvas() {
    canvas.addEventListener('mousedown', onCanvasDown);
    canvas.addEventListener('mousemove', onCanvasMove);
    canvas.addEventListener('mouseup', onCanvasUp);
    canvas.addEventListener('dblclick', onCanvasDblClick);
    canvas.addEventListener('contextmenu', onCanvasContext);
    document.addEventListener('keydown', onKeyDown);
  }

  function getCanvasXY(e) {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  function onCanvasDown(e) {
    if (e.button === 2) return; // right-click handled elsewhere
    const [sx, sy] = getCanvasXY(e);
    const [ix, iy] = screenToImage(sx, sy);
    closeContextMenu();

    // ── SELECT TOOL ──
    if (state.activeTool === 'select') {
      // Check vertex drag first
      if (state.selectedId !== null) {
        const vt = findVertexAt(sx, sy);
        if (vt) {
          state.draggingVertex = vt;
          return;
        }
      }
      // Check if clicked on an annotation
      const hit = hitTestAnnotation(ix, iy);
      if (hit) {
        selectAnnotation(hit.id);
      } else {
        selectAnnotation(null);
        // Start panning
        state.isPanning = true;
        state.panStart = { x: e.clientX, y: e.clientY, px: state.panX, py: state.panY };
        container.classList.add('dragging');
      }
      return;
    }

    // ── POLYGON TOOL ──
    if (state.activeTool === 'polygon') {
      if (!state.drawing) {
        state.drawing = true;
        state.drawPoints = [[ix, iy]];
        setStatus('Click to add vertices. Double-click to close polygon.');
      } else {
        state.drawPoints.push([ix, iy]);
      }
      render();
      return;
    }

    // ── RECT / CIRCLE / ELLIPSE ──
    if (['rect', 'circle', 'ellipse'].includes(state.activeTool)) {
      state.drawing = true;
      state.drawStart = [ix, iy];
      state.drawCurrent = [ix, iy];
      setStatus('Drag to define shape. Release to finish.');
      return;
    }
  }

  function onCanvasMove(e) {
    const [sx, sy] = getCanvasXY(e);
    const [ix, iy] = screenToImage(sx, sy);

    // Vertex dragging
    if (state.draggingVertex) {
      const ann = state.annotations.find(a => a.id === state.draggingVertex.annId);
      if (ann) {
        ann.points[state.draggingVertex.pointIndex] = [ix, iy];
        render();
      }
      return;
    }

    // Panning
    if (state.isPanning) {
      state.panX = state.panStart.px + (e.clientX - state.panStart.x);
      state.panY = state.panStart.py + (e.clientY - state.panStart.y);
      render();
      return;
    }

    // Drawing shapes
    if (state.drawing && ['rect', 'circle', 'ellipse'].includes(state.activeTool)) {
      state.drawCurrent = [ix, iy];
      render();
      return;
    }

    // Polygon preview line
    if (state.drawing && state.activeTool === 'polygon') {
      state.drawCurrent = [ix, iy];
      render();
      return;
    }

    // Update status with coords
    if (state.image) {
      const imgX = Math.round(ix);
      const imgY = Math.round(iy);
      if (imgX >= 0 && imgX < state.image.width && imgY >= 0 && imgY < state.image.height) {
        setStatus(`Pixel: (${imgX}, ${imgY})  |  Zoom: ${Math.round(state.zoom * 100)}%`);
      }
    }
  }

  function onCanvasUp(e) {
    // End vertex drag
    if (state.draggingVertex) {
      state.draggingVertex = null;
      render();
      return;
    }

    // End panning
    if (state.isPanning) {
      state.isPanning = false;
      container.classList.remove('dragging');
      return;
    }

    // End rect/circle/ellipse drawing
    if (state.drawing && ['rect', 'circle', 'ellipse'].includes(state.activeTool)) {
      const [sx, sy] = getCanvasXY(e);
      state.drawCurrent = screenToImage(sx, sy);
      const pts = shapeToPoints(state.activeTool, state.drawStart, state.drawCurrent);
      if (pts.length >= 3) {
        finishDrawing(state.activeTool, pts);
      } else {
        cancelDrawing();
      }
      return;
    }
  }

  function onCanvasDblClick(e) {
    if (state.activeTool === 'polygon' && state.drawing && state.drawPoints.length >= 3) {
      // Close polygon
      const pts = [...state.drawPoints];
      // close it
      pts.push([...pts[0]]);
      finishDrawing('polygon', pts);
    } else if (state.activeTool === 'select') {
      // Double-click to edit
      if (state.selectedId !== null) {
        openEditModal(state.selectedId);
      }
    }
  }

  function onCanvasContext(e) {
    e.preventDefault();
    const [sx, sy] = getCanvasXY(e);
    const [ix, iy] = screenToImage(sx, sy);
    const hit = hitTestAnnotation(ix, iy);
    if (hit) {
      selectAnnotation(hit.id);
      showContextMenu(e.clientX, e.clientY, hit.id);
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Escape') {
      cancelDrawing();
      closeContextMenu();
      closeModal();
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      if (state.selectedId !== null && !modalOverlay.classList.contains('visible')) {
        deleteAnnotation(state.selectedId);
      }
    }
  }

  // ── SHAPE GENERATION ──
  function shapeToPoints(type, start, end) {
    const [x0, y0] = start;
    const [x1, y1] = end;

    if (type === 'rect') {
      return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]];
    }

    if (type === 'circle') {
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      const r = Math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2) / 2;
      return generateEllipsePoints(cx, cy, r, r, 36);
    }

    if (type === 'ellipse') {
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      const rx = Math.abs(x1 - x0) / 2, ry = Math.abs(y1 - y0) / 2;
      return generateEllipsePoints(cx, cy, rx, ry, 36);
    }

    return [];
  }

  function generateEllipsePoints(cx, cy, rx, ry, n) {
    const pts = [];
    for (let i = 0; i <= n; i++) {
      const a = (2 * Math.PI * i) / n;
      pts.push([cx + rx * Math.cos(a), cy + ry * Math.sin(a)]);
    }
    return pts;
  }

  // ── FINISH / CANCEL DRAWING ──
  function finishDrawing(type, points) {
    state.drawing = false;
    state.drawPoints = [];
    state.drawStart = null;
    state.drawCurrent = null;

    // Store temporarily and open modal
    state._pendingShape = { type, points };
    openNewModal();
    render();
  }

  function cancelDrawing() {
    state.drawing = false;
    state.drawPoints = [];
    state.drawStart = null;
    state.drawCurrent = null;
    state._pendingShape = null;
    render();
  }

  // ── HIT TESTING ──
  function hitTestAnnotation(ix, iy) {
    // Iterate in reverse so top-drawn shapes are tested first
    for (let i = state.annotations.length - 1; i >= 0; i--) {
      const ann = state.annotations[i];
      if (pointInPolygon(ix, iy, ann.points)) {
        return ann;
      }
    }
    return null;
  }

  function pointInPolygon(x, y, pts) {
    let inside = false;
    for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
      const xi = pts[i][0], yi = pts[i][1];
      const xj = pts[j][0], yj = pts[j][1];
      const intersect = ((yi > y) !== (yj > y))
        && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
      if (intersect) inside = !inside;
    }
    return inside;
  }

  function findVertexAt(sx, sy) {
    const ann = state.annotations.find(a => a.id === state.selectedId);
    if (!ann) return null;
    const threshold = 8;
    for (let i = 0; i < ann.points.length; i++) {
      const [vx, vy] = imageToScreen(ann.points[i][0], ann.points[i][1]);
      const dx = sx - vx, dy = sy - vy;
      if (dx * dx + dy * dy < threshold * threshold) {
        return { annId: ann.id, pointIndex: i };
      }
    }
    return null;
  }

  // ── SELECTION ──
  function selectAnnotation(id) {
    state.selectedId = id;
    refreshList();
    render();
    // Scroll list item into view
    if (id !== null) {
      const el = document.querySelector(`.ann-item[data-id="${id}"]`);
      if (el) el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }

  // ── DELETE ──
  function deleteAnnotation(id) {
    state.annotations = state.annotations.filter(a => a.id !== id);
    if (state.selectedId === id) state.selectedId = null;
    refreshList();
    render();
    setStatus('Annotation deleted');
  }

  // ── ANNOTATION LIST ──
  function refreshList() {
    annCount.textContent = state.annotations.length;
    if (state.annotations.length === 0) {
      annList.innerHTML = '<div class="list-empty">No annotations loaded</div>';
      return;
    }
    annList.innerHTML = '';
    state.annotations.forEach(ann => {
      const item = document.createElement('div');
      item.className = 'ann-item' + (ann.id === state.selectedId ? ' selected' : '');
      item.dataset.id = ann.id;
      item.innerHTML = `
        <div class="ann-swatch" style="background:${ann.color}"></div>
        <div class="ann-info">
          <div class="ann-name">${esc(ann.label)}</div>
          <div class="ann-type">${ann.type} · ${ann.points.length} pts</div>
        </div>
        <div class="ann-actions">
          <button class="ann-action-btn edit" title="Edit">✎</button>
          <button class="ann-action-btn del" title="Delete">✕</button>
        </div>
      `;
      item.addEventListener('click', (e) => {
        if (e.target.closest('.ann-action-btn')) return;
        selectAnnotation(ann.id);
      });
      item.querySelector('.edit').addEventListener('click', () => openEditModal(ann.id));
      item.querySelector('.del').addEventListener('click', () => deleteAnnotation(ann.id));
      annList.appendChild(item);
    });
  }

  function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  // ── MODAL ──
  function bindModal() {
    $('#modal-close').addEventListener('click', closeModal);
    $('#modal-cancel').addEventListener('click', closeModal);
    $('#modal-save').addEventListener('click', saveModal);
    $('#add-attr').addEventListener('click', addAttrRow);
    modalOverlay.addEventListener('click', (e) => {
      if (e.target === modalOverlay) closeModal();
    });
  }

  let modalMode = 'new'; // 'new' | 'edit'
  let modalEditId = null;

  function openNewModal() {
    modalMode = 'new';
    modalEditId = null;
    $('#modal-title').textContent = 'New Annotation';
    $('#modal-name').value = '';
    $('#attr-list').innerHTML = '';
    $('#modal-delete').style.display = 'none';
    modalOverlay.classList.add('visible');
    setTimeout(() => $('#modal-name').focus(), 100);
  }

  function openEditModal(id) {
    const ann = state.annotations.find(a => a.id === id);
    if (!ann) return;
    modalMode = 'edit';
    modalEditId = id;
    $('#modal-title').textContent = 'Edit Annotation';
    $('#modal-name').value = ann.label;
    $('#attr-list').innerHTML = '';
    if (ann.attributes) {
      Object.entries(ann.attributes).forEach(([k, v]) => {
        addAttrRow(null, k, String(v));
      });
    }
    $('#modal-delete').style.display = '';
    $('#modal-delete').onclick = () => {
      deleteAnnotation(id);
      closeModal();
    };
    modalOverlay.classList.add('visible');
    setTimeout(() => $('#modal-name').focus(), 100);
  }

  function closeModal() {
    modalOverlay.classList.remove('visible');
    if (modalMode === 'new' && state._pendingShape) {
      state._pendingShape = null;
    }
  }

  function saveModal() {
    const label = $('#modal-name').value.trim() || 'unlabeled';
    const attrs = {};
    document.querySelectorAll('#attr-list .attr-row').forEach(row => {
      const inputs = row.querySelectorAll('input');
      const k = inputs[0].value.trim();
      const v = inputs[1].value.trim();
      if (k) attrs[k] = v;
    });

    if (modalMode === 'new' && state._pendingShape) {
      const ann = {
        id: state.nextId++,
        label: label,
        type: state._pendingShape.type,
        points: state._pendingShape.points,
        color: COLORS[(state.nextId - 2) % COLORS.length],
        attributes: attrs,
      };
      state.annotations.push(ann);
      state._pendingShape = null;
      selectAnnotation(ann.id);
      setStatus('Annotation created: ' + label);
    } else if (modalMode === 'edit' && modalEditId !== null) {
      const ann = state.annotations.find(a => a.id === modalEditId);
      if (ann) {
        ann.label = label;
        ann.attributes = attrs;
        setStatus('Annotation updated: ' + label);
      }
    }

    refreshList();
    render();
    closeModal();
  }

  function addAttrRow(e, key, value) {
    const row = document.createElement('div');
    row.className = 'attr-row';
    row.innerHTML = `
      <input type="text" placeholder="Key" value="${esc(key || '')}" />
      <input type="text" placeholder="Value" value="${esc(value || '')}" />
      <button class="attr-remove" title="Remove">✕</button>
    `;
    row.querySelector('.attr-remove').addEventListener('click', () => row.remove());
    $('#attr-list').appendChild(row);
  }

  // ── CONTEXT MENU ──
  function bindContextMenu() {
    contextMenu.querySelectorAll('button').forEach(btn => {
      btn.addEventListener('click', () => {
        const action = btn.dataset.action;
        const id = parseInt(contextMenu.dataset.annId, 10);
        if (action === 'edit') openEditModal(id);
        if (action === 'delete') deleteAnnotation(id);
        closeContextMenu();
      });
    });
    document.addEventListener('click', (e) => {
      if (!contextMenu.contains(e.target)) closeContextMenu();
    });
  }

  function showContextMenu(x, y, annId) {
    contextMenu.style.left = x + 'px';
    contextMenu.style.top = y + 'px';
    contextMenu.dataset.annId = annId;
    contextMenu.classList.add('visible');
  }

  function closeContextMenu() {
    contextMenu.classList.remove('visible');
  }

  // ── DRAG & DROP ──
  function bindDragDrop() {
    container.addEventListener('dragover', (e) => {
      e.preventDefault();
      container.classList.add('drag-over');
    });
    container.addEventListener('dragleave', () => {
      container.classList.remove('drag-over');
    });
    container.addEventListener('drop', (e) => {
      e.preventDefault();
      container.classList.remove('drag-over');
      const files = e.dataTransfer.files;
      for (const f of files) {
        if (f.type.startsWith('image/')) { loadImageFile(f); break; }
        if (f.name.endsWith('.json')) { loadJSONFile(f); break; }
      }
    });
  }

  // ── EXPORT ──
  function exportJSON() {
    let output;

    if (state.originalJSON) {
      // Rebuild from original structure
      output = JSON.parse(JSON.stringify(state.originalJSON));

      // Group annotations by label
      const groups = {};
      state.annotations.forEach(ann => {
        if (!groups[ann.label]) groups[ann.label] = [];
        groups[ann.label].push(ann);
      });

      // Rebuild Labels array
      output.Labels = [];
      Object.entries(groups).forEach(([label, anns]) => {
        const shapes = anns.map(ann => ({
          vertex_color: 'default',
          line_color: 'default',
          fill_color: 'default',
          Type: ann.type,
          points: ann.points.map(p => [p[0], p[1]]),
          Image_List: state.imageName || 'image.jpg',
          Attributes: { ...ann.attributes },
        }));
        output.Labels.push({
          Label: label,
          Shapes: shapes,
          Status: 'OK',
          Unit_ID: 0,
        });
      });
    } else {
      // Build from scratch
      const groups = {};
      state.annotations.forEach(ann => {
        if (!groups[ann.label]) groups[ann.label] = [];
        groups[ann.label].push(ann);
      });
      output = {
        flags: {},
        Labels: Object.entries(groups).map(([label, anns]) => ({
          Label: label,
          Shapes: anns.map(ann => ({
            vertex_color: 'default',
            line_color: 'default',
            fill_color: 'default',
            Type: ann.type,
            points: ann.points.map(p => [p[0], p[1]]),
            Image_List: state.imageName || 'image.jpg',
            Attributes: { ...ann.attributes },
          })),
          Status: 'OK',
          Unit_ID: 0,
        })),
      };
    }

    const blob = new Blob([JSON.stringify(output, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'annotations_export.json';
    a.click();
    URL.revokeObjectURL(url);
    setStatus('Exported ' + state.annotations.length + ' annotations');
  }

  // ── RENDERING ──
  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!state.image) return;

    ctx.save();
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.zoom, state.zoom);

    // Draw image
    ctx.drawImage(state.image, 0, 0);

    // Draw annotations
    state.annotations.forEach(ann => {
      drawAnnotation(ann, ann.id === state.selectedId);
    });

    // Draw in-progress shape
    if (state.drawing) {
      drawInProgress();
    }

    ctx.restore();
  }

  function drawAnnotation(ann, selected) {
    if (ann.points.length < 2) return;
    const pts = ann.points;

    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i][0], pts[i][1]);
    }
    ctx.closePath();

    // Fill
    const baseColor = ann.color;
    if (selected) {
      ctx.fillStyle = hexToRgba(baseColor, 0.35);
    } else {
      ctx.fillStyle = hexToRgba(baseColor, 0.18);
    }
    ctx.fill();

    // Stroke
    ctx.lineWidth = selected ? 2.5 / state.zoom : 1.5 / state.zoom;
    if (selected) {
      ctx.strokeStyle = '#4af0a0';
      ctx.setLineDash([]);
    } else {
      ctx.strokeStyle = hexToRgba(baseColor, 0.85);
      ctx.setLineDash([]);
    }
    ctx.stroke();

    // Vertex handles for selected
    if (selected) {
      const handleR = 4 / state.zoom;
      pts.forEach(p => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], handleR, 0, Math.PI * 2);
        ctx.fillStyle = '#0e0f11';
        ctx.fill();
        ctx.strokeStyle = '#4af0a0';
        ctx.lineWidth = 1.5 / state.zoom;
        ctx.stroke();
      });
    }
  }

  function drawInProgress() {
    ctx.setLineDash([6 / state.zoom, 4 / state.zoom]);
    ctx.lineWidth = 1.5 / state.zoom;
    ctx.strokeStyle = '#f0c94a';
    ctx.fillStyle = 'rgba(240, 201, 74, 0.12)';

    if (state.activeTool === 'polygon' && state.drawPoints.length > 0) {
      ctx.beginPath();
      ctx.moveTo(state.drawPoints[0][0], state.drawPoints[0][1]);
      for (let i = 1; i < state.drawPoints.length; i++) {
        ctx.lineTo(state.drawPoints[i][0], state.drawPoints[i][1]);
      }
      if (state.drawCurrent) {
        ctx.lineTo(state.drawCurrent[0], state.drawCurrent[1]);
      }
      ctx.stroke();

      // Vertex dots
      const r = 4 / state.zoom;
      state.drawPoints.forEach(p => {
        ctx.beginPath();
        ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
        ctx.fillStyle = '#f0c94a';
        ctx.fill();
      });
    }

    if (['rect', 'circle', 'ellipse'].includes(state.activeTool) && state.drawStart && state.drawCurrent) {
      const pts = shapeToPoints(state.activeTool, state.drawStart, state.drawCurrent);
      if (pts.length >= 3) {
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i][0], pts[i][1]);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    }

    ctx.setLineDash([]);
  }

  function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // ── STATUS BAR ──
  function setStatus(msg) {
    statusBar.textContent = msg;
  }

  // ── GO ──
  init();

})();
