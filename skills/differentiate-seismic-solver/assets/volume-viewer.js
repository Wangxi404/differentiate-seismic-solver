/*
MIT License
Copyright (c) 2026 Sid

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
(() => {
    "use strict";
    const data = JSON.parse(document.getElementById("volume-data").textContent);
    const view = document.getElementById("three-d-models");
    const slider = document.getElementById("threshold");
    const markers = document.getElementById("show-acquisition");
    const canvases = [...view.querySelectorAll("canvas")];
    const axes = data.axes;
    const size = axes.map(axis => axis.length);
    const strides = [size[1] * size[2], size[2], 1];
    const edges = axes.map(axis => [
        axis[0], ...axis.slice(1).map((v, i) => (v + axis[i]) / 2), axis[axis.length - 1]
    ]);
    const bounds = axes.map(axis => [axis[0], axis[axis.length - 1]]);
    for (const points of Object.values(data.acquisition)) {
        for (const p of points) {
            for (let a = 0; a < 3; a++) {
                bounds[a][0] = Math.min(bounds[a][0], p[a]);
                bounds[a][1] = Math.max(bounds[a][1], p[a]);
            }
        }
    }
    const center = bounds.map(([lo, hi]) => (lo + hi) / 2);
    const diagonal = Math.hypot(...bounds.map(([lo, hi]) => hi - lo));
    const names = ["X", "Y", "Depth"];
    const format = number => Number(number.toPrecision(5)).toString();
    document.getElementById("volume-extents").textContent = axes.map((axis, a) =>
        names[a] + " " + format(axis[0]) + "–" + format(axis[axis.length - 1]) + " " + data.coordinate_unit
    ).join(" · ");

    let azimuth = -0.93, elevation = 0.48, zoom = 1, frame = 0;
    const limit = (value, lo, hi) => Math.min(hi, Math.max(lo, value));

    function draw(canvas, values) {
        const width = canvas.clientWidth, height = canvas.clientHeight;
        if (!width || !height) return;
        const pixelRatio = window.devicePixelRatio || 1;
        const w = Math.round(width * pixelRatio), h = Math.round(height * pixelRatio);
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
        const ctx = canvas.getContext("2d");
        ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        ctx.clearRect(0, 0, width, height);
        const ca = Math.cos(azimuth), sa = Math.sin(azimuth);
        const ce = Math.cos(elevation), se = Math.sin(elevation);
        const eye = [ce * ca, ce * sa, -se];
        const sign = eye.map(c => c >= 0 ? 1 : -1);
        const scale = 0.87 * Math.min(width, height) * zoom / diagonal;
        const right = [-sa, ca, 0].map(v => v * scale);
        const down = [se * ca, se * sa, ce].map(v => v * scale);
        function project(point) {
            let x = width / 2, y = height / 2;
            for (let a = 0; a < 3; a++) {
                x += (point[a] - center[a]) * right[a];
                y += (point[a] - center[a]) * down[a];
            }
            return [x, y];
        }
        const steps = edges.map((axis, a) => axis.slice(1).map((v, i) => [
            (v - axis[i]) * right[a], (v - axis[i]) * down[a]
        ]));
        const starts = sign.map((s, a) => s === 1 ? 0 : size[a] - 1);
        const low = data.limits[0], high = data.limits[1], threshold = Number(slider.value);
        const present = index => values[index] !== null && values[index] >= threshold;

        let visible = false;
        // A ray toward the eye is monotone in each coordinate. Traversing cells
        // from far to near orders their exposed faces on a rectilinear grid.
        for (let i = starts[0]; i >= 0 && i < size[0]; i += sign[0]) {
            for (let j = starts[1]; j >= 0 && j < size[1]; j += sign[1]) {
                for (let k = starts[2]; k >= 0 && k < size[2]; k += sign[2]) {
                    const index = i * strides[0] + j * strides[1] + k;
                    if (!present(index)) continue;
                    const cell = [i, j, k];
                    let origin;
                    for (let a = 0; a < 3; a++) {
                        if (Math.abs(eye[a]) < 1e-8) continue;
                        const neighbor = cell[a] + sign[a];
                        if (neighbor >= 0 && neighbor < size[a] && present(index + sign[a] * strides[a])) continue;
                        if (!origin) {
                            visible = true;
                            origin = project([edges[0][i], edges[1][j], edges[2][k]]);
                            const color = Math.floor(data.palette.length * (values[index] - low) / (high - low));
                            ctx.fillStyle = data.palette[limit(color, 0, data.palette.length - 1)];
                            ctx.strokeStyle = ctx.fillStyle;
                            ctx.lineWidth = 0.45;
                        }
                        const u = (a + 1) % 3, v = (a + 2) % 3;
                        const du = steps[u][cell[u]], dv = steps[v][cell[v]];
                        const shift = sign[a] === 1 ? steps[a][cell[a]] : [0, 0];
                        const x = origin[0] + shift[0], y = origin[1] + shift[1];
                        ctx.beginPath();
                        ctx.moveTo(x, y);
                        ctx.lineTo(x + du[0], y + du[1]);
                        ctx.lineTo(x + du[0] + dv[0], y + du[1] + dv[1]);
                        ctx.lineTo(x + dv[0], y + dv[1]);
                        ctx.closePath();
                        ctx.fill();
                        ctx.stroke();
                    }
                }
            }
        }

        if (!visible) {
            ctx.fillStyle = "#526679";
            ctx.font = "12px system-ui";
            ctx.textAlign = "center";
            ctx.fillText("No cells at this threshold", width / 2, 18);
        }
        if (markers && markers.checked) {
            for (const name of ["receivers", "sources"]) {
                const source = name === "sources";
                ctx.fillStyle = source ? "#b02e2b" : "#202830";
                for (const point of data.acquisition[name] || []) {
                    const [x, y] = project(point);
                    const n = source ? 10 : 3;
                    ctx.beginPath();
                    for (let i = 0; i < n; i++) {
                        const theta = 2 * Math.PI * i / n - Math.PI / 2;
                        const radius = source ? (i % 2 ? 2 : 5) : 2.7;
                        const px = x + radius * Math.cos(theta), py = y + radius * Math.sin(theta);
                        if (i === 0) ctx.moveTo(px, py);
                        else ctx.lineTo(px, py);
                    }
                    ctx.closePath();
                    ctx.fill();
                }
            }
        }
        // Orientation key points along increasing physical coordinates.
        ctx.strokeStyle = "#34495d";
        ctx.fillStyle = "#34495d";
        ctx.font = "11px system-ui";
        ctx.lineWidth = 1;
        for (let a = 0; a < 3; a++) {
            const dx = right[a] * 18 / scale, dy = down[a] * 18 / scale;
            const x = 32, y = height - 24;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x + dx, y + dy);
            ctx.stroke();
            ctx.textAlign = dx < -2 ? "right" : "left";
            ctx.fillText(["X", "Y", "Z"][a], x + dx * 1.18, y + dy * 1.18 + 3);
        }
    }

    function refresh() {
        document.getElementById("threshold-value").textContent = format(Number(slider.value));
        if (view.open && !frame) frame = requestAnimationFrame(() => {
            frame = 0;
            for (const canvas of canvases) draw(canvas, data.fields[canvas.dataset.field]);
        });
    }
    function rotate(dx, dy) {
        azimuth += dx;
        elevation = limit(elevation + dy, -1.45, 1.45);
        refresh();
    }
    for (const canvas of canvases) {
        let drag;
        canvas.onpointerdown = event => {
            canvas.setPointerCapture(event.pointerId);
            drag = [event.clientX, event.clientY];
        };
        canvas.onpointermove = event => {
            if (!drag) return;
            rotate(-(event.clientX - drag[0]) * 0.009, (event.clientY - drag[1]) * 0.009);
            drag = [event.clientX, event.clientY];
        };
        canvas.onlostpointercapture = () => { drag = null; };
        canvas.onkeydown = event => {
            const keys = {ArrowLeft: [0.12, 0], ArrowRight: [-0.12, 0], ArrowUp: [0, 0.12], ArrowDown: [0, -0.12]};
            if (keys[event.key]) {
                event.preventDefault();
                rotate(...keys[event.key]);
            }
        };
        canvas.addEventListener("wheel", event => {
            event.preventDefault();
            zoom = limit(zoom * Math.exp(-event.deltaY * 0.001), 0.45, 3);
            refresh();
        }, {passive: false});
    }
    document.getElementById("reset-view").onclick = () => {
        azimuth = -0.93;
        elevation = 0.48;
        zoom = 1;
        slider.value = slider.min;
        if (markers) markers.checked = true;
        refresh();
    };
    slider.oninput = refresh;
    if (markers) markers.onchange = refresh;
    view.addEventListener("toggle", refresh);
    new ResizeObserver(refresh).observe(document.getElementById("model-volumes"));
    refresh();
})();
