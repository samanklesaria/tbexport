from __future__ import annotations

import os
import json
from html import escape
from typing import Optional, TypedDict
import fire
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt


STYLES = """\
<style>
    .item-container { margin-bottom: 40px; padding: 20px; border: 1px solid #eee; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .item-container img { max-width: 100%; height: auto; display: block; margin-top: 15px; border-radius: 4px; }
    .item-container h3 { margin-top: 0; color: #1a73e8; font-size: 1.2rem; margin-bottom: 5px; }
    .item-container .meta { color: #666; font-size: 0.85rem; margin-bottom: 15px; }
    .scrubber-control { background: #f8f9fa; border: 1px solid #e0e0e0; padding: 15px; border-radius: 6px; margin-top: 15px; }
    .scrubber-label { font-size: 0.9rem; font-weight: bold; display: flex; justify-content: space-between; margin-bottom: 8px; }
    .step-slider { width: 100%; margin: 10px 0; cursor: pointer; }
    .item-container audio { display: block; width: 100%; margin-top: 10px; }
    .item-container table { width: 100%; border-collapse: collapse; }
    .item-container td { vertical-align: top; padding: 8px; }
    .item-container tr { background: none !important; }
</style>
"""

SCRUBBER_JS = """\
(function() {
    function init() {
        document.querySelectorAll('.scrubber-control[data-audio-steps]').forEach(function(el) {
            // data-initialized guard prevents double-binding when init() is called
            // on dynamically inserted content after DOMContentLoaded
            if (el.dataset.initialized) return;
            el.dataset.initialized = '1';
            var steps = JSON.parse(el.dataset.audioSteps);
            var pathTemplate = el.dataset.audioPath;
            var slider = el.querySelector('.step-slider');
            var display = el.querySelector('.step-display');
            var player = el.querySelector('audio');
            slider.addEventListener('input', function(e) {
                var step = steps[parseInt(e.target.value)];
                display.textContent = step;
                var wasPlaying = !player.paused;
                player.src = pathTemplate.replace('{step}', step);
                player.load();
                if (wasPlaying) player.play();
            });
        });
        document.querySelectorAll('[data-image-steps]').forEach(function(el) {
            if (el.dataset.initialized) return;
            el.dataset.initialized = '1';
            var steps = JSON.parse(el.dataset.imageSteps);
            var slider = el.querySelector('.step-slider');
            var display = el.querySelector('.step-display');
            var img = el.querySelector('img');
            slider.addEventListener('input', function(e) {
                var s = steps[parseInt(e.target.value)];
                display.textContent = s.step;
                img.src = s.src;
            });
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
"""


class StepEntry(TypedDict):
    step: int
    src: str  # relative path from HTML file to the asset


def _to_list(val: Optional[str | list[str]]) -> list[str]:
    "Normalize summary tags provided to the CLI"
    return [val] if isinstance(val, str) else (val or [])


def _expand_groups(
    requested: list[str], available: list[str]
) -> list[tuple[str, list[str]]]:
    """Map each requested tag to itself (exact match) or its children (prefix match)."""
    groups: list[tuple[str, list[str]]] = []
    for t in requested:
        if t in available:
            groups.append((t, [t]))
        else:
            matches = sorted(a for a in available if a.startswith(t + '/'))
            groups.append((t, matches if matches else [t]))
    return groups


def _item(title: str, meta: str, body: str) -> str:
    return (
        f'<div class="item-container">'
        f'<h3>{title}</h3>'
        f'<div class="meta">{meta}</div>'
        f'{body}'
        f'</div>\n'
    )


def _table(label: str, kind: str, cells: list[str]) -> str:
    rows = []
    for i in range(0, len(cells), 3):
        chunk = cells[i:i+3]
        tds = ''.join(f'<td style="width:33%">{c}</td>' for c in chunk)
        tds += '<td></td>' * (3 - len(chunk))
        rows.append(f'<tr>{tds}</tr>')
    header = _compact_header(f"{kind} Summary: {label}", f"{len(cells)} components")
    return f'<div class="item-container">{header}<table>{"".join(rows)}</table></div>\n'


def _sublabel(tag: str) -> str:
    return f'<div class="meta">{tag.split("/")[-1]}</div>'


def _compact_header(left: str, right: str) -> str:
    return (f'<div class="meta" style="display:flex;justify-content:space-between">'
            f'<span>{left}</span><span>{right}</span></div>')


def _audio_scrubber(path_template: str, steps: list[int]) -> str:
    initial_src = path_template.replace('{step}', str(steps[0]))
    return (
        f'<div class="scrubber-control"'
        f' data-audio-steps="{escape(json.dumps(steps))}"'
        f' data-audio-path="{escape(path_template)}">'
        f'<div class="scrubber-label">'
        f'<span>Step Scrubber</span>'
        f'<span><strong class="step-display">{steps[0]}</strong></span>'
        f'</div>'
        f'<input type="range" class="step-slider" min="0" max="{len(steps)-1}" value="0" step="1" />'
        f'<audio controls src="{initial_src}" type="audio/wav"></audio>'
        f'</div>'
    )


def _image_scrubber(tag: str, steps_data: list[StepEntry]) -> str:
    initial = steps_data[0]
    return (
        f'<div data-image-steps="{escape(json.dumps(steps_data))}">'
        f'<div class="scrubber-control">'
        f'<div class="scrubber-label">'
        f'<span>Step Scrubber</span>'
        f'<span><strong class="step-display">{initial["step"]}</strong></span>'
        f'</div>'
        f'<input type="range" class="step-slider" min="0" max="{len(steps_data)-1}" value="0" step="1" />'
        f'</div>'
        f'<img src="{initial["src"]}" alt="{tag}" />'
        f'</div>'
    )


def _audio_html(tag: str, path_template: str, steps: list[int]) -> str:
    if len(steps) == 1:
        src = path_template.replace('{step}', str(steps[0]))
        header = _compact_header(tag, f"step {steps[0]}")
        body = f'<audio controls><source src="{src}" type="audio/wav"></audio>'
    else:
        header = _compact_header(tag, f"Contains {len(steps)} steps")
        body = _audio_scrubber(path_template, steps)
    return f'<div class="item-container">{header}{body}</div>\n'


def _audio_cell(tag: str, path_template: str, steps: list[int]) -> str:
    sublabel = tag.split('/')[-1]
    if len(steps) == 1:
        src = path_template.replace('{step}', str(steps[0]))
        return (_compact_header(sublabel, f"step {steps[0]}") +
                f'<audio controls><source src="{src}" type="audio/wav"></audio>')
    return _compact_header(sublabel, '') + _audio_scrubber(path_template, steps)


def _image_html(tag: str, steps_data: list[StepEntry]) -> str:
    if len(steps_data) == 1:
        ev = steps_data[0]
        return _item(f"Image Summary: {tag}", f"step {ev['step']}",
                     f'<img src="{ev["src"]}" alt="{tag}" />')
    return _item(f"Image Summary: {tag}", f"Contains {len(steps_data)} steps",
                 _image_scrubber(tag, steps_data))


def _image_cell(tag: str, steps_data: list[StepEntry]) -> str:
    if len(steps_data) == 1:
        ev = steps_data[0]
        return _sublabel(tag) + f'<img src="{ev["src"]}" alt="{tag}" />'
    return _sublabel(tag) + _image_scrubber(tag, steps_data)


def extract(
    log_dir: str,
    scalars: Optional[str | list[str]] = None,
    images: Optional[str | list[str]] = None,
    audio: Optional[str | list[str]] = None,
) -> Optional[str]:
    """Parses a TensorBoard log directory and extracts requested tags into static HTML.

    Args:
        log_dir: Path to the directory containing tfevents files.
        scalars: Tag name or list of scalar tags to plot.
        images: Tag name or list of image tags to extract.
        audio: Tag name or list of audio tags to extract.
    """
    if not os.path.exists(log_dir):
        return f"Error: The log directory '{log_dir}' does not exist."

    target_scalars = _to_list(scalars)
    target_images = _to_list(images)
    target_audio = _to_list(audio)

    if not any([target_scalars, target_images, target_audio]):
        return "Warning: No tags specified. Use --scalars, --images, or --audio."

    print(f"Loading events from {log_dir}...")
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.IMAGES: 0,
            event_accumulator.AUDIO: 0,
        }
    )
    ea.Reload()
    tags = ea.Tags()

    with open("scrubber.js", "w", encoding="utf-8") as f:
        f.write(SCRUBBER_JS)

    scalar_groups = _expand_groups(target_scalars, tags['scalars'])
    image_groups = _expand_groups(target_images, tags['images'])
    audio_groups = _expand_groups(target_audio, tags['audio'])

    # Collect all labels in order, deduplicating while preserving first occurrence
    seen: set[str] = set()
    all_labels: list[str] = []
    for label, _ in scalar_groups + image_groups + audio_groups:
        if label not in seen:
            seen.add(label)
            all_labels.append(label)

    scalar_map = dict(scalar_groups)
    image_map = dict(image_groups)
    audio_map = dict(audio_groups)

    for label in all_labels:
        slug = label.replace('/', '_')
        output = slug + '.html'
        assets_dirname = slug + '_assets'
        os.makedirs(assets_dirname, exist_ok=True)

        sections: list[str] = []

        group = scalar_map.get(label, [])
        cells: list[tuple[str, str, float, int]] = []
        for tag in group:
            if tag not in tags['scalars']:
                print(f"Warning: Scalar tag '{tag}' not found.")
                continue
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]

            plt.figure(figsize=(7, 4), dpi=150)
            plt.plot(steps, values, color='#1a73e8', linewidth=2)
            plt.title(f"Metric: {tag}", fontsize=12, fontweight='bold', pad=10)
            plt.xlabel("Step", fontsize=10)
            plt.ylabel("Value", fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            filename = f"scalar_{tag.replace('/', '_')}.png"
            plt.savefig(os.path.join(assets_dirname, filename))
            plt.close()

            cells.append((tag, f"{assets_dirname}/{filename}", values[-1], steps[-1]))

        if cells:
            if len(cells) == 1:
                tag, src, value, step = cells[0]
                sections.append(_item(f"Scalar Trend: {tag}",
                                      f"Final Value: {value:.4f} at Step {step}",
                                      f'<img src="{src}" alt="{tag} plot" />'))
            else:
                cell_htmls = [
                    f'{_sublabel(tag)}<div class="meta">{value:.4f} @ step {step}</div>'
                    f'<img src="{src}" alt="{tag} plot" />'
                    for tag, src, value, step in cells
                ]
                sections.append(_table(label, "Scalar", cell_htmls))

        group = image_map.get(label, [])
        image_cells: list[tuple[str, list[StepEntry]]] = []
        for tag in group:
            if tag not in tags['images']:
                print(f"Warning: Image tag '{tag}' not found.")
                continue
            steps_data: list[StepEntry] = []
            for event in ea.Images(tag):
                filename = f"image_{tag.replace('/', '_')}_step{event.step}.png"
                with open(os.path.join(assets_dirname, filename), "wb") as f:
                    f.write(event.encoded_image_string)
                steps_data.append({"step": event.step, "src": f"{assets_dirname}/{filename}"})
            image_cells.append((tag, steps_data))

        if image_cells:
            if len(image_cells) == 1:
                tag, steps_data = image_cells[0]
                sections.append(_image_html(tag, steps_data))
            else:
                sections.append(_table(label, "Image",
                                       [_image_cell(tag, sd) for tag, sd in image_cells]))

        group = audio_map.get(label, [])
        audio_cells: list[tuple[str, str, list[int]]] = []
        for tag in group:
            if tag not in tags['audio']:
                print(f"Warning: Audio tag '{tag}' not found.")
                continue
            slug = tag.replace('/', '_')
            path_template = f"{assets_dirname}/audio_{slug}_step{{step}}.wav"
            steps: list[int] = []
            for event in ea.Audio(tag):
                filename = f"audio_{slug}_step{event.step}.wav"
                with open(os.path.join(assets_dirname, filename), "wb") as f:
                    f.write(event.encoded_audio_string)
                steps.append(event.step)
            audio_cells.append((tag, path_template, steps))

        if audio_cells:
            if len(audio_cells) == 1:
                tag, path_template, steps = audio_cells[0]
                sections.append(_audio_html(tag, path_template, steps))
            else:
                sections.append(_table(label, "Audio",
                                       [_audio_cell(tag, pt, st) for tag, pt, st in audio_cells]))

        with open(output, "w", encoding="utf-8") as f:
            f.write(STYLES + "".join(sections))

        print(f"  {output}  (assets: {assets_dirname}/)")

    return None

def main():
    fire.Fire(extract)
