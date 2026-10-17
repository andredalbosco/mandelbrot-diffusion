"""Browse saved Mandelbrot tensors, paired zoom inputs, and rendering metadata.

uv run python dataset_viewwer.py --dataset data/zoom4x-main-single
"""

import argparse
import json
from pathlib import Path
import sys

import torch
from PySide6 import QtCore, QtGui, QtWidgets


def pixmap(image):
    """Copy a CHW float tensor into a Qt-owned RGB image."""
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    pixels = (image.detach().cpu().clamp(0, 1) * 255).round().byte().permute(1, 2, 0).contiguous().numpy()
    height, width = pixels.shape[:2]
    owned = QtGui.QImage(pixels.data, width, height, pixels.strides[0], QtGui.QImage.Format.Format_RGB888).copy()
    return QtGui.QPixmap.fromImage(owned)


def read_split(path):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    keys = ("inputs", "targets") if "inputs" in payload and "targets" in payload else ("images",)
    for key in keys:
        images = payload[key]
        if images.ndim != 4 or images.shape[1] not in (1, 3) or len(images) == 0:
            raise ValueError(f"{key} must be a nonempty [N,1|3,H,W] tensor")
    if len(keys) == 2 and payload["inputs"].shape != payload["targets"].shape:
        raise ValueError("Paired input and target shapes differ")
    return payload, keys


class ImagePanel(QtWidgets.QWidget):
    def __init__(self, title):
        super().__init__()
        self.title = title
        self.pixels = QtGui.QPixmap()
        self.smooth = False
        self.setMinimumSize(160, 160)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#17191c"))
        painter.setPen(QtGui.QColor("#e7e8eb"))
        painter.drawText(self.rect().adjusted(12, 4, -12, 0), QtCore.Qt.AlignmentFlag.AlignTop, self.title)
        if self.pixels.isNull():
            return
        area = self.rect().adjusted(10, 32, -10, -10)
        size = self.pixels.size().scaled(area.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        target = QtCore.QRect(QtCore.QPoint(0, 0), size)
        target.moveCenter(area.center())
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, self.smooth)
        painter.drawPixmap(target, self.pixels)

    def set_image(self, image, title=None):
        self.pixels = pixmap(image)
        if title:
            self.title = title
        self.update()


class DatasetViewer(QtWidgets.QMainWindow):
    page_size = 24

    def __init__(self, dataset, split="train", index=0):
        super().__init__()
        self.setWindowTitle("Mandelbrot · Dataset viewer")
        self.resize(1400, 980)
        self.payload, self.keys, self.manifest = None, (), {}
        self.page = -1
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        toolbar = QtWidgets.QHBoxLayout()
        self.datasets = QtWidgets.QComboBox()
        self.datasets.setMinimumContentsLength(24)
        self.datasets.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.splits = QtWidgets.QComboBox()
        self.splits.addItems(["train", "val"])
        self.splits.setCurrentText(split)
        open_button = QtWidgets.QPushButton("Open folder…")
        toolbar.addWidget(QtWidgets.QLabel("Dataset"))
        toolbar.addWidget(self.datasets, 1)
        toolbar.addWidget(self.splits)
        toolbar.addWidget(open_button)
        layout.addLayout(toolbar)
        navigation = QtWidgets.QHBoxLayout()
        previous = QtWidgets.QPushButton("← Previous")
        following = QtWidgets.QPushButton("Next →")
        self.index = QtWidgets.QSpinBox()
        self.index.setMinimum(1)
        self.count = QtWidgets.QLabel()
        self.smoothing = QtWidgets.QCheckBox("Smooth display scaling")
        self.difference = QtWidgets.QCheckBox("Show absolute difference ×4")
        for widget in (previous, self.index, self.count, following, self.smoothing, self.difference):
            navigation.addWidget(widget)
        navigation.addStretch()
        layout.addLayout(navigation)
        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        panels = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.input = ImagePanel("Image")
        self.target = ImagePanel("True rendered target")
        self.error = ImagePanel("Absolute difference ×4")
        for panel in (self.input, self.target, self.error):
            panels.addWidget(panel)
        panels.setChildrenCollapsible(False)
        self.panels = panels
        layout.addWidget(panels, 3)
        lower = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.thumbnails = QtWidgets.QListWidget()
        self.thumbnails.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.thumbnails.setResizeMode(QtWidgets.QListView.ResizeMode.Adjust)
        self.thumbnails.setMovement(QtWidgets.QListView.Movement.Static)
        self.thumbnails.setIconSize(QtCore.QSize(88, 88))
        self.thumbnails.setGridSize(QtCore.QSize(110, 114))
        self.metadata = QtWidgets.QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.metadata.setPlaceholderText("Rendering metadata")
        lower.addWidget(self.thumbnails)
        lower.addWidget(self.metadata)
        lower.setStretchFactor(0, 2)
        lower.setStretchFactor(1, 1)
        layout.addWidget(lower, 2)
        page_controls = QtWidgets.QHBoxLayout()
        previous_page = QtWidgets.QPushButton("← Previous page")
        next_page = QtWidgets.QPushButton("Next page →")
        self.page_label = QtWidgets.QLabel()
        for widget in (previous_page, self.page_label, next_page):
            page_controls.addWidget(widget)
        page_controls.addStretch()
        layout.addLayout(page_controls)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Arrow keys: previous/next image · thumbnails show targets for paired datasets")

        candidates = sorted(p for p in (Path(__file__).resolve().parent / "data").iterdir()
                            if p.is_dir() and any((p / f"{s}.pt").exists() for s in ("train", "val")))
        dataset = dataset.resolve()
        if dataset not in candidates:
            candidates.append(dataset)
        for candidate in candidates:
            self.datasets.addItem(candidate.name, str(candidate))
        self.datasets.setCurrentIndex(candidates.index(dataset))
        self.datasets.currentIndexChanged.connect(self.load_dataset)
        self.splits.currentTextChanged.connect(self.load_dataset)
        self.index.valueChanged.connect(self.show_sample)
        self.thumbnails.currentItemChanged.connect(self.select_thumbnail)
        self.smoothing.toggled.connect(self.show_sample)
        self.difference.toggled.connect(self.show_sample)
        open_button.clicked.connect(self.open_folder)
        previous.clicked.connect(lambda: self.advance(-1))
        following.clicked.connect(lambda: self.advance(1))
        previous_page.clicked.connect(lambda: self.advance(-self.page_size))
        next_page.clicked.connect(lambda: self.advance(self.page_size))
        for key, delta in (("Left", -1), ("Right", 1)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(key), self)
            shortcut.activated.connect(lambda d=delta: self.advance(d))
        self.load_dataset()
        self.index.setValue(index + 1)

    def load_dataset(self):
        directory = Path(self.datasets.currentData())
        self.page = -1
        try:
            self.payload, self.keys = read_split(directory / f"{self.splits.currentText()}.pt")
            manifest = directory / "manifest.json"
            self.manifest = json.loads(manifest.read_text()) if manifest.exists() else {}
        except (OSError, RuntimeError, ValueError, KeyError) as error:
            self.payload = None
            self.summary.setText(f"Cannot open dataset: {error}")
            self.thumbnails.clear()
            self.metadata.clear()
            for panel in (self.input, self.target, self.error):
                panel.pixels = QtGui.QPixmap()
                panel.update()
            return
        count = len(self.payload[self.keys[0]])
        with QtCore.QSignalBlocker(self.index):
            self.index.setRange(1, count)
            self.index.setValue(1)
        self.count.setText(f"/ {count:,}")
        self.show_sample()

    def show_sample(self):
        if self.payload is None:
            return
        index = self.index.value() - 1
        paired = len(self.keys) == 2
        sample = self.payload[self.keys[0]][index]
        target = self.payload[self.keys[-1]][index]
        for image in (sample, target):
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                self.summary.setText("Invalid sample: expected finite pixel values in [0,1]")
                return
        self.input.set_image(sample, "Upsampled input" if paired else "Rendered image")
        self.target.setVisible(paired)
        self.error.setVisible(paired and self.difference.isChecked())
        self.difference.setEnabled(paired)
        if paired:
            self.target.set_image(target)
            self.error.set_image((target - sample).abs() * 4)
        for panel in (self.input, self.target, self.error):
            panel.smooth = self.smoothing.isChecked()
            panel.update()
        channels, height, width = sample.shape
        label = "Paired zooms" if paired else "Unconditional images"
        summary = f"{label} · {width} × {height} · {'RGB' if channels == 3 else 'grayscale'} · {sample.dtype}"
        if paired:
            mse = (sample - target).square().mean().item()
            summary += f" · input/target MSE {mse:.6f}"
        self.summary.setText(summary)
        records = self.payload.get("records", [])
        details = {"dataset": self.datasets.currentData(), "split": self.splits.currentText(),
                   "sample_index": index, "shape_CHW": list(sample.shape),
                   "pixel_range": [sample.min().item(), sample.max().item()]}
        if len(records) > index:
            details["sample"] = records[index]
        if "masks" in self.payload:
            details["mask_shape"] = list(self.payload["masks"][index].shape)
        else:
            details["coverage_masks"] = "Not stored in this dataset"
        details["manifest"] = {k: v for k, v in self.manifest.items() if k not in ("regions", "quality")}
        self.metadata.setPlainText(json.dumps(details, indent=2))
        page = index // self.page_size
        with QtCore.QSignalBlocker(self.thumbnails):
            if page != self.page:
                self.page = page
                self.thumbnails.clear()
                images = self.payload[self.keys[-1]]
                for i in range(page * self.page_size, min((page + 1) * self.page_size, len(images))):
                    item = QtWidgets.QListWidgetItem(QtGui.QIcon(pixmap(images[i])), str(i + 1))
                    item.setData(QtCore.Qt.ItemDataRole.UserRole, i)
                    self.thumbnails.addItem(item)
            self.thumbnails.setCurrentRow(index % self.page_size)
        count = len(self.payload[self.keys[0]])
        self.page_label.setText(f"Page {page + 1} / {(count + self.page_size - 1) // self.page_size}")

    def advance(self, delta):
        self.index.setValue(self.index.value() + delta)

    def select_thumbnail(self, current, previous):
        if current:
            self.index.setValue(current.data(QtCore.Qt.ItemDataRole.UserRole) + 1)

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Open dataset folder", self.datasets.currentData())
        if not folder:
            return
        position = self.datasets.findData(folder)
        if position < 0:
            self.datasets.addItem(Path(folder).name, folder)
            position = self.datasets.count() - 1
        self.datasets.setCurrentIndex(position)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).resolve().parent / "data/zoom4x-main-single")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--index", type=int, default=0, help="Zero-based sample index")
    parser.add_argument("--screenshot", type=Path, help="Save a viewer screenshot and exit")
    args = parser.parse_args()
    if args.index < 0 or not args.dataset.is_dir():
        parser.error("Need a dataset directory and nonnegative index")
    torch.set_num_threads(4)
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("Mandelbrot dataset viewer")
    window = DatasetViewer(args.dataset, args.split, args.index)
    if window.payload is None:
        parser.error(window.summary.text())
    if args.screenshot:
        window.show()

        def capture():
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(args.screenshot)):
                raise RuntimeError("Could not save screenshot")
            app.quit()

        QtCore.QTimer.singleShot(200, capture)
    else:
        window.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
