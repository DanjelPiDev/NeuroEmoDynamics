DARK_MODE_STYLE = """
    QWidget {
        background-color: #2e2e2e;
        color: #ffffff;
    }
    QLineEdit, QSpinBox, QComboBox, QTextEdit, QPlainTextEdit {
        background-color: #3e3e3e;
        color: #ffffff;
        border: 1px solid #5a5a5a;
        border-radius: 4px;
        padding: 4px;
    }
    QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled, QTextEdit:disabled, QPlainTextEdit:disabled {
        background-color: #968686;
    }
    QPushButton {
        background-color: #5a5a5a;
        color: #ffffff;
        border: 1px solid #7a7a7a;
        border-radius: 4px;
        padding: 6px;
    }
    QPushButton:hover {
        background-color: #6a6a6a;
    }
    QPushButton:disabled {
        background-color: #968686;
    }
    QProgressBar {
        border: 1px solid #5a5a5a;
        text-align: center;
        color: #ffffff;
        background-color: #3e3e3e;
    }
    QProgressBar::chunk {
        background-color: #7a7a7a;
    }
    QGroupBox {
        border: 1px solid #5a5a5a;
        border-radius: 6px;
        margin-top: 6px;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 2px 10px;
    }
    QCheckBox, QLabel {
        color: #ffffff;
    }
    QFrame {
        border: 1px solid #5a5a5a;
    }
"""

def set_dark_mode(app):
    app.setStyleSheet(DARK_MODE_STYLE)