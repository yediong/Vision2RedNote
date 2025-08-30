import sys
import torch
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QTextEdit, QFileDialog, QSplitter, QFrame)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QPainter
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)
import texttophoto

# --------------------------
# Qwen2VL 图生文模型加载
# --------------------------
MODEL_DIR = "./qwen2-finetuned0824_3/checkpoint-625"
BASE_MODEL_DIR = "Qwen2-VL-2B-Instruct"
processor = None
model = None

def load_model():
    global processor, model
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            BASE_MODEL_DIR,
            trust_remote_code=True
        )
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")

def generate_description(image_path: str) -> str:
    global processor, model
    if processor is None or model is None:
        return "错误：模型未加载完成，请稍后再试"
    try:
        image = Image.open(image_path).convert("RGB")
        if image.size[0] == 0 or image.size[1] == 0:
            return "错误：无效的图片文件"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "请用小红书风格描述这张图片，要活泼可爱，多用emoji表情！✨"}
                ]
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=processor.tokenizer.eos_token_id
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return output_text
    except Exception as e:
        return f"生成描述失败: {str(e)}"

# --------------------------
# PyQt5 UI
# --------------------------
class ImageLabel(QLabel):
    def __init__(self, hint_text, main_window, parent=None):
        super().__init__(parent)
        self.hint_text = hint_text
        self.main_window = main_window
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #F5F5F5;
                border: 2px dashed #D8D8D8;
                border-radius: 8px;
                color: #888888;
                font-size: 20px;
            }
        """)
        self.setText(hint_text)
        self.setMinimumSize(300, 200)
        self._has_image = False

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        self._has_image = True
        self.setText("")  # 清空提示文字

    def clearPixmap(self):
        super().clear()
        self._has_image = False
        self.setText(self.hint_text)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                pixmap = QPixmap(file_path).scaled(
                    self.width(), self.height(),
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.setPixmap(pixmap)
                self.main_window.current_image_path = file_path

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "图片文件 (*.png *.jpg *.jpeg)"
        )
        if file_path:
            pixmap = QPixmap(file_path).scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.setPixmap(pixmap)
            self.main_window.current_image_path = file_path

class XiaohongshuStyleApp(QMainWindow):
    """小红书风格描述器主界面"""
    def __init__(self):
        super().__init__()
        self.current_image_path = ""
        self.generated_image_path = ""
        self.emoji_text = ""
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('小红书风格描述器')
        #self.setGeometry(300, 200, 900, 600)
        self.setStyleSheet("background-color: #FFFFFF;")
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        left_splitter = QSplitter(Qt.Vertical)
        image_frame = QFrame()
        image_layout = QHBoxLayout(image_frame)
        image_layout.setContentsMargins(5, 5, 5, 5)
        self.input_image = ImageLabel("拖拽图片或点击上传", main_window=self)
        image_layout.addWidget(self.input_image)
        self.output_image = QLabel("生成图片将显示在此处")
        self.output_image.setAlignment(Qt.AlignCenter)
        self.output_image.setStyleSheet("""
            QLabel {
                background-color: #F9F9F9;
                border: 1px solid #E8E8E8;
                border-radius: 8px;
                color: #999999;
                font-size: 14px;
            }
        """)
        self.output_image.setMinimumSize(300, 200)
        image_layout.addWidget(self.output_image)
        self.text_display = QTextEdit()
        self.text_display.setPlaceholderText("生成的小红书风格描述将显示在此处...")
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: #FFFBF0;
                border: 1px solid #FFE8C5;
                border-radius: 6px;
                padding: 10px;
                font-size: 18px;
                color: #333333;
            }
        """)
        left_splitter.addWidget(image_frame)
        left_splitter.addWidget(self.text_display)
        left_splitter.setSizes([400, 200])
        button_frame = QFrame()
        button_layout = QVBoxLayout(button_frame)
        button_layout.setContentsMargins(10, 20, 10, 20)
        button_layout.setSpacing(15)
        self.text_btn = QPushButton("生成文本描述")
        self.text_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF2442;
                color: white;
                border-radius: 6px;
                padding: 12px 20px;
                font-size: 16px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #FF4760;
            }
        """)
        self.text_btn.clicked.connect(self.generate_text_description)
        button_layout.addWidget(self.text_btn)
        self.fresh_btn = QPushButton("生成清新风格")
        self.fresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A90E2;
                color: white;
                border-radius: 6px;
                padding: 12px 20px;
                font-size: 16px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #5C9AE6;
            }
        """)
        self.fresh_btn.clicked.connect(lambda: self.generate_style_image("fresh"))
        button_layout.addWidget(self.fresh_btn)
        self.comic_btn = QPushButton("生成彩铅风格")
        self.comic_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: white;
                border-radius: 6px;
                padding: 12px 20px;
                font-size: 16px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #555555;
            }
        """)
        self.comic_btn.clicked.connect(lambda: self.generate_style_image("comic"))
        button_layout.addWidget(self.comic_btn)
        self.cartoon_btn = QPushButton("生成卡通风格")
        self.cartoon_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF6B6B;
                color: white;
                border-radius: 6px;
                padding: 12px 20px;
                font-size: 16px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #FF8E8E;
            }
        """)
        self.cartoon_btn.clicked.connect(lambda: self.generate_style_image("cartoon"))
        button_layout.addWidget(self.cartoon_btn)
        button_layout.addStretch(1)
        main_layout.addWidget(left_splitter, 4)
        main_layout.addWidget(button_frame, 1)
        self.setCentralWidget(central_widget)

        big_btn_style = """
            QPushButton {
                background-color: %s;
                color: white;
                border-radius: 10px;
                padding: 20px 30px;
                font-size: 26px;
                font-weight: bold;
                min-width: 180px;
                min-height: 60px;
            }
            QPushButton:hover {
                background-color: %s;
            }
        """
        self.text_btn.setStyleSheet(big_btn_style % ("#FF2442", "#FF4760"))
        self.text_btn.setMinimumHeight(70)

        self.fresh_btn.setStyleSheet(big_btn_style % ("#4A90E2", "#5C9AE6"))
        self.fresh_btn.setMinimumHeight(70)

        self.comic_btn.setStyleSheet(big_btn_style % ("#333333", "#555555"))
        self.comic_btn.setMinimumHeight(70)

        self.cartoon_btn.setStyleSheet(big_btn_style % ("#FF6B6B", "#FF8E8E"))
        self.cartoon_btn.setMinimumHeight(70)
        # ...existing code...
        self.setCentralWidget(central_widget)
        self.showMaximized()  # 默认全屏
    
    def generate_text_description(self):
        """图生文：生成小红书风格文本描述"""
        if not self.current_image_path:
            self.text_display.setPlainText("⚠️ 请先上传图片！")
            return
        self.text_display.setPlainText("正在生成描述，请稍候...")
        desc = generate_description(self.current_image_path)
        self.emoji_text = desc
        self.text_display.setPlainText(desc)
    
    def generate_style_image(self, style_type):
        """文生图：生成指定风格的图片"""
        if not self.emoji_text:
            self.text_display.setPlainText("⚠️ 请先生成文本描述！")
            return
        style_names = {
            "fresh": "清新风格",
            "comic": "彩色铅笔风格",
            "cartoon": "吉卜力卡通风格"
        }
        style_hint = style_names.get(style_type, "指定风格")
        self.text_display.append(f"\n\n🖼️ 正在生成{style_hint}图片...请稍候！")
        prompt = f"请生成{style_hint}的图片，内容描述如下：" + self.emoji_text
        try:
            self.generated_image_path = str(texttophoto.send_image(prompt))
            pixmap = QPixmap(self.generated_image_path)
            if pixmap.isNull():
                self.text_display.append("❌ 图片生成失败，未能加载图片文件。")
                return
            scaled_pixmap = pixmap.scaled(
                QSize(self.output_image.width(), self.output_image.height()),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.output_image.setPixmap(scaled_pixmap)
            self.text_display.append(f"✅ {style_hint}图片生成成功！")
        except Exception as e:
            self.text_display.append(f"❌ 生成{style_hint}图片时出错: {str(e)}")
            print(str(e))

if __name__ == '__main__':
    load_model()
    app = QApplication(sys.argv)
    window = XiaohongshuStyleApp()
    window.show()
    sys.exit(app.exec_())