import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QPushButton, QLabel, QTextEdit, QFileDialog, QSplitter, QFrame)
from PyQt5.QtCore import Qt, QMimeData
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PyQt5.QtCore import QSize
import texttophoto

class ImageLabel(QLabel):
    """支持拖拽上传的自定义图片标签"""
    def __init__(self, hint_text, parent=None):
        super().__init__(parent)
        self.hint_text = hint_text
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #F5F5F5;
                border: 2px dashed #D8D8D8;
                border-radius: 8px;
                color: #888888;
                font-size: 16px;
            }
        """)
        self.setText(hint_text)
        self.setMinimumSize(300, 200)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.setPixmap(QPixmap(file_path).scaled(
                    self.width(), self.height(), 
                    Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                self.parent().current_image_path = file_path

    def mousePressEvent(self, event):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", 
            "图片文件 (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.setPixmap(QPixmap(file_path).scaled(
                self.width(), self.height(), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.parent().current_image_path = file_path

class XiaohongshuStyleApp(QMainWindow):
    """小红书风格描述器主界面"""
    def __init__(self):
        super().__init__()
        self.current_image_path = ""
        self.generated_image_path = ""
        self.initUI()
        
    def initUI(self):
        # 主窗口设置
        self.setWindowTitle('小红书风格描述器')
        self.setGeometry(300, 200, 900, 600)
        self.setStyleSheet("background-color: #FFFFFF;")
        
        # 创建中央部件和主布局
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧区域 - 使用QSplitter实现上下分区
        left_splitter = QSplitter(Qt.Vertical)
        
        # 上部分：图片展示区
        image_frame = QFrame()
        image_layout = QHBoxLayout(image_frame)
        image_layout.setContentsMargins(5, 5, 5, 5)
        
        # 输入图片区域
        self.input_image = ImageLabel("拖拽图片或点击上传", parent=self)
        image_layout.addWidget(self.input_image)
        
        
        # 输出图片区域
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
        
        # 下部分：文本展示区
        self.text_display = QTextEdit()
        self.text_display.setPlaceholderText("生成的小红书风格描述将显示在此处...")
        self.text_display.setStyleSheet("""
            QTextEdit {
                background-color: #FFFBF0;
                border: 1px solid #FFE8C5;
                border-radius: 6px;
                padding: 10px;
                font-size: 14px;
                color: #333333;
            }
        """)
        
        # 将上下部分添加到splitter
        left_splitter.addWidget(image_frame)
        left_splitter.addWidget(self.text_display)
        left_splitter.setSizes([400, 200])
        
        # 右侧按钮区域
        button_frame = QFrame()
        button_layout = QVBoxLayout(button_frame)
        button_layout.setContentsMargins(10, 20, 10, 20)
        button_layout.setSpacing(15)  # 稍微减小间距以容纳更多按钮
        
        # 生成文本描述按钮
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
        
        # 修改为"生成清新风格图片"按钮
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
        
        # 新增"生成黑白漫画风格图片"按钮
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
        
        # 新增"生成卡通风格图片"按钮
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
        
        # 占位按钮（保持布局平衡）
        button_layout.addStretch(1)
        
        # 添加到主布局
        main_layout.addWidget(left_splitter, 4)
        main_layout.addWidget(button_frame, 1)
        
        # 设置中央部件
        self.setCentralWidget(central_widget)
    
    def generate_text_description(self):
        """生成小红书风格文本描述（外部函数接口）"""
        
        self.emoji_text = gettext(self.current_image_path)
        self.text_display.setPlainText(self.emoji_text)
    
    def generate_style_image(self, style_type):
        """生成指定风格的图片"""
        '''
        if not self.current_image_path:
            self.text_display.setPlainText("⚠️ 请先上传图片！")
            return
        '''
        # 根据风格类型显示不同提示
        style_names = {
            "fresh": "清新风格",
            "comic": "彩色铅笔风格",
            "cartoon": "吉卜力卡通风格"
        }
        
        style_hint = style_names.get(style_type, "指定风格")
        self.text_display.append(f"\n\n🖼️ 正在生成{style_hint}图片...请稍候！")
        prompt=f"请生成{style_hint}的图片，内容描述如下："+self.emoji_text
        # 实际生成图片的逻辑
        try:
            # 模拟生成成功后显示图片
            
            self.generated_image_path = str(texttophoto.send_image(prompt) ) # 生成图片的路径
            '''
            self.output_image.setPixmap(QPixmap(self.generated_image_path).scaled(
                self.output_image.width(), self.output_image.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))'''
            print(self.generated_image_path)
            pixmap = QPixmap(self.generated_image_path)
            
            if pixmap.isNull():
                print("Failed to load image from:", self.generated_image_path)
            # 进行错误处理，例如显示一个占位符图片或提示信息
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

def gettext(picturepath):
    """生成小红书风格文本描述（外部函数接口）"""
    # 示例文本
    emoji_text = "牛排的纹理清晰可见，每一块都煎得恰到好处，外焦里嫩，看着就让人垂涎欲滴。口水预警！🤤,**旁边的蔬菜也毫不逊色**，翠绿的西兰花和红艳艳的小番茄，不仅颜色好看，还给整道菜增添了生机。健康又美味，爱了爱了！**甜甜的南瓜片**在一旁静静陪伴，金黄的颜色和软糯的口感，绝对是牛排的最佳搭档。每一口都是幸福的味道呀！"
    print("生成的文本:", emoji_text)
    return emoji_text


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = XiaohongshuStyleApp()
    window.show()
    sys.exit(app.exec_())