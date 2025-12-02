import gradio as gr
import yaml

# Mod_RWKV
from Mod_RWKV.infer.worldmodel import Worldinfer
llm_path='./checkpoints/lm_weights/nonencoder'
encoder_path='./checkpoints/siglip2-base-patch16-384'
encoder_type='siglip' #[clip, whisper, siglip, speech]
mod_rwkv_model = Worldinfer(model_path=llm_path, encoder_type=encoder_type, encoder_path=encoder_path)

# img_path = './docs/03-Confusing-Pictures.jpg'
# image = Image.open(img_path).convert('RGB')
# text = '\x16User: Pleas discribe this image~\x17Assistant:'
# result,_ = mod_rwkv_model.generate(text, image)
# print(result)

# 创建Gradio界面
def create_interface():

    with gr.Blocks(title="Mod RWKV") as demo:
        demo.queue()

        with gr.Tab("RWKV LM Assistant"):
            gr.Markdown("""
            <h1 style='text-align: center; color: #1a73e8;'>RWKV 语言模型助手</h1>
            <p style='text-align: center; font-size: 16px;'>与RWKV语言模型进行对话。RWKV是一个开源的大语言模型，具有优秀的性能和效率。</p>
            """)

            with gr.Row():
                # 左侧添加图像输入区域
                with gr.Column(scale=2):
                    rwkv_image_input = gr.Image(type="pil", label="上传图片", height=300)
                    image_description = gr.Textbox(label="图像描述", interactive=False, lines=8, max_lines=10)
                    auto_describe_checkbox = gr.Checkbox(label="自动使用图像描述作为上下文", value=True)

                    def generate_image_description(image):
                        if image is not None:
                            text = '\x16User: Please describe this image\x17Assistant:'
                            result, _ = mod_rwkv_model.generate(text, image)
                            return result
                        return ""

                    # 当图像上传时自动生成描述
                    rwkv_image_input.change(
                        fn=generate_image_description,
                        inputs=[rwkv_image_input],
                        outputs=[image_description]
                    )

                # 右侧保持原有聊天界面
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        height="60vh",
                        label="对话记录",
                        elem_id="chatbot-container",
                        render_markdown=True
                    )       
                    with gr.Row():
                        msg = gr.Textbox(label="输入消息", placeholder="请输入您的问题...", scale=9)
                        send_btn = gr.Button("发送", variant="primary", scale=1)
                    with gr.Row():
                        clear_btn = gr.Button("清空对话")
                        examples = gr.Examples(
                            examples=["RWKV是什么，和Transfomer相比有什么区别?", "请鉴赏一下这幅图片", "这个图片描述了什么？"],
                            inputs=msg
                        )

            # 修改RWKV API调用函数
            def rwkv_chat(message, history, image, image_desc, use_auto_desc):
                import requests
                import json

                # API配置
                api_url = "http://127.0.0.1:8000/v4/chat/completions"

                # 构建消息内容
                with open('doc/chat_material.yaml', 'r', encoding='utf-8') as f:
                    corpora = yaml.safe_load(f)

                # 关键词映射到语料
                keyword_to_corpus = {
                    "bisrnet": ["bisrnet", "bi-srnet", "sscd", "语义变化检测", "双时相语义"],
                    "rs_basic": ["遥感", "rs", "ground sample distance", "gsd", "光谱"],
                    "rwkv":["rwkv","rnn","lm","线性注意力","receptance weighted key value", "RWKV"],
                    "general": []  # 默认语料
                }

                # 根据用户问题选择合适的语料
                def select_corpus(user_question):
                    user_question_lower = user_question.lower()
                    for corpus_key, keywords in keyword_to_corpus.items():
                        for keyword in keywords:
                            if keyword.lower() in user_question_lower:
                                return corpora[corpus_key]
                    # 默认返回general语料
                    return corpora["general"]

                # 动态选择语料
                selected_corpus = select_corpus(message)

                # 构建消息列表
                messages = [{"role": "user", "content": selected_corpus}]

                # 如果有图像且启用自动描述，则将其添加到上下文中
                if image is not None and use_auto_desc and image_desc:
                    messages.append({"role": "user", "content": f"图像描述信息: {image_desc}"})

                messages.append({"role": "user", "content": message})

                # 请求参数
                payload = {
                    "messages": messages,  # 只发送当前消息
                    "max_tokens": 8192,
                    "stop_tokens": [0, 261, 24281],
                    "temperature": 1.0,
                    "noise": 1.5,
                    "stream": True,
                    "enable_think": True,
                    "chunk_size": 8
                }

                headers = {
                    "Content-Type": "application/json"
                }

                try:
                    response = requests.post(api_url, data=json.dumps(payload), headers=headers, stream=True)
                    response.raise_for_status()

                    full_response = ""
                    thinking_content = ""
                    answer_content = ""
                    thinking_finished = False

                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data: "):
                                data_str = decoded_line[6:]  # 移除"data: "前缀
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    if "choices" in data and len(data["choices"]) > 0:
                                        delta = data["choices"][0].get("delta", {})
                                        content = delta.get("content", "")
                                        full_response += content

                                        # 处理思考过程和正文的分离
                                        if not thinking_finished:
                                            if "</think>" in content:
                                                # 找到思考结束标记
                                                parts = content.split("</think>", 1)
                                                thinking_content += parts[0]
                                                if len(parts) > 1:
                                                    answer_content += parts[1]
                                                thinking_finished = True
                                            else:
                                                # 仍在思考阶段
                                                thinking_content += content
                                        else:
                                            # 思考已完成，添加到答案内容
                                            answer_content += content

                                        # 构造显示内容：思考过程(如果有) + 正文
                                        display_content = ""
                                        if thinking_content or "</think>" in full_response:
                                            # 如果有思考内容或者已经结束思考，显示思考过程
                                            if thinking_content.strip():
                                                # 添加 open 属性使 details 默认展开
                                                display_content += f"<details open><summary>思考过程</summary>\n\n{thinking_content}\n\n</details>\n\n"
                                            display_content += "---\n"

                                        display_content += answer_content

                                        # 实时更新聊天界面，保持输入框内容不变
                                        # 保留完整历史记录在UI上显示
                                        yield message, history + [{"role": "user", "content": message}, {"role": "assistant", "content": display_content}]
                                except json.JSONDecodeError:
                                    continue
                                
                    # 最终完整响应，清空输入框
                    # 同样处理最终响应
                    display_content = ""
                    if thinking_content or "</think>" in full_response:
                        if thinking_content.strip():
                            # 添加 open 属性使 details 默认展开
                            display_content += f"<details open><summary>思考过程</summary>\n\n{thinking_content}\n\n</details>\n\n"
                        display_content += "---\n"

                    display_content += answer_content

                    yield "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": display_content}]
                except Exception as e:
                    error_msg = f"请求失败: {str(e)}"
                    yield "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": error_msg}]

            # 更新事件绑定
            msg.submit(rwkv_chat, [msg, chatbot, rwkv_image_input, image_description, auto_describe_checkbox], [msg, chatbot], queue=True) 
            send_btn.click(rwkv_chat, [msg, chatbot, rwkv_image_input, image_description, auto_describe_checkbox], [msg, chatbot], queue=True) 
            clear_btn.click(lambda: None, None, chatbot, queue=False)

    return demo

# 主函数
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Ocean())